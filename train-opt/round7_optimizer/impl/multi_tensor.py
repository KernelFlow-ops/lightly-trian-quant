"""Three-launch CUDA AdamW with cached metadata and original optimizer state."""
from __future__ import annotations

import os
from pathlib import Path

import torch

_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault("MAX_JOBS", "6")
        root = Path(__file__).resolve().parent
        cache = Path('/root/autodl-tmp/train-opt-tmp/C/multi_tensor_build')
        cache.mkdir(parents=True, exist_ok=True)
        _EXTENSION = load(name='trainopt_multi_tensor', sources=[str(root / 'multi_tensor.cu')],
                          build_directory=str(cache), extra_cuda_cflags=['-O3', '-lineinfo', '--fmad=false'],
                          extra_cflags=['-O3'])
    return _EXTENSION


class MultiTensorAdamW:
    """Parameters with missing gradients retain their independent step counts.

    Metadata is rebuilt in eager mode only when gradient/state addresses change.
    CUDA graph warmup prepares it; capture and replay perform no host copies.
    """
    def __init__(self, optimizer, *, shadows=None):
        self.optimizer = optimizer
        self.shadows = shadows or {}
        self.signature = None
        self.buffers = None
        self.module = extension()

    def _prepare(self):
        rows, hyper, sizes, tensors = [], [], [], []
        for group in self.optimizer.param_groups:
            if group.get('amsgrad') or group.get('maximize'):
                raise ValueError('multi_tensor supports ordinary AdamW only')
            for p in group['params']:
                if p.grad is None or p.numel() == 0:
                    continue
                g = p.grad
                if not p.is_cuda or p.dtype != torch.float32 or g.dtype != torch.float32 or g.is_sparse:
                    raise ValueError('multi_tensor requires dense FP32 CUDA master weights and gradients')
                if not p.is_contiguous() or not g.is_contiguous():
                    raise ValueError('multi_tensor requires contiguous parameters/gradients')
                state = self.optimizer.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] = torch.zeros((), device=p.device, dtype=torch.float32)
                if state['step'].device != p.device:
                    state['step'] = state['step'].to(device=p.device, dtype=torch.float32)
                lr = group['lr']
                if not isinstance(lr, torch.Tensor) or lr.device != p.device:
                    raise ValueError('multi_tensor requires device learning rate tensors')
                shadow = self.shadows.get(id(p))
                values = [p, g, state['exp_avg'], state['exp_avg_sq'], state['step'], lr]
                rows.append([v.data_ptr() for v in values] + [0 if shadow is None else shadow.data_ptr()])
                hyper.append([*group['betas'], group['weight_decay'], group['eps']])
                tensor_id = len(rows) - 1
                sizes.append(p.numel())
                tensors.extend(values)
                if shadow is not None:
                    tensors.append(shadow)
        signature = tuple(tuple(v) for v in rows), tuple(tuple(v) for v in hyper)
        if signature != self.signature:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('multi_tensor metadata changed during capture; warm up with all required gradients')
            device = tensors[0].device if tensors else self.optimizer.param_groups[0]['params'][0].device
            chunks = [[i, offset, min(4096, size - offset)] for i, size in enumerate(sizes) for offset in range(0, size, 4096)]
            self.buffers = (
                torch.tensor(rows, device=device, dtype=torch.int64).reshape(-1, 7),
                torch.tensor(chunks, device=device, dtype=torch.int64).reshape(-1, 3),
                torch.tensor(hyper, device=device, dtype=torch.float32).reshape(-1, 4),
                torch.empty(len(chunks), device=device), torch.empty((len(rows), 4), device=device),
                torch.zeros((), device=device),
            )
            self.signature = signature
            self.tensors = tensors  # Keep all graph-referenced storage alive.

    @torch.no_grad()
    def step(self, max_norm):
        self._prepare()
        self.module.step(*self.buffers, max_norm)
        # Preserve the scheduler's optimizer-before-scheduler ordering contract.
        self.optimizer._opt_called = True
        return self.buffers[-1]
