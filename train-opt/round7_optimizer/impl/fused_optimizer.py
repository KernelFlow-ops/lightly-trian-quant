"""Foreach gradient clipping used by the capturable AdamW patch.

The optimizer itself is PyTorch's fused AdamW kernel.  This helper keeps the
global norm and clipping factor on device and applies one foreach operation to
the gradients, so the clip path adds no Python scalar synchronization.
"""
from __future__ import annotations

import torch
from torch import Tensor


def fused_clip_grad_norm_(parameters, max_norm: float, eps: float = 1e-6) -> Tensor:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        if params:
            return torch.zeros((), device=params[0].device)
        return torch.tensor(0.0)
    grads = [p.grad for p in params]
    # _foreach_norm computes one scalar per tensor without copying gradients to
    # the host.  A device reduction preserves the usual L2 global norm.
    norms = torch._foreach_norm(grads, 2.0)
    total = torch.linalg.vector_norm(torch.stack(tuple(norms)), ord=2)
    if max_norm > 0:
        scale = torch.as_tensor(max_norm, dtype=total.dtype, device=total.device)
        scale = (scale / (total + eps)).clamp(max=1.0)
        torch._foreach_mul_(grads, scale)
    return total
