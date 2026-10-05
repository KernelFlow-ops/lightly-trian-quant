"""True CUDA C++ packed-QKV attention reference path.

The extension is compiled lazily into the per-agent train-opt temporary
directory.  It intentionally keeps a small deterministic implementation so
that every input gradient can be compared against PyTorch before any tuned
kernel replaces it.
"""
from __future__ import annotations

import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

_EXT = None


def _extension():
    global _EXT
    if _EXT is None:
        root = Path(__file__).resolve().parents[2]
        src = Path(__file__).with_name("cuda_attention.cu")
        cache = Path("/root/autodl-tmp/train-opt-tmp/B/cuda_attention_build_v2")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MAX_JOBS", "6")
        _EXT = load(
            name="train_opt_cuda_attention_v2",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_cflags=["-O3"],
            build_directory=str(cache),
            verbose=False,
        )
    return _EXT


class _CudaAttention(torch.autograd.Function):
    _fast = False

    @classmethod
    def bind(cls, fast: bool):
        cls._fast = bool(fast)
        return cls

    @staticmethod
    def forward(ctx, qkv, sin, cos, compact_mask, prefix: int):
        assert qkv.is_cuda and qkv.is_contiguous()
        if sin is None:
            sin = qkv.new_empty((0,), dtype=torch.float32)
            cos = qkv.new_empty((0,), dtype=torch.float32)
        if compact_mask is None:
            compact_mask = torch.empty((0,), device=qkv.device, dtype=torch.bool)
        sin = sin.contiguous()
        cos = cos.contiguous()
        compact_mask = compact_mask.contiguous()
        Q = 0 if compact_mask.numel() == 0 else int(compact_mask.shape[1])
        ext = _extension()
        if _CudaAttention._fast and hasattr(ext, "forward_fast"):
            out, lse = ext.forward_fast(qkv, sin, cos, compact_mask, int(prefix), Q)
        else:
            out, lse = ext.forward(qkv, sin, cos, compact_mask, int(prefix), Q)
        ctx.save_for_backward(qkv, sin, cos, compact_mask, out, lse)
        ctx.prefix = int(prefix)
        ctx.fast = bool(_CudaAttention._fast)
        return out.reshape(qkv.shape[0], qkv.shape[1], qkv.shape[3] * 64), lse.reshape(qkv.shape[0] * qkv.shape[3], qkv.shape[1])

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        qkv, sin, cos, compact_mask, out, lse = ctx.saved_tensors
        grad_out = grad_out.contiguous().reshape_as(out)
        ext = _extension()
        Q = 0 if compact_mask.numel() == 0 else int(compact_mask.shape[1])
        if getattr(ctx, "fast", False) and hasattr(ext, "backward_fast"):
            dqkv = ext.backward_fast(qkv, sin, cos, compact_mask, out, lse, grad_out, ctx.prefix, Q)
        else:
            dqkv = ext.backward(qkv, sin, cos, compact_mask, out, lse, grad_out, ctx.prefix, Q)
        return dqkv, None, None, None, None


def attention(qkv, rope=None, compact_mask=None, prefix=None, return_lse=False, fast=False):
    if prefix is None:
        prefix = 0 if rope is None else qkv.shape[1] - rope[0].shape[0]
    sin, cos = (None, None) if rope is None else rope
    fn = _CudaAttention.bind(bool(fast))
    out, lse = fn.apply(qkv, sin, cos, compact_mask, int(prefix))
    return (out, lse) if return_lse else out
