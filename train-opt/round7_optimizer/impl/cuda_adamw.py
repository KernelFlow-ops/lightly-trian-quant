"""Lazy loader and Python adapter for the real CUDA AdamW kernel."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

_EXT: Any | None = None


def load_extension() -> Any:
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_adamw requires CUDA")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    cache = Path("/root/autodl-tmp/train-opt-tmp/C/cuda_adamw_build")
    cache.mkdir(parents=True, exist_ok=True)
    _EXT = load(
        name="trainopt_cuda_adamw",
        sources=[str(root / "cuda_adamw.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_cflags=["-O3"],
        build_directory=str(cache),
        verbose=False,
    )
    return _EXT


def fused_adamw_step(
    params: list[torch.Tensor], grads: list[torch.Tensor],
    exp_avg: list[torch.Tensor], exp_avg_sq: list[torch.Tensor],
    steps: list[torch.Tensor], lrs: list[torch.Tensor], wds: list[torch.Tensor],
    shadows: list[torch.Tensor] | None, max_norm: float,
    beta1: float, beta2: float, eps: float,
) -> torch.Tensor:
    if not params:
        return torch.zeros((), device="cuda")
    return load_extension().fused_adamw_step(
        params, grads, exp_avg, exp_avg_sq, steps, lrs, wds,
        [] if shadows is None else shadows,
        max_norm, beta1, beta2, eps,
    )
