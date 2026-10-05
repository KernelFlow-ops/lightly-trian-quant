"""Python side of the round13 elementwise kernels (``block_kernels.cu``), JIT-built into the tmp build dir."""
from __future__ import annotations

import os
from pathlib import Path

import torch

_EXT = None
_HERE = Path(__file__).resolve().parent
_BUILD = Path("/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/build/block_kernels")
SUPPORTED_D = (384, 768, 1024)
GRID_MULT = int(os.environ.get("R13_GRID_MULT", "1"))


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("MAX_JOBS", "8")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _BUILD.mkdir(parents=True, exist_ok=True)
        _EXT = load(
            name="trainopt_r13_block_kernels",
            sources=[str(_HERE / "block_kernels.cu")],
            build_directory=str(_BUILD),
            extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17", "-Xptxas", "-v"],
            verbose=bool(int(os.environ.get("R13_VERBOSE_BUILD", "0"))),
        )
    return _EXT


_EMPTY = {}


def _empty(device, dtype=torch.float32):
    key = (device, dtype)
    if key not in _EMPTY:
        _EMPTY[key] = torch.empty(0, device=device, dtype=dtype)
    return _EMPTY[key]


def ln_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    """fp32 ``x[M,D]`` -> (bf16 ``y``, mean, rstd)."""
    y, mean, rstd = extension().ln_fwd(x, weight, bias, float(eps))
    return y, mean, rstd


def ln_bwd_res(x, mean, rstd, weight, dy, dres, y=None, gamma=None):
    """LayerNorm backward + residual-gradient add (+ optional ``g = gamma * dx`` branch prep).

    Returns ``(dx fp32, g bf16 or None, sums)`` where ``sums`` is ``[K, D]`` fp32 with rows
    ``[dW, dB]`` and, when ``gamma`` is given, ``[dGamma, dBiasBranch]``.
    """
    has_gam = gamma is not None
    e = _empty(x.device)
    eb = _empty(x.device, torch.bfloat16)
    dx, g, out = extension().rowbwd(x, mean, rstd, weight, dy, dres, y if has_gam else eb, gamma if has_gam else e, True, has_gam, GRID_MULT)
    return dx, (g if has_gam else None), out


def gamma_prep(dres, y, gamma):
    """``g = gamma * dres`` (bf16) and the column sums ``[dGamma, dBiasBranch]`` (fc2 / residual-branch prep)."""
    e = _empty(dres.device)
    eb = _empty(dres.device, torch.bfloat16)
    _, g, out = extension().rowbwd(e, e, e, e, eb, dres, y, gamma, False, True, GRID_MULT)
    return g, out


def colsum_bf16(x: torch.Tensor) -> torch.Tensor:
    return extension().colsum_bf16(x)
