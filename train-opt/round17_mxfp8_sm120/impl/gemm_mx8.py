"""MXFP8 GEMM layer of the round17 block: E4M3 + UE8M0 (1x32) operands, fp32 accumulation, round13's fused epilogues.

Operand layouts (see ``mx8.py``; every scale block runs along the contraction axis)::

    fwd    Y  [M,N] = X [M,K]  @ W^T          A = X.a_row()      B = W.b_row_t()      (K-major both)
    dgrad  dX [M,K] = dY [M,N] @ W            A = dY.a_row()     B = W.b_col()        (N-major B: ldmatrix.trans.b8)
    wgrad  dW [N,K] = dY^T @ X  (fp32 out)    A = dY.a_col_t()   B = X.b_col()        (M-major A, N-major B, K = padded tokens)

Backend: QuACK 0.6.5 SM120 block-scaled GEMM (``mma.sync kind::mxf8f6f4.block_scale``, the only FP8 path at full rate on the
5090).  The epilogues are round13's (bias | bias + gamma + fp32 residual | bias + GELU(erf) | GELU' with fused bias-gradient
column sum); they take the block-scaled operands through the ``SFA/SFB/bs_format_*`` arguments of the epilogue frontend.
All calls are launched on the current stream; autotune/JIT happen on the first call (warm-up step).
"""
from __future__ import annotations

import os

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")

import torch

from . import mx8

_Q = None
WGRAD_SPLITK = os.environ.get("R17_WGRAD_SPLITK", "none")  # "none" | "auto" (autotuned serial split-K)


def _q():
    global _Q
    if _Q is None:
        from quack.gemm_interface import _prep_blockscaled, _unpack_operand
        from quack.gemm_interface import gemm as quack_gemm

        from round13_gemm_fusion_sm120.impl import gemm_ops as g13

        _Q = dict(gemm=quack_gemm, unpack=_unpack_operand, prep=_prep_blockscaled, epi=g13._quack())
    return _Q


def _kw(sfa, sfb):
    n = mx8.fmt_name()
    return {"SFA": sfa, "SFB": sfb, "bs_format_a": n, "bs_format_b": n}


def _row(v: torch.Tensor) -> torch.Tensor:
    return v.reshape(1, -1)


# ------------------------------------------------------------------------------------------------ forward
def fwd_bias(xq: mx8.MX8, wq: mx8.MX8, bias):
    """bf16 ``y = x @ w^T + b`` (bias fp32 [N] or None)."""
    y = torch.empty(xq.rows, wq.rows, device=xq.qr.device, dtype=torch.bfloat16)
    q = _q()
    if bias is None:
        q["gemm"](xq.a_row(), wq.b_row_t(), out=y)
    else:
        q["epi"]["bias"](xq.qr, wq.qr.t(), out={"D": y}, bias=_row(bias), **_kw(xq.sfr_v, wq.sfr_v))
    return y


def fwd_res(xq, wq, bias, gamma, resid):
    """``y = x @ w^T + b`` (bf16, saved);  ``out = resid + gamma * y`` (fp32 residual stream).  Returns (out, y)."""
    M, N = xq.rows, wq.rows
    out = torch.empty(M, N, device=xq.qr.device, dtype=torch.float32)
    y = torch.empty(M, N, device=xq.qr.device, dtype=torch.bfloat16)
    _q()["epi"]["res"](xq.qr, wq.qr.t(), resid, out={"D": out, "y": y}, bias=_row(bias), gamma=_row(gamma), **_kw(xq.sfr_v, wq.sfr_v))
    return out, y


def fwd_gelu(xq, wq, bias):
    """fc1: ``z = x @ w^T + b`` (bf16) and ``h = GELU_erf(z)`` (bf16).  Returns (z, h)."""
    M, N = xq.rows, wq.rows
    z = torch.empty(M, N, device=xq.qr.device, dtype=torch.bfloat16)
    h = torch.empty(M, N, device=xq.qr.device, dtype=torch.bfloat16)
    _q()["epi"]["gelu"](xq.qr, wq.qr.t(), out={"D": z, "h": h}, bias=_row(bias), **_kw(xq.sfr_v, wq.sfr_v))
    return z, h


# ------------------------------------------------------------------------------------------------ backward
def dgrad(dyq: mx8.MX8, wq: mx8.MX8):
    """``dx = dy @ w`` (bf16 [M, K])."""
    dx = torch.empty(dyq.rows, wq.cols, device=dyq.qr.device, dtype=torch.bfloat16)
    _q()["gemm"](dyq.a_row(), wq.b_col(), out=dx)
    return dx


def dgrad_dgelu(dyq, wq, z):
    """fc2 dgrad with the GELU' epilogue: ``dz = (dy @ w) * GELU'(z)`` (bf16) and ``db = sum_m dz`` (fp32 [4d])."""
    dz = torch.empty(dyq.rows, wq.cols, device=dyq.qr.device, dtype=torch.bfloat16)
    res = _q()["epi"]["dgelu"](dyq.qr, wq.qc, z, out={"D": dz}, **_kw(dyq.sfr_v, wq.sfc_v))
    return dz, res["db"].reshape(-1)


def wgrad(dyq: mx8.MX8, xq: mx8.MX8):
    """``dw = dy^T @ x`` in fp32 (the Parameter-gradient dtype); the contraction runs over the zero-padded token axis."""
    assert dyq.mpad == xq.mpad and dyq.qc is not None and xq.qc is not None
    dw = torch.empty(dyq.cols, xq.cols, device=dyq.qc.device, dtype=torch.float32)
    A, B = dyq.a_col_t(), xq.b_col()
    if WGRAD_SPLITK == "auto":
        _q()["gemm"](A, B, out=dw, split_k=None)
    else:
        _q()["gemm"](A, B, out=dw)
    return dw
