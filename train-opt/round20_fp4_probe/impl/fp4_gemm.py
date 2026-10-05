"""Real QuACK FP4 GEMMs on ``fp4_ref.FP4Tensor`` operands (nv4 / mx4), all TN with K-major operands (CONTRACTS 11.2).

    fwd    Y  [M,N] = X.q  . W.q^T            X row copy (along K), W row copy (along K)
    dgrad  dX [M,K] = dY.q . W.qt^T           dY row copy (along N), W^T copy (along N)
    wgrad  dW [N,K] = dY.qt . X.qt^T  (fp32)  dY^T and X^T copies (along the zero padded token axis)

Epilogues: nv4 uses ``fp4_epi.nv4_epilogues()`` with ``ts = ts_a * ts_b`` as a device scalar; mx4 has no per-tensor scale and uses round13's
epilogues unchanged.  Plain products go through the QuACK gemm with ``BlockScaledOperand`` containers (per-tensor scale folded in as alpha).
All calls are launched on the current stream (capturable after the first call: JIT / autotune are cached on disk).
"""
from __future__ import annotations

import torch

from . import fp4_cands as C
from . import fp4_epi

_E13 = None


def _e13():
    global _E13
    if _E13 is None:
        _E13 = C.epilogues()
    return _E13


def _f4(q: torch.Tensor) -> torch.Tensor:
    return q.view(torch.float4_e2m1fn_x2)


def _row(v: torch.Tensor) -> torch.Tensor:
    return v.reshape(1, -1)


def _ts(fmt, a, b):
    return (a * b).reshape(1) if fmt == "nv4" else None


def _epi(fmt: str):
    return fp4_epi.nv4_epilogues() if fmt == "nv4" else _e13()


def _ops(fmt, qa, sfa, qb, sfb, tsa, tsb):
    """(A [M,K/2] f4x2, B^T view, epilogue kwargs, ts) from raw copies."""
    return _f4(qa), _f4(qb).t(), C.epi_kw(sfa, sfb, fmt), _ts(fmt, tsa, tsb)


def plain(fmt, qa, sfa, tsa, qb, sfb, tsb, out):
    """``out[M,N] = A . B^T`` (A [M,K/2], B [N,K/2] raw copies) through the QuACK block-scaled gemm."""
    C.quack_fp4(C.operand((qa, sfa, tsa), fmt), C.operand((qb, sfb, tsb), fmt), out)
    return out


# ------------------------------------------------------------------------------------------------ forward  (x.q row copy, w.q row copy)
def fwd_plain(x, w):
    y = torch.empty(x.rows, w.rows, device=x.q.device, dtype=torch.bfloat16)
    return plain(x.fmt, x.q, x.sf, x.ts, w.q, w.sf, w.ts, y)


def fwd_bias(x, w, bias):
    if bias is None:
        return fwd_plain(x, w)
    fmt = x.fmt
    y = torch.empty(x.rows, w.rows, device=x.q.device, dtype=torch.bfloat16)
    qa, qb, kw, ts = _ops(fmt, x.q, x.sf, w.q, w.sf, x.ts, w.ts)
    kwe = {"ts": ts} if ts is not None else {}
    _epi(fmt)["bias"](qa, qb, out={"D": y}, bias=_row(bias), **kwe, **kw)
    return y


def fwd_res(x, w, bias, gamma, resid):
    fmt = x.fmt
    M, N = x.rows, w.rows
    out = torch.empty(M, N, device=x.q.device, dtype=torch.float32)
    y = torch.empty(M, N, device=x.q.device, dtype=torch.bfloat16)
    qa, qb, kw, ts = _ops(fmt, x.q, x.sf, w.q, w.sf, x.ts, w.ts)
    kwe = {"ts": ts} if ts is not None else {}
    _epi(fmt)["res"](qa, qb, resid, out={"D": out, "y": y}, bias=_row(bias), gamma=_row(gamma), **kwe, **kw)
    return out, y


def fwd_gelu(x, w, bias):
    fmt = x.fmt
    M, N = x.rows, w.rows
    z = torch.empty(M, N, device=x.q.device, dtype=torch.bfloat16)
    h = torch.empty(M, N, device=x.q.device, dtype=torch.bfloat16)
    qa, qb, kw, ts = _ops(fmt, x.q, x.sf, w.q, w.sf, x.ts, w.ts)
    kwe = {"ts": ts} if ts is not None else {}
    _epi(fmt)["gelu"](qa, qb, out={"D": z, "h": h}, bias=_row(bias), **kwe, **kw)
    return z, h


# ------------------------------------------------------------------------------------------------ backward
def dgrad(dy, w):
    """dX [M, K] bf16 = dY.q . W.qt^T  (dy: row copy along N; w: transposed copy [K, N/2])."""
    dx = torch.empty(dy.rows, w.cols, device=dy.q.device, dtype=torch.bfloat16)
    return plain(dy.fmt, dy.q, dy.sf, dy.ts, w.qt, w.sft, w.tst, dx)


def dgrad_dgelu(dy, w, z):
    """fc2 dgrad with the GELU' epilogue: ``dz = (dy @ w) * GELU'(z)`` (bf16), ``db = sum_m dz`` (fp32 [4d])."""
    fmt = dy.fmt
    dz = torch.empty(dy.rows, w.cols, device=dy.q.device, dtype=torch.bfloat16)
    qa, qb, kw, ts = _ops(fmt, dy.q, dy.sf, w.qt, w.sft, dy.ts, w.tst)
    kwe = {"ts": ts} if ts is not None else {}
    res = _epi(fmt)["dgelu"](qa, qb, z, out={"D": dz}, **kwe, **kw)
    return dz, res["db"].reshape(-1)


def wgrad(dy, x):
    """dW [N, K] fp32 = dY^T . X over the (zero padded) token axis: dy.qt [N, Mp/2], x.qt [K, Mp/2]."""
    assert dy.rpad == x.rpad, (dy.rpad, x.rpad)
    dw = torch.empty(dy.cols, x.cols, device=dy.qt.device, dtype=torch.float32)
    return plain(dy.fmt, dy.qt, dy.sft, dy.tst, x.qt, x.sft, x.tst, dw)
