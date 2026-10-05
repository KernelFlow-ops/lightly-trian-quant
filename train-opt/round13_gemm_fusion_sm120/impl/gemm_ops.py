"""GEMM layer of the round13 fused block: bf16 operands, fp32 accumulation, fused epilogues.

Layouts (row-major torch tensors; ``x [M,K]``, ``w [N,K]`` = nn.Linear weight, ``dy [M,N]``):

  fwd   y  [M,N] = x  @ w^T          epilogues: bias | bias + gamma*residual (fp32 out) | bias + GELU(erf)
  dgrad dx [M,K] = dy @ w            epilogue : none | * GELU'(z) with fused bias-grad column sum
  wgrad dw [N,K] = dy^T @ x          fp32 output, written straight to the gradient tensor

Backends
  quack : QuACK 0.6.5 SM120 GEMM (CuTe DSL, TMA + warp-level MMA, persistent dynamic scheduler) with
          ``@gemm_epilogue`` custom epilogues (the fn frontend); autotuned once per (shape, epilogue) and cached on disk.
  torch : cuBLAS/cuBLASLt via ``torch.mm`` (``out_dtype=torch.float32`` for wgrad)

The per-(kind, linear) choice is ``POLICY`` below (filled from the round13 candidate matrix).  Everything is
launched on the current stream, allocation goes through the caching allocator and JIT/autotune happen on the
first (warm-up) call, so the block is CUDA-graph capturable after one eager step (CONTRACTS section 5).

MXFP8 readiness: operands enter through ``_a``/``_b`` views only; a block-scaled operand container (QuACK
``BlockScaledOperand``) can replace the bf16 tensors for the same call signatures.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")

import torch

_QUACK = None


def _quack():
    """Lazy import so that CPU-only environments can import the package."""
    global _QUACK
    if _QUACK is None:
        from quack.activation import dgelu_erf, gelu_erf
        from quack.epilogue import gemm_epilogue
        from quack.epilogue.ops import RowVecReduce
        from quack.gemm_interface import gemm as quack_gemm

        @gemm_epilogue()
        def bias_epi(acc, bias):
            return {"D": acc + bias}

        @gemm_epilogue(outputs=("y",))
        def res_epi(acc, c, bias, gamma):
            # y = acc + bias (bf16, saved for dGamma);  D = resid + gamma * y (fp32 residual stream)
            y = acc + bias
            return {"D": c + gamma * y, "y": y}

        @gemm_epilogue(outputs=("h",))
        def gelu_epi(acc, bias):
            # D = z = acc + bias (pre-activation, bf16, kept for GELU');  h = GELU_erf(z) (bf16)
            z = acc + bias
            return {"D": z, "h": gelu_erf(z)}

        @gemm_epilogue(reduces={"db": RowVecReduce("db")})
        def dgelu_epi(acc, c):
            # acc = dh = dy @ W2; c = z.  dz = dh * GELU'_erf(z); db = sum_m dz (bias gradient of fc1)
            dz, _ = dgelu_erf(c, acc)
            return {"D": dz, "db": dz}

        _QUACK = dict(gemm=quack_gemm, bias=bias_epi, res=res_epi, gelu=gelu_epi, dgelu=dgelu_epi)
    return _QUACK


# (kind, linear) -> backend.  kind in {"fwd", "dgrad", "wgrad"}; linear in {"qkv", "proj", "fc1", "fc2"}.
# Filled from profile/gemm_matrix_final.json (see NOTES section 4); "quack" unless torch/cuBLAS was faster.
DEFAULT_POLICY = {
    ("fwd", "qkv"): "quack", ("fwd", "proj"): "quack", ("fwd", "fc1"): "quack", ("fwd", "fc2"): "quack",
    ("dgrad", "qkv"): "quack", ("dgrad", "proj"): "quack", ("dgrad", "fc1"): "quack", ("dgrad", "fc2"): "quack",
    ("wgrad", "qkv"): "auto", ("wgrad", "proj"): "auto", ("wgrad", "fc1"): "auto", ("wgrad", "fc2"): "auto",
}
# wgrad contracts over M = B*N tokens (7203 / 14406, not a multiple of the 128 tile and a poor fit for 170 SMs), so the
# best backend depends on the output size: cuBLAS (split-K heuristics) vs QuACK (plain / autotuned split-K).
# Picked from the same-machine candidate matrix (profile/gemm_matrix_*.json, NOTES section 4.1); keyed by hidden size.
WGRAD_AUTO = {
    1024: {"qkv": "quack", "proj": "quack", "fc1": "torch", "fc2": "torch"},
    768: {"qkv": "quack_sk", "proj": "torch", "fc1": "torch", "fc2": "quack_sk"},
}
POLICY = dict(DEFAULT_POLICY)
if os.environ.get("R13_GEMM_BACKEND"):  # ablation switch: force one backend for every GEMM
    POLICY = {k: os.environ["R13_GEMM_BACKEND"] for k in POLICY}


def _row(v: torch.Tensor) -> torch.Tensor:
    return v.reshape(1, -1)


# ------------------------------------------------------------------------------------------------ forward
def linear_bias(x, w, b, name="qkv"):
    """bf16 ``y = x @ w^T + b`` (b fp32 [N] or None)."""
    M, N = x.shape[0], w.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    if POLICY[("fwd", name)] == "quack":
        q = _quack()
        if b is None:
            q["gemm"](x, w.t(), out=y)
        else:
            q["bias"](x, w.t(), out={"D": y}, bias=_row(b))
        return y
    torch.mm(x, w.t(), out=y)
    if b is not None:
        y += b.to(torch.bfloat16)
    return y


def linear_res(x, w, b, gamma, resid, name):
    """``y = x @ w^T + b`` (bf16, saved);  ``out = resid + gamma * y`` (fp32 residual stream).  Returns (out, y)."""
    M, N = x.shape[0], w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    if POLICY[("fwd", name)] == "quack":
        _quack()["res"](x, w.t(), resid, out={"D": out, "y": y}, bias=_row(b), gamma=_row(gamma))
        return out, y
    yf = torch.mm(x, w.t()).float() + b
    y.copy_(yf)
    torch.addcmul(resid, yf, gamma, out=out)
    return out, y


def linear_gelu(x, w, b):
    """fc1: ``z = x @ w^T + b`` (bf16 pre-activation) and ``h = GELU_erf(z)`` (bf16).  Returns (z, h)."""
    M, N = x.shape[0], w.shape[0]
    z = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    h = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    if POLICY[("fwd", "fc1")] == "quack":
        _quack()["gelu"](x, w.t(), out={"D": z, "h": h}, bias=_row(b))
        return z, h
    zf = torch.mm(x, w.t()).float() + b
    z.copy_(zf)
    h.copy_(torch.nn.functional.gelu(zf))
    return z, h


# ------------------------------------------------------------------------------------------------ backward
def dgrad(dy, w, name):
    """``dx = dy @ w`` (bf16)."""
    M, K = dy.shape[0], w.shape[1]
    dx = torch.empty(M, K, device=dy.device, dtype=torch.bfloat16)
    if POLICY[("dgrad", name)] == "quack":
        _quack()["gemm"](dy, w, out=dx)
    else:
        torch.mm(dy, w, out=dx)
    return dx


def dgrad_dgelu(dy, w, z):
    """fc2 dgrad with the GELU' epilogue: ``dz = (dy @ w) * GELU'(z)`` (bf16) and ``db = sum_m dz`` (fp32 [4d])."""
    M, N = dy.shape[0], w.shape[1]
    dz = torch.empty(M, N, device=dy.device, dtype=torch.bfloat16)
    if POLICY[("dgrad", "fc2")] == "quack":
        res = _quack()["dgelu"](dy, w, z, out={"D": dz})
        return dz, res["db"].reshape(-1)
    dh = torch.mm(dy, w).float()
    zf = z.float().requires_grad_()
    with torch.enable_grad():
        dzf = torch.autograd.grad(torch.nn.functional.gelu(zf), zf, dh)[0]
    dz.copy_(dzf)
    return dz, dzf.sum(0)


def wgrad_backend(name, N, K):
    b = POLICY[("wgrad", name)]
    if b == "auto":
        b = WGRAD_AUTO.get(min(N, K), {}).get(name, "torch")
    return b


def wgrad(dy, x, name):
    """``dw = dy^T @ x`` in fp32 (the Parameter-gradient dtype), fp32 accumulation."""
    N, K = dy.shape[1], x.shape[1]
    backend = wgrad_backend(name, N, K)
    if backend in ("quack", "quack_sk"):
        dw = torch.empty(N, K, device=dy.device, dtype=torch.float32)
        if backend == "quack_sk":  # autotuned split-K (serial mode: deterministic)
            _quack()["gemm"](dy.t(), x, out=dw, split_k=None)
        else:
            _quack()["gemm"](dy.t(), x, out=dw)
        return dw
    return torch.mm(dy.t(), x, out_dtype=torch.float32)
