"""Round 10 GEMM candidates, part 1: r5_v1 (bf16, fused epilogue; the current best) and cuBLAS bf16 / fp16 / fp16-accumulate (no epilogue)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from bench10 import MODELS, ROOT, Cand, R_FOR

for _p in (ROOT, ROOT / "round5_gemm_sm120"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BF, F16 = torch.bfloat16, torch.float16
_R5 = None


def r5():
    global _R5
    if _R5 is None:
        from round5_gemm_sm120.impl import ops as o  # noqa: E402  (read-only import; tuned_cfg.json + cubin cache in infer-opt-tmp/gemm120)

        _R5 = o
    return _R5


def _batch(M: int) -> tuple[int, int]:
    """(B crops, tokens per crop) for the GEMM row count M (2 crops per frame; 4 for the 2-frame batch mode)."""
    B = 4 if M in (4116, 4516) else 2
    return B, M // B


def _rand(*shape, dtype=BF, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(dtype)


def make_r5(s: dict):
    """r5_v1 tuned kernel incl. its epilogue: qkv+RoPE+[3,B,N,H,64] layout | proj+gamma+fp32 residual | fc1+bias+GELU | fc2+gamma+fp32 residual."""
    o = r5()
    M, N, K, kind, model = s["M"], s["N"], s["K"], s["kind"], s["model"]
    if kind not in ("qkv", "proj", "fc1", "fc2"):
        return None
    B, Nt = _batch(M)
    d, H = MODELS[model]["d"], MODELS[model]["H"]
    R = R_FOR(N * K * 2)
    X = _rand(B, Nt, K, scale=0.5)
    Ws = [_rand(N, K, scale=0.03) for _ in range(R)]
    bs = [_rand(N, scale=0.1) for _ in range(R)]
    fns = []
    if kind == "qkv":
        sin = torch.randn(1024, 64, device="cuda").contiguous()
        cos = torch.randn(1024, 64, device="cuda").contiguous()
        for W, b in zip(Ws, bs):
            fns.append(lambda W=W, b=b: o.qkv_rope(X, W, b, sin, cos, Nt - 1024, H))
    elif kind == "fc1":
        for W, b in zip(Ws, bs):
            fns.append(lambda W=W, b=b: o.gemm_bias_gelu(X, W, b))
    else:
        res = torch.randn(B, Nt, N, device="cuda", dtype=torch.float32)
        g = torch.rand(N, device="cuda", dtype=torch.float32)
        for W, b in zip(Ws, bs):
            fns.append(lambda W=W, b=b: o.gemm_bias_residual(X, W, b, res, g))
    return fns


def make_cublas(dtype, fp16acc=False):
    def make(s):
        M, N, K = s["M"], s["N"], s["K"]
        R = R_FOR(N * K * 2)
        X = _rand(M, K, dtype=dtype, scale=0.5)
        Ws = [_rand(N, K, dtype=dtype, scale=0.03) for _ in range(R)]
        flag = torch.backends.cuda.matmul.allow_fp16_accumulation
        # the flag is read when the kernel is launched / captured: set it around the capture by wrapping each closure
        def mk(W):
            def f():
                old = torch.backends.cuda.matmul.allow_fp16_accumulation
                torch.backends.cuda.matmul.allow_fp16_accumulation = fp16acc
                try:
                    return F.linear(X, W)
                finally:
                    torch.backends.cuda.matmul.allow_fp16_accumulation = old
            return f
        return [mk(W) for W in Ws]
    return make


def base_cands(names=None) -> list[Cand]:
    allc = [
        Cand("r5_v1", make_r5, "r5_v1 tuned kernel WITH epilogue (RoPE/GELU/gamma+fp32 residual)"),
        Cand("cublas_bf16", make_cublas(BF), "F.linear bf16/fp32-acc, no bias/epilogue, bf16 out"),
        Cand("cublas_fp16", make_cublas(F16), "F.linear fp16/fp32-acc, no epilogue"),
        Cand("cublas_fp16acc", make_cublas(F16, True), "F.linear fp16 with allow_fp16_accumulation=True, no epilogue"),
    ]
    return [c for c in allc if names is None or c.name in names]


def cands(names):
    return base_cands(names)
