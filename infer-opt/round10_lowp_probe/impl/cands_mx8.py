"""Round 10 GEMM candidates, part 3: MXFP8 (E4M3 + UE8M0 1x32) and INT8.

  cublaslt_mx8   torch._scaled_mm (cuBLASLt block-scaled FP8), bf16 out, NO epilogue           (operands from train-opt r17 `quant_dual`)
  quack_mx8      train-opt r17 gemm_mx8 (QuACK SM120 block-scaled GEMM) WITH epilogue:         qkv: bias only (RoPE+layout missing), proj/fc2: bias+gamma+fp32 residual
                 (also writes the bf16 `y` the training epilogue saves), fc1: bias+GELU (also writes the pre-activation z)
  spark_mx8_v1/v2   Spark fp8gemm_mx (variant 1 = cp.async 128x128x64, 2 = TMA warp-specialised; M padded to 128), bf16 out, NO epilogue
  int8_mm        torch._int_mm (s8 x s8 -> s32), NO dequant/bias epilogue
train-opt imports follow the round17/20 recipe (PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src) but QuACK's disk cache and r17's extension
build go to infer-opt-tmp/round10_lowp_probe (nothing is written under train-opt-tmp).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from bench10 import MODELS, Cand, R_FOR, TMP
import cands_base as cb

os.environ["QUACK_CACHE_DIR"] = str(TMP / "quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
for _p in ("/root/lightly-train/train-opt", "/root/lightly-train/src"):
    if _p not in sys.path:
        sys.path.append(_p)

BF = torch.bfloat16
_M = {}


def r17():
    if "mx8" not in _M:
        from round17_mxfp8_sm120.impl import gemm_mx8 as G
        from round17_mxfp8_sm120.impl import mx8
        mx8._BUILD = TMP / "build" / "r17_mx8"
        _M["mx8"], _M["G"] = mx8, G
    return _M["mx8"], _M["G"]


def spark():
    if "spark" not in _M:
        import spark_ext
        _M["spark"] = spark_ext.extension()
    return _M["spark"]


def _pad128(n):
    return -(-n // 128) * 128


def _common(s):
    M, N, K = s["M"], s["N"], s["K"]
    B, Nt = cb._batch(M)
    R = R_FOR(N * K)       # fp8 weight bytes = N*K
    X = cb._rand(M, K, scale=0.5)
    Ws = [cb._rand(N, K, scale=0.03) for _ in range(R)]
    return M, N, K, B, Nt, R, X, Ws


def make_cublaslt_mx8(s):
    mx8, _ = r17()
    M, N, K, B, Nt, R, X, Ws = _common(s)
    xq = mx8.quantize(X, row=True, col=False)
    wqs = [mx8.quantize(W, row=True, col=False) for W in Ws]
    sa = xq.sfr.view(torch.float8_e8m0fnu).flatten()
    out = []
    for wq in wqs:
        sb = wq.sfr.view(torch.float8_e8m0fnu).flatten()
        out.append(lambda wq=wq, sb=sb: torch._scaled_mm(xq.qr, wq.qr.t(), scale_a=sa, scale_b=sb, out_dtype=BF))
    return out


def make_quack_mx8(s):
    mx8, G = r17()
    M, N, K, B, Nt, R, X, Ws = _common(s)
    kind = s["kind"]
    xq = mx8.quantize(X, row=True, col=False)
    wqs = [mx8.quantize(W, row=True, col=False) for W in Ws]
    bias = torch.randn(N, device="cuda") * 0.1
    gamma = torch.rand(N, device="cuda")
    res = torch.randn(M, N, device="cuda")
    if kind == "qkv":
        return [(lambda wq=wq: G.fwd_bias(xq, wq, bias)) for wq in wqs]
    if kind == "fc1":
        return [(lambda wq=wq: G.fwd_gelu(xq, wq, bias)) for wq in wqs]
    return [(lambda wq=wq: G.fwd_res(xq, wq, bias, gamma, res)) for wq in wqs]


def make_spark_mx8(variant):
    def make(s):
        ext = spark()
        M, N, K, B, Nt, R, X, Ws = _common(s)
        Mp = _pad128(M) if variant == 2 else M
        if not ext.fp8_supports(Mp, N, K, variant):
            return None
        Xp = torch.zeros(Mp, K, device="cuda", dtype=BF)
        Xp[:M] = X
        xq = torch.zeros(Mp, K, device="cuda", dtype=torch.uint8)
        xs = torch.zeros(Mp, K // 32, device="cuda", dtype=torch.uint8)
        ext.fp8_quantize_mx(Xp, xq, xs)
        out = torch.empty(Mp, N, device="cuda", dtype=BF)
        fns = []
        for W in Ws:
            wq = torch.empty(N, K, device="cuda", dtype=torch.uint8)
            ws = torch.empty(N, K // 32, device="cuda", dtype=torch.uint8)
            ext.fp8_quantize_mx(W, wq, ws)
            fns.append(lambda wq=wq, ws=ws: ext.fp8gemm_mx(xq, wq, out, xs, ws, variant))
        return fns
    return make


def make_int8_mm(s):
    M, N, K, B, Nt, R, X, Ws = _common(s)
    xq = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    return [(lambda wq=torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8): torch._int_mm(xq, wq.t())) for _ in range(R)]


def cands(names):
    allc = [
        Cand("cublaslt_mx8", make_cublaslt_mx8, "cuBLASLt MXFP8 _scaled_mm, bf16 out, NO epilogue, NO quantization"),
        Cand("quack_mx8", make_quack_mx8, "QuACK MXFP8 WITH epilogue (qkv: bias only, no RoPE/layout; fc1: bias+GELU (+z write); proj/fc2: bias+gamma+residual (+y write)), NO quantization"),
        Cand("spark_mx8_v1", make_spark_mx8(1), "Spark fp8gemm_mx v1 (cp.async), bf16 out, NO epilogue, NO quantization"),
        Cand("spark_mx8_v2", make_spark_mx8(2), "Spark fp8gemm_mx v2 (TMA, warp-specialised; M padded to 128), bf16 out, NO epilogue, NO quantization"),
        Cand("int8_mm", make_int8_mm, "torch._int_mm s8xs8->s32 (int32 out, no dequant/bias epilogue), NO quantization"),
    ]
    return [c for c in allc if names is None or c.name in names]
