"""Round 10 GEMM candidates, part 4: NVFP4 (nv4: e2m1 + e4m3/16 + fp32 tensor scale) and MXFP4 (mx4: e2m1 + ue8m0/32).

  cublaslt_{nv4,mx4}   torch.nn.functional.scaled_mm (cuBLASLt)                 bf16 out, no epilogue
  quack_{nv4,mx4}      train-opt r20 fp4_gemm (QuACK SM120 block-scaled GEMM)   WITH epilogue (qkv: bias only, no RoPE/layout; fc1 bias+GELU (+z); proj/fc2 bias+gamma+residual (+y))
  spark_{nv4,mx4}_v{1,2}   Spark fp4gemm (variant 2 = TMA, M padded to 128)     bf16 out, no epilogue
  qutlass_{nv4,mx4}    QuTLASS matmul_{nvf4,mxf4}_bf16_tn (prebuilt .so under train-opt-tmp, loaded read-only)   bf16 out, no epilogue
Operands are built with the pure-torch reference quantiser (setup only, not timed); the activation quantiser cost is measured separately (bench_quant.py).
"""
from __future__ import annotations

import os
import sys

import torch

from bench10 import Cand, R_FOR, TMP
import cands_base as cb
import cands_mx8  # noqa: F401  (sets QUACK_CACHE_DIR and the train-opt sys.path)

BF = torch.bfloat16
_M = {}


def mods():
    if "R" not in _M:
        from round20_fp4_probe.impl import fp4_cands as C
        from round20_fp4_probe.impl import fp4_gemm as G4
        from round20_fp4_probe.impl import fp4_ref as R
        _M.update(R=R, C=C, G4=G4)
    return _M["R"], _M["C"], _M["G4"]


def spark():
    return cands_mx8.spark()


def qut():
    if "qut" not in _M:
        torch.ops.load_library("/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/build/qutlass/_qutlass_C.so")
        _M["qut"] = torch.ops._qutlass_C
    return _M["qut"]


def _pad128(n):
    return -(-n // 128) * 128


def _setup(s, fmt, pad_m=False):
    R, C, G4 = mods()
    M, N, K = s["M"], s["N"], s["K"]
    Mq = _pad128(M) if pad_m else M
    nW = R_FOR(N * K // 2)
    X = torch.zeros(Mq, K, device="cuda", dtype=BF)
    X[:M] = cb._rand(M, K, scale=0.5)
    a = R.quantize_matrix(X, fmt)
    bs = [R.quantize_matrix(cb._rand(N, K, scale=0.03), fmt) for _ in range(nW)]
    return R, C, G4, M, N, K, Mq, a, bs


def _t(R, fmt, trip, rows, cols):
    q, sf, ts = trip
    return R.FP4Tensor(fmt, rows, cols, R.pad128(rows), q=q, sf=sf, ts=ts)


def make_cublaslt(fmt):
    def make(s):
        R, C, G4, M, N, K, Mq, a, bs = _setup(s, fmt)
        return [(lambda b=b: C.scaled_mm_fp4(a, b, fmt, BF)) for b in bs]
    return make


def make_quack(fmt):
    def make(s):
        R, C, G4, M, N, K, Mq, a, bs = _setup(s, fmt)
        x = _t(R, fmt, a, M, K)
        ws = [_t(R, fmt, b, N, K) for b in bs]
        kind = s["kind"]
        bias = torch.randn(N, device="cuda") * 0.1
        gamma = torch.rand(N, device="cuda")
        res = torch.randn(M, N, device="cuda")
        if kind == "qkv":
            return [(lambda w=w: G4.fwd_bias(x, w, bias)) for w in ws]
        if kind == "fc1":
            return [(lambda w=w: G4.fwd_gelu(x, w, bias)) for w in ws]
        return [(lambda w=w: G4.fwd_res(x, w, bias, gamma, res)) for w in ws]
    return make


def make_spark(fmt, variant):
    def make(s):
        ext = spark()
        R, C, G4, M, N, K, Mq, a, bs = _setup(s, fmt, pad_m=(variant == 2))
        if not ext.fp4_supports(Mq, N, K, variant):
            return None
        out = torch.empty(Mq, N, device="cuda", dtype=BF)
        return [(lambda b=b: C.spark_fp4(ext, a, b, out, fmt, variant)) for b in bs]
    return make


def make_qutlass(fmt):
    def make(s):
        q = qut()
        R, C, G4, M, N, K, Mq, a, bs = _setup(s, fmt)
        sd = torch.float8_e4m3fn if fmt == "nv4" else torch.float8_e8m0fnu
        vec = R.VEC[fmt]
        fn = q.matmul_nvf4_bf16_tn if fmt == "nv4" else q.matmul_mxf4_bf16_tn
        out = []
        for b in bs:
            alpha = (a[2] * b[2]).reshape(1).contiguous() if fmt == "nv4" else torch.ones(1, device="cuda")
            out.append(lambda b=b, alpha=alpha: fn(a[0].view(torch.uint8), b[0].view(torch.uint8), a[1].view(sd).reshape(-1, K // vec), b[1].view(sd).reshape(-1, K // vec), alpha))
        return out
    return make


def cands(names):
    allc = []
    for fmt in ("nv4", "mx4"):
        allc += [
            Cand(f"cublaslt_{fmt}", make_cublaslt(fmt), f"cuBLASLt {fmt} scaled_mm, bf16 out, NO epilogue, NO quantization"),
            Cand(f"quack_{fmt}", make_quack(fmt), f"QuACK {fmt} WITH epilogue (qkv: bias only; fc1: bias+GELU+z; proj/fc2: bias+gamma+residual+y), NO quantization"),
            Cand(f"spark_{fmt}_v1", make_spark(fmt, 1), f"Spark fp4gemm {fmt} v1, bf16 out, NO epilogue, NO quantization"),
            Cand(f"spark_{fmt}_v2", make_spark(fmt, 2), f"Spark fp4gemm {fmt} v2 (TMA; M padded to 128), bf16 out, NO epilogue, NO quantization"),
            Cand(f"qutlass_{fmt}", make_qutlass(fmt), f"QuTLASS {fmt} GEMM, bf16 out, NO epilogue, NO quantization"),
        ]
    return [c for c in allc if names is None or c.name in names]
