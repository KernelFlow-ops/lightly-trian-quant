"""Host (launch) cost per call of the round17 MXFP8 ops vs round13 bf16 ops: CPU time to enqueue N calls (no sync inside the loop).
A launch cost above the GPU time of the kernel makes the forward host-bound (the r16 CUDA graph removes it)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))

import torch

from common.bench import gpu_lock
from round13_gemm_fusion_sm120.impl import gemm_ops as g13
from round17_mxfp8_sm120.impl import gemm_mx8 as G
from round17_mxfp8_sm120.impl import mx8


def host_us(fn, n=60):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / n * 1e6


def main():
    d, M = 1024, 7203
    N, K = 4 * d, d
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = (torch.randn(N, K, device="cuda") * 0.05)
    Wb = W.to(torch.bfloat16)
    b = torch.randn(N, device="cuda")
    gam = torch.rand(K, device="cuda")
    resid = torch.randn(M, K, device="cuda")
    Z = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dY = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    with gpu_lock():
        Xq, Wq, dYq = mx8.quantize(X), mx8.quantize(W), mx8.quantize(dY)
        W2q = mx8.quantize(W.t().contiguous())  # fc2-like [K, N]
        res = {}
        res["quantize dual bf16 [M,1024]"] = host_us(lambda: mx8.quantize(X))
        res["quantize dual fp32 weight [4096,1024]"] = host_us(lambda: mx8.quantize(W))
        res["mx8 fwd_bias"] = host_us(lambda: G.fwd_bias(Xq, Wq, b))
        res["mx8 fwd_gelu"] = host_us(lambda: G.fwd_gelu(Xq, Wq, b))
        res["mx8 fwd_res (fc2-like)"] = host_us(lambda: G.fwd_res(dYq, W2q, gam, gam, resid))
        res["mx8 dgrad"] = host_us(lambda: G.dgrad(dYq, Wq))
        res["mx8 wgrad"] = host_us(lambda: G.wgrad(dYq, Xq))
        res["bf16 fwd_bias (r13)"] = host_us(lambda: g13.linear_bias(X, Wb, b, "fc1"))
        res["bf16 fwd_gelu (r13)"] = host_us(lambda: g13.linear_gelu(X, Wb, b))
        res["bf16 dgrad (r13)"] = host_us(lambda: g13.dgrad(dY, Wb, "fc1"))
    for k, v in res.items():
        print(f"{k:42s} {v:8.1f} us host/call")


if __name__ == "__main__":
    main()
