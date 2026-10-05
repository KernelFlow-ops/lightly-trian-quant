"""nsys entry for ONE block variant (fwd / bwd NVTX ranges) -> exact per-block kernel lists and GPU time split.

  common/profile_nsys.sh <prefix> tests/profile_block.py --variant r13_full --model vitl16 --joint 0
  python tests/nsys_categorize.py <prefix>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/lightly-train/src")
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "tests"))

import torch

from bench_block import build_variants
from common.bench import gpu_lock, nvtx
from test_fused_block import DIMS, make_block, make_rope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="r13_full")
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    d, heads, B = DIMS[args.model]
    N = 1029 + (100 if args.joint else 0)
    blk = make_block(d, heads)
    x = torch.randn(B, N, d, device="cuda") * 2.0
    x[..., 7] = 1.5e5
    g = torch.randn(B, N, d, device="cuda") * 0.05
    rope = make_rope(5, "cuda")
    mask = None
    if args.joint:
        mask = torch.ones(B, N, N, dtype=torch.bool, device="cuda")
        mask[:, :100, 105:] = torch.rand(B, 100, 1024, device="cuda") > 0.5
    from round13_gemm_fusion_sm120.impl import gemm_ops

    ops, ctx, policy = build_variants(blk, bool(args.joint))[args.variant]
    if policy is not None:
        gemm_ops.POLICY.clear()
        gemm_ops.POLICY.update(policy)
    with gpu_lock():
        def once(tag):
            xin = x.clone().requires_grad_()
            with nvtx(f"{tag}/fwd"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = ops.block(ctx, 0, blk, xin, rope, mask)
            with nvtx(f"{tag}/bwd"):
                out.backward(g)
            for p in blk.parameters():
                p.grad = None

        for _ in range(8):
            once("warm")
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for i in range(args.iters):
            once(f"blk_{args.variant}")
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
