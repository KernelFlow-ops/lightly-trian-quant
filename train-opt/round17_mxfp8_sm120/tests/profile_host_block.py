"""Host-side (python) profile of one eager MX8 block fwd+bwd: where does the launch cost go?  cProfile over N iterations after warm-up,
GPU synced outside the profiled region so only CPU enqueue time is measured (queue never fills for one block)."""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache")
ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/lightly-train/src")
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "tests"))
sys.path.insert(0, str(ROOT / "round17_mxfp8_sm120" / "tests"))

import torch

from common.bench import gpu_lock
from test_fused_block import DIMS, make_block, make_rope
from bench_block_mx8 import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all")
    ap.add_argument("--kind", default="r17")
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    d, heads, B = DIMS[args.model]
    N = 1029
    blk = make_block(d, heads)
    x = torch.randn(B, N, d, device="cuda") * 2.0
    g = torch.randn(B, N, d, device="cuda") * 0.05
    rope = make_rope(5, "cuda")
    with gpu_lock():
        ops, ctx = build(blk, args.kind, args.variant)
        # --- per-function host timers (inclusive) ---
        import collections
        acc = collections.defaultdict(lambda: [0.0, 0])

        def wrap(mod, name):
            fn = getattr(mod, name)

            def w(*a, **k):
                t = time.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    r = acc[f"{mod.__name__.split('.')[-1]}.{name}"]
                    r[0] += time.perf_counter() - t
                    r[1] += 1

            setattr(mod, name, w)

        from round13_gemm_fusion_sm120.impl import block_kernels as bk13, gemm_ops as g13
        from round17_mxfp8_sm120.impl import gemm_mx8 as gmx, mx8 as mxm
        for mod, names in ((bk13, ("ln_fwd", "ln_bwd_res", "gamma_prep", "colsum_bf16")), (g13, ("linear_bias", "linear_res", "linear_gelu", "dgrad", "dgrad_dgelu", "wgrad")),
                           (gmx, ("fwd_bias", "fwd_res", "fwd_gelu", "dgrad", "dgrad_dgelu", "wgrad")), (mxm, ("quantize",))):
            for n in names:
                wrap(mod, n)
        # the fused_block modules bind these names through the module objects (bk.ln_fwd, G.fwd_bias, mx8.quantize, ops.dgrad): patched above

        def once():
            xin = x.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = ops.block(ctx, 0, blk, xin, rope, None)
            out.backward(g)

        for _ in range(8):
            once()
        torch.cuda.synchronize()
        # plain host time per fwd+bwd (no profiler)
        t0 = time.perf_counter()
        for _ in range(args.n):
            once()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        print(f"{args.kind}:{args.variant}: host enqueue {(t1 - t0) / args.n * 1e3:.3f} ms per block fwd+bwd, wall incl. GPU {(t2 - t0) / args.n * 1e3:.3f} ms")
        torch.cuda.synchronize()
        acc.clear()
        torch.cuda.synchronize()
        for _ in range(args.n):
            once()
        torch.cuda.synchronize()
        print("per-function inclusive host time per block fwd+bwd (us, calls):")
        tot = 0.0
        for k, (t, n) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
            print(f"   {k:28s} {t / args.n * 1e6:8.1f} us  {n / args.n:5.1f} calls  ({t / n * 1e6:6.1f} us/call)")
            tot += t
        print(f"   {'sum of wrapped':28s} {tot / args.n * 1e6:8.1f} us")
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(args.n):
            once()
        pr.disable()
        torch.cuda.synchronize()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22)
    print("\n".join(l[:170] for l in s.getvalue().splitlines()[:40]))


if __name__ == "__main__":
    main()
