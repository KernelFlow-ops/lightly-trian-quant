"""Gate A at block level: round17 MXFP8 block vs the fp32 block, next to the round13 bf16 fused block (same-precision reference).

Real DINOv3 SelfAttentionBlock (LayerScale, LinearKMaskedBias, Mlp GELU(erf)), real shapes, spiky massive-activation residual,
plain (N=1029) and joint (N=1129, masked) blocks.  For every tensor (out, dx, all Parameter gradients) cos / rel-L2 vs fp32.
MXFP8 error is expected to be a few percent (E4M3 has 3 mantissa bits); the gate here is cos >= 0.99 for out / dx and the global
gradient cos >= 0.99; per-parameter values are recorded.

  python tests/test_block_mx8.py [--models vitl16 vitb16] [--variants all fc ...] [--json profile/gateA_block_mx8.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache")
ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "tests"))
sys.path.insert(0, "/root/lightly-train/src")

import torch

from common.bench import gpu_lock
from common.ops import default_ops
import test_fused_block as T  # round13 helpers: make_block, make_rope, run_block, cos_rel, DIMS, _Wrap


def case(model, joint, spiky, variants, seed=0):
    from round13_gemm_fusion_sm120.impl import fused_block as fb13
    from round17_mxfp8_sm120.impl import fused_block_mx8 as fb17

    d, heads, B = T.DIMS[model]
    N = 1029 + (100 if joint else 0)
    dev = "cuda"
    blk = T.make_block(d, heads, seed)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    x = torch.randn(B, N, d, device=dev, generator=gen) * 2.0
    if spiky:
        x[..., 7] = 1.5e5
        x[..., d // 2 + 3] = -8.0e4
        x[:, ::97] *= 30.0
    g = torch.randn(B, N, d, device=dev, generator=gen) * 0.05
    rope = T.make_rope(5, dev)
    mask = None
    if joint:
        mask = torch.ones(B, N, N, dtype=torch.bool, device=dev)
        mask[:, :100, 105:] = torch.rand(B, 100, 1024, device=dev, generator=gen) > 0.5
    ops_ref = default_ops()
    cfg = SimpleNamespace(precision="bf16-mixed")
    ctx = SimpleNamespace(ops=ops_ref, shadow={}, meta={"enable_autocast": True}, model=T._Wrap(blk), rng=None, cfg=cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    out32, g32 = T.run_block(blk, ops_ref, ctx, x, rope, mask, g, autocast=False)

    def run_with(make):
        ops_f = default_ops()
        ctx_f = SimpleNamespace(ops=ops_f, shadow={}, meta={"enable_autocast": True}, model=ctx.model, rng=None, cfg=cfg)
        blockfn = make(ctx_f, ops_f.block)
        blockfn.shadows.refresh()
        ops_f.block = blockfn
        return T.run_block(blk, ops_f, ctx_f, x, rope, mask, g, autocast=True)

    results = {"r13_bf16": run_with(lambda c, prev: fb13.make_block(c, prev))}
    for v in variants:
        pol = fb17.Policy([p for p in v.split("+") if p])
        results[v] = run_with(lambda c, prev, pol=pol: fb17.make_block(c, prev, pol))
    rows = {}
    for name in ["out"] + sorted(g32):
        a32 = out32 if name == "out" else g32[name]
        rows[name] = {k: list(T.cos_rel(o if name == "out" else gr[name], a32)) for k, (o, gr) in results.items()}
    # global gradient cosine (all params + x)
    glob = {}
    for k, (o, gr) in results.items():
        dot = sum(float(torch.dot(gr[n].double().reshape(-1), g32[n].double().reshape(-1))) for n in g32)
        n1 = math.sqrt(sum(float(gr[n].double().pow(2).sum()) for n in g32))
        n2 = math.sqrt(sum(float(g32[n].double().pow(2).sum()) for n in g32))
        glob[k] = dot / (n1 * n2)
    return rows, glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vitl16", "vitb16"])
    ap.add_argument("--variants", nargs="+", default=["all", "fc"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--joint", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--spiky", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    allr, ok = {}, True
    with gpu_lock():
        for model in args.models:
            for joint in args.joint:
                for spiky in args.spiky:
                    key = f"{model}_{'joint' if joint else 'plain'}_{'spiky' if spiky else 'normal'}"
                    rows, glob = case(model, bool(joint), bool(spiky), args.variants)
                    allr[key] = {"rows": rows, "global_grad_cos": glob}
                    line = [f"{k}: gcos={v:.5f} out_cos={rows['out'][k][0]:.5f} dx_cos={rows['x'][k][0]:.5f} min_param_cos={min(r[k][0] for n, r in rows.items() if n not in ('out', 'x')):.4f}" for k, v in glob.items()]
                    print(key, " | ".join(line), flush=True)
                    for k in args.variants:
                        ok &= glob[k] >= 0.99 and rows["out"][k][0] >= 0.99 and rows["x"][k][0] >= 0.99
    if args.json:
        Path(args.json).write_text(json.dumps(allr, indent=1))
    print("ALL PASS (global grad cos, out, dx >= 0.99)" if ok else "SOME BELOW 0.99")


if __name__ == "__main__":
    main()
