"""Per-block fwd / bwd GPU time (CUDA-graph replay: no host launch cost, and proof that the MXFP8 block is capturable) of the round17
MXFP8 policies vs the round13 bf16 block, real shapes, same attention.  Includes the per-step weight quantization (block index 0).

  python tests/bench_block_mx8.py --models vitl16 vitb16 --variants all fc fc+proj --json profile/bench_block_mx8.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache")
ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/lightly-train/src")
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "tests"))

import torch

from common.bench import gpu_lock
from common.ops import default_ops
from test_fused_block import DIMS, _Wrap, make_block, make_rope
from bench_block import _graph_time


def build(blk, kind, variant=None):
    from round11_bf16_mixed.impl.attn import AttnCfg, make_attn_core
    from round13_gemm_fusion_sm120.impl import fused_block as fb13
    from round17_mxfp8_sm120.impl import fused_block_mx8 as fb17

    cfg = AttnCfg(rope="triton", plain="flash", joint="cudnn", qsplit=True, nq=100)
    ops = default_ops()
    ctx = SimpleNamespace(ops=ops, shadow={}, meta={"enable_autocast": True}, model=_Wrap(blk), rng=None, cfg=SimpleNamespace(precision="bf16-mixed"))
    ops.attn_core = make_attn_core(cfg, fallback=ops.attn_core)
    if kind == "r13":
        ops.block = fb13.make_block(ctx, ops.block)
    else:
        ops.block = fb17.make_block(ctx, ops.block, fb17.Policy([p for p in variant.split("+") if p]))
    return ops, ctx


def time_variant(ops, ctx, blk, x, rope, mask, g, iters=20, warmup=5, repeats=3):
    def fwd_only():
        xin = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ops.block(ctx, 0, blk, xin, rope, mask)
        return xin, out

    def fwd_bwd():
        xin, out = fwd_only()
        out.backward(g)

    for _ in range(warmup):
        fwd_bwd()
    torch.cuda.synchronize()
    params = list(blk.parameters())
    f = _graph_time(fwd_only, params, iters, repeats)
    t = _graph_time(fwd_bwd, params, iters, repeats)
    for p in params:
        p.grad = None
    return {"fwd_ms": f, "bwd_ms": t - f, "fwdbwd_ms": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vitl16", "vitb16"])
    ap.add_argument("--variants", nargs="+", default=["all", "fc", "fc+proj"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    allr = {}
    with gpu_lock():
        for model in args.models:
            d, heads, B = DIMS[model]
            for joint in (False, True):
                N = 1029 + (100 if joint else 0)
                blk = make_block(d, heads)
                x = torch.randn(B, N, d, device="cuda") * 2.0
                x[..., 7] = 1.5e5
                g = torch.randn(B, N, d, device="cuda") * 0.05
                rope = make_rope(5, "cuda")
                mask = None
                if joint:
                    mask = torch.ones(B, N, N, dtype=torch.bool, device="cuda")
                    mask[:, :100, 105:] = torch.rand(B, 100, 1024, device="cuda") > 0.5
                key = f"{model}_{'joint' if joint else 'plain'}"
                allr[key] = {}
                specs = [("r13_full", "r13", None)] + [(f"mx8_{v}", "r17", v) for v in args.variants]
                for name, kind, v in specs:
                    ops, ctx = build(blk, kind, v)
                    try:
                        r = time_variant(ops, ctx, blk, x, rope, mask, g)
                    except Exception as exc:
                        r = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                    allr[key][name] = r
                    base = allr[key]["r13_full"].get("fwdbwd_ms")
                    if "error" in r:
                        print(f"{key:14s} {name:16s} ERROR {r['error']}", flush=True)
                    else:
                        print(f"{key:14s} {name:16s} fwd {r['fwd_ms']:6.3f}  bwd {r['bwd_ms']:6.3f}  fwd+bwd {r['fwdbwd_ms']:6.3f} ms  ({base / r['fwdbwd_ms']:.2f}x vs r13_full)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
