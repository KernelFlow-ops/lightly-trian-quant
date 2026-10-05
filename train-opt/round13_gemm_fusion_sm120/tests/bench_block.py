"""Per-block fwd / bwd timing of the fused round13 block vs torch-autocast and round5 blocks (real shapes, same attention).

All variants use the identical ``attn_core`` (round11 flash / cuDNN-joint, triton RoPE), so the difference is the
non-attention part of the block.  CUDA events, warmup 10, 30 timed iterations x 3 repeats (median).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/lightly-train/src")
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "tests"))

import torch

from common.bench import gpu_lock
from common.ops import default_ops
from test_fused_block import DIMS, _Wrap, make_block, make_rope


def build_variants(blk, joint):
    from round11_bf16_mixed.impl.attn import AttnCfg, make_attn_core
    from round13_gemm_fusion_sm120.impl import fused_block, gemm_ops
    import round5_block_fusion.impl.patch as r5

    cfg = AttnCfg(rope="triton", plain="flash", joint="cudnn", qsplit=True, nq=100)
    wrap = _Wrap(blk)
    out = {}

    def mk(name, policy=None, kind="default"):
        ops = default_ops()
        ctx = SimpleNamespace(ops=ops, shadow={}, meta={"enable_autocast": True}, model=wrap, rng=None,
                              cfg=SimpleNamespace(precision="bf16-mixed"))
        ops.attn_core = make_attn_core(cfg, fallback=ops.attn_core)
        if kind == "r5":
            r5.apply(ctx, "ln_fuse")
        elif kind == "r13":
            gemm_ops.POLICY.clear()
            gemm_ops.POLICY.update(policy)
            fb = fused_block.make_block(ctx, ops.block)
            ops.block = fb
        out[name] = (ops, ctx, policy)

    mk("autocast_default")
    mk("r5_ln_fuse", kind="r5")
    full = dict(gemm_ops.DEFAULT_POLICY)
    mk("r13_full", full, "r13")
    mk("r13_wgrad_quack", {k: "quack" for k in full}, "r13")
    mk("r13_torch_gemm", {k: "torch" for k in full}, "r13")
    return out


def _graph_time(fn, params, iters=20, repeats=3):
    """GPU time of ``fn`` (captured once into a CUDA graph; removes host launch overhead)."""
    for p in params:
        p.grad = None
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
            for p in params:
                p.grad = None
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    reps = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        reps.append(a.elapsed_time(b) / iters)
    del g
    return statistics.median(reps)


def time_variant(ops, ctx, blk, x, rope, mask, g, policy, iters=20, warmup=5, repeats=3):
    from round13_gemm_fusion_sm120.impl import gemm_ops

    def fwd_only():
        xin = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ops.block(ctx, 0, blk, xin, rope, mask)
        return xin, out

    def fwd_bwd():
        xin, out = fwd_only()
        out.backward(g)

    if policy is not None:
        gemm_ops.POLICY.clear()
        gemm_ops.POLICY.update(policy)
    for _ in range(warmup):  # eager warm-up: JIT / autotune / allocator
        fwd_bwd()
    torch.cuda.synchronize()
    params = list(blk.parameters())
    f = _graph_time(fwd_only, params, iters, repeats)
    t = _graph_time(fwd_bwd, params, iters, repeats)
    for p in params:
        p.grad = None
    return {"fwd_ms": f, "fwdbwd_ms": t, "bwd_ms": t - f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vitl16", "vitb16"])
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
                variants = build_variants(blk, joint)
                key = f"{model}_{'joint' if joint else 'plain'}"
                allr[key] = {}
                for name, (ops, ctx, policy) in variants.items():
                    r = time_variant(ops, ctx, blk, x, rope, mask, g, policy)
                    allr[key][name] = r
                    print(f"{key:14s} {name:18s} fwd {r['fwd_ms']:7.3f} ms  bwd {r['bwd_ms']:7.3f} ms  fwd+bwd {r['fwdbwd_ms']:7.3f} ms", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
