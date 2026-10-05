"""Gate A at block level: fused round13 block vs fp32 reference and vs the torch-autocast (T-BF16) block.

Real DINOv3 ``SelfAttentionBlock`` (LayerScale, LinearKMaskedBias qkv, Mlp GELU(erf)), real shapes, spiky
(massive-activation) residual stream, plain (N=1029) and joint (N=1129, masked) blocks.  For each case the
output and every gradient (x, all Parameters) is compared against the fp32 block, and the fused error is
required to stay within a small factor of the error of the torch-autocast block (the same-precision reference).

  python tests/test_fused_block.py [--models vitl16 vitb16] [--json profile/gateA_block.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/lightly-train/src")

import torch
import torch.nn as nn

from common.bench import gpu_lock
from common.ops import default_ops

DIMS = {"vitl16": (1024, 16, 7), "vitb16": (768, 12, 14)}


class _Wrap(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([blk])


def make_block(d, heads, seed=0):
    from lightly_train._models.dinov3.dinov3_src.layers.block import SelfAttentionBlock

    torch.manual_seed(seed)
    blk = SelfAttentionBlock(
        dim=d, num_heads=heads, ffn_ratio=4.0, qkv_bias=True, proj_bias=True, ffn_bias=True, init_values=1e-5,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), mask_k_bias=True,
    ).cuda()
    with torch.no_grad():
        for p in blk.parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.03)
            else:
                p.normal_(0, 0.1)
        blk.norm1.weight.uniform_(0.5, 1.5)
        blk.norm2.weight.uniform_(0.5, 1.5)
        blk.ls1.gamma.uniform_(0.05, 0.6)   # large LayerScale: the branch matters in the residual sum
        blk.ls2.gamma.uniform_(0.05, 0.6)
        mask = torch.ones_like(blk.attn.qkv.bias)
        mask[d:2 * d] = 0.0                 # k bias masked, as in DINOv3
        blk.attn.qkv.bias_mask.copy_(mask)
    return blk


def make_rope(n_prefix, dev):
    # fp32 [1024, 64] sin/cos with the DINOv3 structure (first half == second half)
    pos = torch.rand(1024, 32, device=dev) * 6.28
    sin, cos = torch.sin(pos).repeat(1, 2), torch.cos(pos).repeat(1, 2)
    return sin, cos


def cos_rel(a, b):
    a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    c = float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))
    r = float((a - b).norm() / b.norm().clamp_min(1e-30))
    return c, r


def run_block(blk, ops, ctx, x, rope, mask, g, autocast):
    for p in blk.parameters():
        p.grad = None
    xin = x.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = ops.block(ctx, 0, blk, xin, rope, mask)
    out.backward(g)
    grads = {n: p.grad.detach().clone() for n, p in blk.named_parameters() if p.grad is not None}
    grads["x"] = xin.grad.detach().clone()
    return out.detach(), grads


def case(model, joint, seed=0, spiky=True):
    from round13_gemm_fusion_sm120.impl import fused_block

    d, heads, B = DIMS[model]
    n_prefix = 5
    N = 1029 + (100 if joint else 0)
    dev = "cuda"
    blk = make_block(d, heads, seed)
    g_ = torch.Generator(device=dev).manual_seed(seed + 1)
    x = torch.randn(B, N, d, device=dev, generator=g_) * 2.0
    if spiky:
        x[..., 7] = 1.5e5
        x[..., d // 2 + 3] = -8.0e4
        x[:, ::97] *= 30.0
    g = torch.randn(B, N, d, device=dev, generator=g_) * 0.05
    rope = make_rope(n_prefix, dev)
    mask = None
    if joint:
        mask = torch.ones(B, N, N, dtype=torch.bool, device=dev)
        mask[:, :100, 105:] = torch.rand(B, 100, 1024, device=dev, generator=g_) > 0.5
    ops_ref = default_ops()
    ctx = SimpleNamespace(ops=ops_ref, shadow={}, meta={"enable_autocast": True}, model=_Wrap(blk), rng=None)
    ctx.cfg = SimpleNamespace(precision="bf16-mixed")
    # fp32 reference (TF32 off): the exact source block semantics
    torch.backends.cuda.matmul.allow_tf32 = False
    out32, g32 = run_block(blk, ops_ref, ctx, x, rope, mask, g, autocast=False)
    out_ac, g_ac = run_block(blk, ops_ref, ctx, x, rope, mask, g, autocast=True)
    ops_f = default_ops()
    ctx_f = SimpleNamespace(ops=ops_f, shadow={}, meta={"enable_autocast": True}, model=ctx.model, rng=None,
                            cfg=ctx.cfg)
    fused = fused_block.make_block(ctx_f, ops_f.block)
    fused.shadows.refresh()
    ops_f.block = fused
    out_f, g_f = run_block(blk, ops_f, ctx_f, x, rope, mask, g, autocast=True)
    rows = {}
    for name in ["out"] + sorted(g32):
        a32 = out32 if name == "out" else g32[name]
        aac = out_ac if name == "out" else g_ac[name]
        af = out_f if name == "out" else g_f[name]
        c_ac, r_ac = cos_rel(aac, a32)
        c_f, r_f = cos_rel(af, a32)
        c_af, r_af = cos_rel(af, aac)
        rows[name] = {"autocast_vs_fp32": [c_ac, r_ac], "fused_vs_fp32": [c_f, r_f], "fused_vs_autocast": [c_af, r_af]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vitl16", "vitb16"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--joint", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--spiky", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    allr = {}
    ok = True
    with gpu_lock():
        for model in args.models:
            for joint in args.joint:
                for spiky in args.spiky:
                    key = f"{model}_{'joint' if joint else 'plain'}_{'spiky' if spiky else 'normal'}"
                    rows = case(model, bool(joint), spiky=bool(spiky))
                    allr[key] = rows
                    worst_cos = min(r["fused_vs_fp32"][0] for r in rows.values())
                    # pass: fused error <= 2x autocast error (+ small absolute floor) for every tensor
                    bad = [n for n, r in rows.items() if r["fused_vs_fp32"][1] > 2.0 * r["autocast_vs_fp32"][1] + 2e-3]
                    ok &= not bad
                    print(f"{key}: worst fused-vs-fp32 cos={worst_cos:.6f}  {'PASS' if not bad else 'FAIL ' + str(bad)}", flush=True)
                    for n in ("out", "x", "attn.qkv.weight", "attn.qkv.bias", "ls1.gamma", "ls2.gamma", "norm1.weight", "norm2.bias", "mlp.fc1.bias", "mlp.fc2.weight"):
                        if n in rows:
                            r = rows[n]
                            print(f"   {n:18s} autocast/fp32 cos={r['autocast_vs_fp32'][0]:.6f} rel={r['autocast_vs_fp32'][1]:.2e} | "
                                  f"fused/fp32 cos={r['fused_vs_fp32'][0]:.6f} rel={r['fused_vs_fp32'][1]:.2e} | fused/autocast cos={r['fused_vs_autocast'][0]:.6f}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(allr, indent=1))
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
