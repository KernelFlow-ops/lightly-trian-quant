"""Analytical FLOPs / minimum-bytes model for DINOv3-EoMT inference.

Used instead of ncu: achieved = analytic_work / measured_kernel_time, compared to
measured peaks from peak_bench.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CONFIGS = {
    # embed_dim, depth, heads, joint blocks (from the semantic checkpoints' model_init_args)
    "vitl16": dict(d=1024, depth=24, heads=16, joint=3),
    "vitb16": dict(d=768, depth=12, heads=12, joint=3),
}


@dataclass
class Op:
    name: str
    flops: float
    bytes: float  # minimum DRAM/L2 traffic assuming perfect reuse, bf16
    count: int = 1


def vit_eomt_ops(
    model: str,
    image: int = 512,
    patch: int = 16,
    queries: int = 100,
    n_prefix: int = 5,
    bytes_per_el: int = 2,
) -> list[Op]:
    c = CONFIGS[model]
    d, depth, joint = c["d"], c["depth"], c["joint"]
    g = image // patch
    n0 = g * g + n_prefix
    n1 = n0 + queries
    ops: list[Op] = [
        Op("patch_embed", 2 * g * g * (3 * patch * patch) * d,
           bytes_per_el * (3 * image * image + 3 * patch * patch * d + g * g * d)),
    ]
    for tag, n, cnt in (("block", n0, depth - joint), ("joint_block", n1, joint)):
        b = bytes_per_el
        ops += [
            Op(f"{tag}.qkv", 2 * n * d * 3 * d, b * (n * d + 3 * d * d + 3 * n * d), cnt),
            Op(f"{tag}.attn", 4 * n * n * d, b * (4 * n * d), cnt),
            Op(f"{tag}.proj", 2 * n * d * d, b * (2 * n * d + d * d), cnt),
            Op(f"{tag}.fc1", 2 * n * d * 4 * d, b * (n * d + 4 * d * d + 4 * n * d), cnt),
            Op(f"{tag}.fc2", 2 * n * 4 * d * d, b * (4 * n * d + 4 * d * d + n * d), cnt),
            Op(f"{tag}.elementwise(ln,add,gelu,rope)", 0, b * 12 * n * d, cnt),
        ]
    # Head: 2 ScaleBlocks (ConvT k2s2 + GELU + DW3x3 + LN2D), mask MLP, einsum.
    h = g
    for i in range(max(1, math.ceil(math.log2(patch)) - 2)):
        ops.append(Op(f"head.convT{i}", 2 * h * h * d * 4 * d,
                      bytes_per_el * (h * h * d + 4 * d * d + 4 * h * h * d)))
        h *= 2
        ops.append(Op(f"head.dw3x3_ln{i}", 2 * 9 * h * h * d, bytes_per_el * 2 * h * h * d))
    ops.append(Op("head.mask_mlp", 3 * 2 * queries * d * d, bytes_per_el * 3 * d * d))
    ops.append(Op("head.einsum", 2 * queries * d * h * h,
                  bytes_per_el * (queries * d + d * h * h + queries * h * h)))
    return ops


def total_gflop(model: str, **kw) -> float:
    return sum(o.flops * o.count for o in vit_eomt_ops(model, **kw)) / 1e9


def waves(m: int, n: int, bm: int = 128, bn: int = 128, sms: int = 170, ctas_per_sm: int = 1) -> dict:
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    slots = sms * ctas_per_sm
    w = tiles / slots
    return {"tiles": tiles, "waves": w, "util_upper_bound": tiles / (math.ceil(w) * slots)}


def efficiency(flops: float, nbytes: float, ms: float, peak_tflops: float, peak_gbps: float) -> dict:
    t = ms * 1e-3
    tf = flops / t / 1e12
    gb = nbytes / t / 1e9
    return {"tflops": tf, "gbps": gb, "pct_compute": 100 * tf / peak_tflops,
            "pct_bw": 100 * gb / peak_gbps}


if __name__ == "__main__":
    for m in CONFIGS:
        print(f"{m}: {total_gflop(m):.1f} GFLOP per 512x512 crop")
        for o in vit_eomt_ops(m):
            print(f"  {o.name:40s} x{o.count:2d}  {o.flops * o.count / 1e9:8.2f} GFLOP")
