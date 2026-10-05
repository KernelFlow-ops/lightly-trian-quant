"""Category table of a step-level nsys profile (``common/profile_nsys.sh`` prefix; reads <prefix>_cuda_gpu_kern_sum.csv and
<prefix>_analysis.json).  Kernel time per step = total / iterations (5 captured steps).

  python tests/nsys_step_categories.py <prefix> [--json out.json] [--top 12]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

CATS = [  # (category, regex on kernel name), first match wins
    ("quantize MX8 (own CUDA)", r"quant_dual_kernel|quant_[a-z_]*kernel"),
    ("GEMM MX8 (QuACK blockscaled)", r"MXF8|mxf8|BlockScaled|blockscaled|Float8E4M3|E4M3FN.*TiledMMA|sf_vec|SFA"),
    ("GEMM bf16 (QuACK)", r"quackgemm|quackepilogue|GemmDefaultSm120|GemmMod"),
    ("GEMM bf16 (cuBLAS / cutlass sm80)", r"cutlass_80_.*gemm|ampere_.*gemm|sm80_xmma_gemm|cublas|gemm_bf16|s16816gemm|Kernel2<cutlass_80|nvjet"),
    ("head conv (cuDNN / cutlass conv)", r"cudnn|nhwc|wgrad2d|dgrad2d|conv2d|implicit_gemm|fprop_optimized|dgrad_optimized|wgrad_optimized|im2col"),
    ("attention (flash / cuDNN SDPA / RoPE)", r"flash|fmha|sdpa|_rotate|rope|_fwd$|_bwd$|cudnn_generated|attention"),
    ("optimizer / grad clip", r"multi_tensor_apply|FusedAdam|clip|foreach|_amp_foreach|norm_kernel.*Tensor"),
    ("LN / row kernels (r13 CUDA)", r"rowbwd|ln_fwd|finalize|colsum|layer_norm|LayerNorm|layernorm|vectorized_layer"),
    ("loss / matcher / mask", r"bce|cross_entropy|softmax|sigmoid|hungarian|cdist|topk|sort"),
]


def classify(name: str) -> str:
    for cat, rx in CATS:
        if re.search(rx, name):
            return cat
    return "elementwise / copy / reduce / other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    pre = Path(args.prefix)
    ana = json.loads(Path(str(pre) + "_analysis.json").read_text()) if Path(str(pre) + "_analysis.json").exists() else {}
    iters = ana.get("iterations", 5) or 5
    rows = list(csv.DictReader(open(str(pre) + "_cuda_gpu_kern_sum.csv", newline="")))
    cat_ns = defaultdict(float)
    cat_n = defaultdict(float)
    top = []
    for r in rows:
        name = r["Name"]
        ns = float(r["Total Time (ns)"])
        n = float(r["Instances"])
        c = classify(name)
        cat_ns[c] += ns
        cat_n[c] += n
        top.append((ns, n, c, name))
    total = sum(cat_ns.values())
    print(f"{pre.name}: kernel time {total / iters / 1e6:.2f} ms/step, {sum(cat_n.values()) / iters:.0f} kernels/step"
          f"{', gpu_busy_ratio %.3f' % ana['gpu_busy_ratio'] if 'gpu_busy_ratio' in ana else ''}{', syncs/step %s' % ana['syncs_per_step'] if 'syncs_per_step' in ana else ''}")
    out = {"prefix": str(pre), "kernel_ms_per_step": total / iters / 1e6, "kernels_per_step": sum(cat_n.values()) / iters, "categories": {}}
    print("| category | ms/step | % | kernels/step |\n|---|---|---|---|")
    for c, ns in sorted(cat_ns.items(), key=lambda kv: -kv[1]):
        print(f"| {c} | {ns / iters / 1e6:.2f} | {100 * ns / total:.1f} | {cat_n[c] / iters:.0f} |")
        out["categories"][c] = {"ms_per_step": ns / iters / 1e6, "pct": 100 * ns / total, "kernels_per_step": cat_n[c] / iters}
    print("top kernels:")
    for ns, n, c, name in sorted(top, reverse=True)[: args.top]:
        short = re.sub(r"\s+", " ", name)[:120]
        print(f"  {n / iters:6.1f} x {ns / n / 1e3:8.1f} us  {ns / iters / 1e6:6.2f} ms/step  [{c.split(' (')[0]}]  {short}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
