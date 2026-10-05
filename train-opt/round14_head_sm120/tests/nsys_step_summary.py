"""Step-level nsys summary for round14 (reads <prefix>_cuda_gpu_kern_sum.csv, <prefix>_nvtx_gpu_proj_sum.csv, <prefix>_analysis.json).

  python tests/nsys_step_summary.py <prefix> [--json out.json] [--top 10]
Prints: kernel time / kernels / GPU busy / syncs / gaps per step, the category table (incl. the r14 head kernels), the NVTX GPU-projection
rows that matter (step, fwd, bwd, predict_*, r14_up_fwd / r14_up_bwd) and the top kernels.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

CATS = [  # (category, regex on kernel name), first match wins
    ("head stage kernels (r14 CUDA)", r"^void (fwd|bwd1|bwd2)_kernel<"),
    ("quantize MX8 (own CUDA)", r"quant_dual_kernel|quant_[a-z_]*kernel"),
    ("GEMM MX8 (QuACK blockscaled)", r"MXF8|mxf8|BlockScaled|blockscaled|Float8E4M3|E4M3FN.*TiledMMA|sf_vec|SFA"),
    ("GEMM bf16 (QuACK)", r"quackgemm|quackepilogue|GemmDefaultSm120|GemmMod"),
    ("GEMM bf16 (cuBLAS / cutlass sm80)", r"cutlass_80_.*gemm|ampere_.*gemm|sm80_xmma_gemm|cublas|gemm_bf16|s16816gemm|Kernel2<cutlass_80|nvjet"),
    ("head conv (cuDNN / cutlass conv)", r"cudnn|nhwc|wgrad2d|dgrad2d|conv2d|implicit_gemm|fprop_optimized|dgrad_optimized|wgrad_optimized|im2col"),
    ("attention (flash / cuDNN SDPA / RoPE)", r"flash|fmha|sdpa|_rotate|rope|_fwd$|_bwd$|cudnn_generated|attention"),
    ("optimizer / grad clip", r"multi_tensor_apply|FusedAdam|clip|foreach|_amp_foreach|norm_kernel.*Tensor"),
    ("LN / row kernels (r13 CUDA, torch LN)", r"rowbwd|ln_fwd|finalize|colsum|layer_norm|LayerNorm|layernorm|vectorized_layer|GammaBeta"),
    ("loss / matcher / mask", r"bce|cross_entropy|softmax|sigmoid|hungarian|cdist|topk|sort"),
]
NVTX_KEEP = re.compile(r"^:(step_\d+|fwd|bwd|loss|clip_opt_sched|predict_\w+|r14_up_fwd|r14_up_bwd|blk_00|blk_12|blk_22|mask_L\d+)$")


def classify(name: str) -> str:
    for cat, rx in CATS:
        if re.search(rx, name):
            return cat
    return "elementwise / copy / reduce / other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    pre = Path(args.prefix)
    ana_p = Path(str(pre) + "_analysis.json")
    ana = json.loads(ana_p.read_text()) if ana_p.exists() else {}
    iters = ana.get("iterations", 5) or 5
    rows = list(csv.DictReader(open(str(pre) + "_cuda_gpu_kern_sum.csv", newline="")))
    cat_ns, cat_n, top = defaultdict(float), defaultdict(float), []
    for r in rows:
        ns, n = float(r["Total Time (ns)"]), float(r["Instances"])
        c = classify(r["Name"])
        cat_ns[c] += ns
        cat_n[c] += n
        top.append((ns, n, c, r["Name"]))
    total = sum(cat_ns.values())
    out = {"prefix": str(pre), "iterations": iters, "kernel_ms_per_step": total / iters / 1e6, "kernels_per_step": sum(cat_n.values()) / iters,
           "gpu_busy_ratio": ana.get("gpu_busy_ratio"), "syncs_per_step": ana.get("syncs_per_step"), "categories": {}, "nvtx_gpu_ms": {}}
    print(f"== {pre.name}: kernel time {out['kernel_ms_per_step']:.2f} ms/step, {out['kernels_per_step']:.0f} kernels/step, gpu_busy {out['gpu_busy_ratio']}, syncs/step {out['syncs_per_step']}")
    print("| category | ms/step | % | kernels/step |\n|---|---:|---:|---:|")
    for c, ns in sorted(cat_ns.items(), key=lambda kv: -kv[1]):
        print(f"| {c} | {ns / iters / 1e6:.2f} | {100 * ns / total:.1f} | {cat_n[c] / iters:.0f} |")
        out["categories"][c] = {"ms_per_step": ns / iters / 1e6, "pct": 100 * ns / total, "kernels_per_step": cat_n[c] / iters}
    nv = Path(str(pre) + "_nvtx_gpu_proj_sum.csv")
    if nv.exists():
        print("NVTX GPU projection (avg per instance, ms; sum per step, ms):")
        per = defaultdict(lambda: [0.0, 0])
        for r in csv.DictReader(open(nv, newline="")):
            name = r["Range"]
            if NVTX_KEEP.match(name):
                key = re.sub(r"step_\d+", "step_*", name)
                per[key][0] += float(r["Total Proj Time (ns)"])
                per[key][1] += int(r["Range Instances"])
        for k, (ns, n) in sorted(per.items()):
            print(f"   {k:16s} n={n:3d}  avg {ns / n / 1e6:7.3f} ms   per step {ns / iters / 1e6:7.3f} ms")
            out["nvtx_gpu_ms"][k] = {"avg_ms": ns / n / 1e6, "per_step_ms": ns / iters / 1e6, "n": n}
    print("top kernels:")
    for ns, n, c, name in sorted(top, reverse=True)[: args.top]:
        print(f"  {n / iters:6.1f} x {ns / n / 1e3:8.1f} us  {ns / iters / 1e6:6.2f} ms/step  [{c.split(' (')[0]}]  {re.sub(r'\s+', ' ', name)[:110]}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
