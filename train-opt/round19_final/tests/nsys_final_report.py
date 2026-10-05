"""round19 nsys report for graph-mode step profiles (5 captured steps, ``--cuda-graph-trace=node``).

Per prefix (``<prefix>_cuda_gpu_kern_sum.csv`` + ``<prefix>_cuda_gpu_trace.csv`` from round16's nsys_inner.sh):
  kernels/step, kernel ms/step, GPU busy ratio and >20 us gaps (round16 tests/trace_busy.py: window = first kernel start .. last kernel end),
  per-category ms/step (classifier below, tuned to the FINAL stack's kernel names; first match wins), top-N kernels.
  python tests/nsys_final_report.py --json profile/nsys_report.json --label L=<prefix> --label B=<prefix> [--top 15] [--other 12]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

TRACE_BUSY = Path("/root/lightly-train/train-opt/round16_full_graph/tests/trace_busy.py")

CATS = [
    ("MXFP8 quantize (own CUDA quant_dual)", r"quant_dual_kernel"),
    ("GEMM MXFP8 (QuACK block-scaled, backbone)", r"quack.*(1168321|ShapeMNK16832)|MXF8|BlockScaled"),
    ("GEMM bf16 QuACK backbone (qkv forward, bias epilogue)", r"quack.*bias_epi"),
    ("GEMM bf16 head (QuACK: ConvT fwd/dgrad, einsum bwd)", r"quack"),
    ("GEMM bf16 head (cutlass sm80: ConvT wgrad, einsum, mask head)", r"cutlass_80_|ampere_|nvjet|cublas|sm80_xmma_gemm|s16816gemm|gemm_bf16|Kernel2<cutlass"),
    ("r14 head fused kernels (GELU+dwconv+LN2D fwd/bwd)", r"^(void )?(fwd|bwd1|bwd2)_kernel|head_(fwd|bwd)"),
    ("conv (cuDNN: patch embed / residual head convs)", r"cudnn|nhwc|wgrad2d|dgrad2d|conv2d|implicit_gemm|fprop_optimized|dgrad_optimized|wgrad_optimized|im2col"),
    ("attention (flash fwd/bwd, RoPE)", r"flash|fmha|sdpa|_rotate|rope|attention"),
    ("optimizer (r15 flat: sumsq_prep + update)", r"update_kernel|sumsq_prep_kernel|multi_tensor_apply|FusedAdam|clip|foreach"),
    ("LN / row kernels (r13 CUDA + torch LN)", r"rowbwd|ln_fwd|finalize_kernel|colsum|layer_norm|LayerNorm|layernorm|vectorized_layer|GammaBeta"),
    ("loss / matcher / mask build", r"bce|cross_entropy|softmax|sigmoid|hungarian|cdist|topk|sort|grid_sampler|upsample_bilinear|point|mask"),
]


def classify(name: str) -> str:
    for cat, rx in CATS:
        if re.search(rx, name):
            return cat
    return "elementwise / copy / reduce / other"


def load_trace_busy():
    spec = importlib.util.spec_from_file_location("trace_busy", TRACE_BUSY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report(prefix: str, steps: int, top: int, other: int, busy_mod) -> dict:
    rows = list(csv.DictReader(open(prefix + "_cuda_gpu_kern_sum.csv", newline="")))
    cat_ns, cat_n = defaultdict(float), defaultdict(float)
    kernels = []
    for r in rows:
        ns, n, name = float(r["Total Time (ns)"]), float(r["Instances"]), r["Name"]
        c = classify(name)
        cat_ns[c] += ns
        cat_n[c] += n
        kernels.append((ns, n, c, name))
    total = sum(cat_ns.values())
    busy = busy_mod.analyze(prefix, steps=steps)
    api = {}
    try:
        for r in csv.DictReader(open(prefix + "_cuda_api_sum.csv", newline="")):
            api[r["Name"]] = int(r["Num Calls"])
    except OSError:
        pass
    nvtx = []
    try:
        for r in csv.DictReader(open(prefix + "_nvtx_sum.csv", newline="")):
            nvtx.append({"name": r.get("Range") or r.get("Name"), "instances": r.get("Instances"), "avg_ms": float(r.get("Avg (ns)", 0)) / 1e6})
    except OSError:
        pass
    out = {"prefix": prefix, "steps": steps, "kernel_ms_per_step": total / steps / 1e6, "kernels_per_step": sum(cat_n.values()) / steps, "busy": busy, "categories": {}, "top": [], "top_other": [],
           "sync_calls_total": sum(v for k, v in api.items() if re.search(r"Synchronize", k)), "sync_calls": {k: v for k, v in api.items() if re.search(r"Synchronize|GraphLaunch|MemcpyAsync|cudaMemcpy", k)},
           "nvtx": nvtx[:8]}
    for c, ns in sorted(cat_ns.items(), key=lambda kv: -kv[1]):
        out["categories"][c] = {"ms_per_step": ns / steps / 1e6, "pct": 100 * ns / total, "kernels_per_step": cat_n[c] / steps}
    for ns, n, c, name in sorted(kernels, reverse=True)[:top]:
        out["top"].append({"ms_per_step": ns / steps / 1e6, "pct": 100 * ns / total, "per_step": n / steps, "avg_us": ns / n / 1e3, "category": c, "name": re.sub(r"\s+", " ", name)[:160]})
    for ns, n, c, name in sorted([k for k in kernels if k[2].startswith("elementwise")], reverse=True)[:other]:
        out["top_other"].append({"ms_per_step": ns / steps / 1e6, "per_step": n / steps, "avg_us": ns / n / 1e3, "name": re.sub(r"\s+", " ", name)[:130]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True, help="LABEL=prefix")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--other", type=int, default=12)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    busy_mod = load_trace_busy()
    allr = {}
    for item in args.label:
        label, prefix = item.split("=", 1)
        allr[label] = report(prefix, args.steps, args.top, args.other, busy_mod)
    for label, r in allr.items():
        b = r["busy"]
        print(f"\n### {label}: {r['kernels_per_step']:.0f} kernels/step, kernel time {r['kernel_ms_per_step']:.2f} ms/step, window {b['span_ms_per_step']:.2f} ms/step, "
              f"GPU busy {b['busy_ratio']:.3f}, gaps>20us {b['gaps_gt20us_per_step']:.1f}/step = {b['gap_ms_per_step']:.2f} ms/step")
        print(f"sync-type API calls in the 5-step window: {r['sync_calls_total']} total {r['sync_calls']} (the window ends with one torch.cuda.synchronize); NVTX (CPU time of the host launch): "
              + ", ".join(f"{n['name']} {n['avg_ms']:.2f} ms" for n in r["nvtx"][:6]))
        print("| category | ms/step | % | kernels/step |\n|---|---:|---:|---:|")
        for c, v in r["categories"].items():
            print(f"| {c} | {v['ms_per_step']:.2f} | {v['pct']:.1f} | {v['kernels_per_step']:.0f} |")
        print(f"top {args.top} kernels:\n| # | ms/step | % | calls/step | avg us | category | kernel |\n|---|---:|---:|---:|---:|---|---|")
        for i, k in enumerate(r["top"], 1):
            print(f"| {i} | {k['ms_per_step']:.2f} | {k['pct']:.1f} | {k['per_step']:.1f} | {k['avg_us']:.1f} | {k['category'].split(' (')[0]} | {k['name'][:100]} |")
        print("largest unclassified (elementwise/other) kernels:")
        for k in r["top_other"]:
            print(f"  {k['per_step']:7.1f} x {k['avg_us']:8.1f} us = {k['ms_per_step']:5.2f} ms/step  {k['name']}")
    if args.json:
        args.json.write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
