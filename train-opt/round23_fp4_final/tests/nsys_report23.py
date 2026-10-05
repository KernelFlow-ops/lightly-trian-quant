"""round23 nsys category report: P3 vs FINAL (graph mode, mid phase, 5 captured steps).  The classifier of round19 (imported, unmodified) with the FP4 categories added in front
(round21 / round22 quantizer kernels act_kernel / w_kernel / amax roll, QuACK NVFP4 GEMMs = MMA atom m16n8k64 or *_ts epilogues); prints a side-by-side table.

  python tests/nsys_report23.py --label L_FINAL=<prefix> --label L_P3=<prefix> ... --json profile/nsys_report23.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

R19 = Path("/root/lightly-train/train-opt/round19_final/tests/nsys_final_report.py")
spec = importlib.util.spec_from_file_location("nsys19", R19)
n19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n19)

FP4_CATS = [
    ("FP4 quantize (R21 act_kernel / w_kernel, amax roll)", r"act_kernel|w_kernel|amax_roll|amax_abs"),
    ("GEMM NVFP4 (QuACK nv4 m16n8k64: head + backbone dgrad / wgrad)", r"ShapeMNK16864|GemmMod_(dgelu|bias|res|gelu|plain)_ts|quack.*(Fp4|FP4|Float4|float4|NVFP4|MXF4|1168641|e2m1)|nvfp4|mxf4"),
]
n19.CATS = FP4_CATS + n19.CATS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--top", type=int, default=16)
    args = ap.parse_args()
    busy_mod = n19.load_trace_busy()
    allr = {}
    for item in args.label:
        label, prefix = item.split("=", 1)
        allr[label] = n19.report(prefix, args.steps, args.top, 14, busy_mod)
    cats = sorted({c for r in allr.values() for c in r["categories"]})
    print("| category | " + " | ".join(allr) + " |")
    print("|---|" + "---:|" * len(allr))
    for c in sorted(cats, key=lambda c: -max(r["categories"].get(c, {"ms_per_step": 0})["ms_per_step"] for r in allr.values())):
        print(f"| {c} | " + " | ".join(f"{r['categories'].get(c, {'ms_per_step': 0.0})['ms_per_step']:.2f}" for r in allr.values()) + " |")
    print("| **kernel total ms/step** | " + " | ".join(f"{r['kernel_ms_per_step']:.2f}" for r in allr.values()) + " |")
    print("| kernels/step | " + " | ".join(f"{r['kernels_per_step']:.0f}" for r in allr.values()) + " |")
    print("| GPU busy ratio | " + " | ".join(f"{r['busy']['busy_ratio']:.3f}" for r in allr.values()) + " |")
    print("| window ms/step | " + " | ".join(f"{r['busy']['span_ms_per_step']:.2f}" for r in allr.values()) + " |")
    for label, r in allr.items():
        print(f"\n### {label} top {args.top} kernels\n| # | ms/step | calls/step | avg us | category | kernel |\n|---|---:|---:|---:|---|---|")
        for i, k in enumerate(r["top"], 1):
            print(f"| {i} | {k['ms_per_step']:.2f} | {k['per_step']:.1f} | {k['avg_us']:.1f} | {k['category'].split(' (')[0]} | {k['name'][:110]} |")
        print("largest unclassified:")
        for k in r["top_other"][:8]:
            print(f"  {k['per_step']:7.1f} x {k['avg_us']:8.1f} us = {k['ms_per_step']:5.2f} ms/step  {k['name'][:110]}")
    if args.json:
        args.json.write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
