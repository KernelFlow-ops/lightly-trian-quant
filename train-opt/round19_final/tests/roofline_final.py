"""Roofline-style estimate of what is left in the FINAL stack (graph mode, mid phase), per category: measured nsys ms/step (tests/nsys_final_report.py json)
vs an ANALYTICAL floor.  Everything here is an estimate from FLOP / byte counts and stated peak rates, not a measurement of a better kernel.

Assumptions (stated, from CONTRACTS / round14-17 measurements):
  BF16 HMMA (fp32 acc) 230 TFLOPS, MXFP8 block-scaled 600 TFLOPS (real GEMMs measured 314-638 in r17), DRAM 1.55 TB/s (r15 measured 1.5-1.65).
  backbone Linear FLOPs: 12 d^2 params/block (qkv 3d^2, proj d^2, fc1 4d^2, fc2 4d^2), 2*M*params per pass, 3 passes (fwd, dgrad, wgrad);
  policy fc+proj+qkv_dw: only the qkv FORWARD (3d^2 of 12d^2, one of three passes) stays bf16.
  attention: fwd 4*B*N^2*d, bwd 2.5x fwd (flash recomputes S); N = 1029 plain, 1129 joint (last 3 blocks).
  head (4 predicts): ConvT1 [B*32^2,C]x[C,4C], ConvT2 [B*64^2,C]x[C,4C], einsum [B,100,C]x[B,C,128^2], mask-head MLP; x3 passes (r14 NOTES: 4.3 ms/predict at 230 TFLOPS for ViT-L).
  optimizer: 36 B/float of DRAM traffic (r15 NOTES section 1: K1 reads g, K2 reads p/g/m/v and writes p/m/v/g).

  python tests/roofline_final.py --report profile/nsys_report.json --json profile/roofline_final.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CFG = {"vitl16": {"d": 1024, "L": 24, "J": 3, "B": 7, "params": 315e6, "label": "L"}, "vitb16": {"d": 768, "L": 12, "J": 3, "B": 14, "params": 86e6 * 1.0, "label": "B"}}
BF16, MX8, BW = 230e12, 600e12, 1.55e12
# floats in the flat optimizer buffers (r15 notes: ViT-L 3.15e8 floats; ViT-B scaled by the measured category time is not used: computed from model params)


def backbone(model: str):
    c = CFG[model]
    d, B = c["d"], c["B"]
    plain, joint = c["L"] - c["J"], c["J"]
    m_sum = B * (plain * 1029 + joint * 1129)                   # sum over blocks of M
    per_pass = 2 * 12 * d * d * m_sum                           # FLOPs of one pass over all four Linears of all blocks
    bf16 = 2 * 3 * d * d * m_sum                                # qkv forward
    mx8 = 3 * per_pass - bf16
    attn = 3.5 * 4 * B * d * (plain * 1029**2 + joint * 1129**2)
    return {"gemm_total_tflop": 3 * per_pass / 1e12, "gemm_mx8_tflop": mx8 / 1e12, "gemm_bf16_tflop": bf16 / 1e12,
            "gemm_floor_ms": (mx8 / MX8 + bf16 / BF16) * 1e3, "gemm_floor_all_bf16_ms": 3 * per_pass / BF16 * 1e3, "gemm_floor_all_mx8_ms": 3 * per_pass / MX8 * 1e3,
            "attn_tflop": attn / 1e12, "attn_floor_ms": attn / BF16 * 1e3}


def head(model: str):
    c = CFG[model]
    d, B, Q = c["d"], c["B"], 100
    fwd = 2 * B * 32**2 * d * 4 * d + 2 * B * 64**2 * d * 4 * d + 2 * B * Q * d * 128**2 + 2 * B * Q * d * d * 3
    tot = 4 * 3 * fwd
    return {"gemm_tflop": tot / 1e12, "gemm_floor_bf16_ms": tot / BF16 * 1e3, "gemm_floor_mx8_ms": tot / MX8 * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--opt-floats", type=float, nargs=2, default=[3.15e8, 8.7e7], help="ViT-L / ViT-B parameter floats (flat buffer size, r15)")
    args = ap.parse_args()
    rep = json.loads(args.report.read_text())
    out = {}
    for label, model in (("L", "vitl16"), ("B", "vitb16")):
        if label not in rep:
            continue
        cats = {k: v["ms_per_step"] for k, v in rep[label]["categories"].items()}

        def cat(prefix):
            return sum(v for k, v in cats.items() if k.startswith(prefix))

        bb, hd = backbone(model), head(model)
        opt_floor = args.opt_floats[0 if model == "vitl16" else 1] * 36 / BW * 1e3
        head_meas = cat("GEMM bf16 head")
        rows = [
            ("backbone GEMM (MXFP8 + bf16 qkv-fwd, QuACK)", cat("GEMM MXFP8") + cat("GEMM bf16 QuACK backbone"), bb["gemm_floor_ms"], "MXFP8 600 / BF16 230 TFLOPS"),
            ("attention (flash fwd/bwd + RoPE)", cat("attention"), bb["attn_floor_ms"], "BF16 230 TFLOPS, RoPE excluded from the floor"),
            ("head GEMMs (ConvT x2 fwd/dgrad/wgrad, einsum, mask head)", head_meas, hd["gemm_floor_bf16_ms"], "BF16 230 TFLOPS (MXFP8 ConvT would be %.1f ms)" % hd["gemm_floor_mx8_ms"]),
            ("r14 fused head kernels (GELU+dwconv+LN2D)", cat("r14 head"), float("nan"), "DRAM-bound (r14: 72-78% of DRAM BW)"),
            ("optimizer (r15)", cat("optimizer"), opt_floor, "DRAM 1.55 TB/s x 36 B/float"),
            ("MXFP8 quantize", cat("MXFP8 quantize"), float("nan"), "DRAM-bound; removable only by fusing into producers"),
            ("LN / row kernels", cat("LN / row"), float("nan"), "DRAM-bound"),
            ("elementwise / copy / other", cat("elementwise"), float("nan"), "DRAM-bound"),
            ("loss / matcher / mask build", cat("loss"), float("nan"), ""),
            ("cuDNN conv (patch embed)", cat("conv"), float("nan"), ""),
        ]
        out[model] = {"kernel_ms_per_step": rep[label]["kernel_ms_per_step"], "window_ms_per_step": rep[label]["busy"]["span_ms_per_step"], "rows": [
            {"part": r[0], "measured_ms": None if r[1] != r[1] else r[1], "floor_ms": None if r[2] != r[2] else r[2], "basis": r[3]} for r in rows], "backbone": bb, "head": hd}
        print(f"\n### {model}: kernel {rep[label]['kernel_ms_per_step']:.1f} ms/step, window {rep[label]['busy']['span_ms_per_step']:.1f} ms/step")
        print("| part | measured ms | analytic floor ms | basis |\n|---|---:|---:|---|")
        for r in rows:
            print(f"| {r[0]} | {'-' if r[1] != r[1] else '%.2f' % r[1]} | {'-' if r[2] != r[2] else '%.2f' % r[2]} | {r[3]} |")
        print(f"backbone Linear {bb['gemm_total_tflop']:.1f} TFLOP/step (MXFP8 {bb['gemm_mx8_tflop']:.1f}, bf16 {bb['gemm_bf16_tflop']:.1f}); all-MXFP8 floor {bb['gemm_floor_all_mx8_ms']:.1f} ms, all-bf16 {bb['gemm_floor_all_bf16_ms']:.1f} ms; "
              f"attention {bb['attn_tflop']:.2f} TFLOP; head GEMMs {hd['gemm_tflop']:.2f} TFLOP/step = {hd['gemm_floor_bf16_ms']:.1f} ms bf16 / {hd['gemm_floor_mx8_ms']:.1f} ms mx8")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
