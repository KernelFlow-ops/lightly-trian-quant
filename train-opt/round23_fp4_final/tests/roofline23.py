"""Roofline-style estimate of what is left in the phase-3 stack P3 (graph mode, mid phase), per category: measured nsys ms/step (tests/nsys_report23.py json) vs an ANALYTICAL floor.
Method of round19_final/tests/roofline_final.py extended to the FP4 operations.  Everything here is an estimate from FLOP / byte counts and stated peak rates, not a measurement of a
better kernel.

Assumptions (stated; from CONTRACTS / round14-22 measurements):
  BF16 HMMA (fp32 acc) 230 TFLOPS; MXFP8 block-scaled 600 TFLOPS (real GEMMs 314-638 in r17); NVFP4 block-scaled 1.2 PFLOPS sustained (R20a / R21b / R22 GEMMs: 0.7-1.25 PFLOPS;
  instruction peak 2.03 PFLOPS is given as the hard lower bound); DRAM 1.55 TB/s.
  backbone Linear FLOPs: 12 d^2 params / block, 2 M params per pass, 3 passes.  P3 policy: forward qkv bf16 (3d^2 of 12d^2), proj / fc1 / fc2 MXFP8; dgrad + wgrad NVFP4 (ViT-L: block 0
  stays MXFP8 for dgrad / wgrad because of ``nofirst``: 1/24 of those FLOPs at 600 TFLOPS).
  head (4 predicts): ConvT1 [B*32^2,C]x[C,4C] and ConvT2 [B*64^2,C]x[C,4C]: forward MXFP8, dgrad + wgrad NVFP4; einsum [B,100,C]x[B,C,128^2] and the mask-head MLP stay bf16 (3 passes).
  attention: fwd 4*B*N^2*d, bwd 2.5x fwd; N = 1029 plain, 1129 joint (last 3 blocks).  optimizer: 36 B/float of DRAM traffic (r15 NOTES).
  FP4 / MXFP8 quantization floor: bytes of the first-touch (cold, DRAM) traffic of the quantizer kernels: forward X read bf16 + writes (mx8 row 1.03 B/elem for proj / fc1 / fc2, nv4 transposed copy
  0.53 B/elem for all four), backward dY read bf16 + nv4 row + col copies (3.06 B/elem), weights fp32 read + mx8 / nv4 copies; the real inputs are mostly L2 resident, so this floor is loose.

  python tests/roofline23.py --report profile/nsys_report23.json --json profile/roofline23.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CFG = {"vitl16": {"d": 1024, "L": 24, "J": 3, "B": 7, "label": "L", "nofirst": True, "opt_floats": 3.15e8},
       "vitb16": {"d": 768, "L": 12, "J": 3, "B": 14, "label": "B", "nofirst": False, "opt_floats": 8.7e7}}
BF16, MX8, NV4, NV4_PEAK, BW = 230e12, 600e12, 1.2e15, 2.029e15, 1.55e12


def backbone(model: str) -> dict:
    c = CFG[model]
    d, B, L = c["d"], c["B"], c["L"]
    plain, joint = L - c["J"], c["J"]
    m_sum = B * (plain * 1029 + joint * 1129)
    per_pass = 2 * 12 * d * d * m_sum
    bf16 = 2 * 3 * d * d * m_sum                                  # qkv forward
    mx8_fwd = per_pass - bf16
    nv_frac = (L - 1) / L if c["nofirst"] else 1.0
    nv4 = 2 * per_pass * nv_frac
    mx8_bwd = 2 * per_pass * (1 - nv_frac)
    attn = 3.5 * 4 * B * d * (plain * 1029**2 + joint * 1129**2)
    # quantization traffic (bytes), per block, tokens M
    M = m_sum / L
    q_fwd = M * d * (2 + 0.53) + 3 * M * d * (2 + 1.03 + 0.53) + 0                       # qkv X (nv4 col only), proj X, fc1 X (mx8 row + nv4 col)
    q_fwd += M * 4 * d * (2 + 1.03 + 0.53)                                               # fc2 X
    q_bwd = (M * 3 * d + M * d + M * 4 * d + M * d) * (2 + 1.06)                         # dY of qkv / proj / fc1 / fc2: bf16 read + nv4 row + col
    q_w = 12 * d * d * (4 + 1.03 + 0.53 + 0.53)                                          # fp32 weights -> mx8 row + nv4 q + qt
    q_bytes = (q_fwd + q_bwd + q_w) * L
    return {"gemm_total_tflop": 3 * per_pass / 1e12, "bf16_fwd_tflop": bf16 / 1e12, "mx8_fwd_tflop": mx8_fwd / 1e12, "nv4_tflop": nv4 / 1e12, "mx8_bwd_tflop": mx8_bwd / 1e12,
            "floor_ms": (bf16 / BF16 + (mx8_fwd + mx8_bwd) / MX8 + nv4 / NV4) * 1e3, "floor_nv4_part_ms": nv4 / NV4 * 1e3, "floor_nv4_part_peak_ms": nv4 / NV4_PEAK * 1e3,
            "floor_mx8_part_ms": (mx8_fwd + mx8_bwd) / MX8 * 1e3, "floor_bf16_part_ms": bf16 / BF16 * 1e3,
            "attn_tflop": attn / 1e12, "attn_floor_ms": attn / BF16 * 1e3, "quant_bytes_gb": q_bytes / 1e9, "quant_floor_ms": q_bytes / BW * 1e3}


def head(model: str) -> dict:
    c = CFG[model]
    d, B, Q = c["d"], c["B"], 100
    convt = 4 * (2 * B * 32**2 * d * 4 * d + 2 * B * 64**2 * d * 4 * d)                 # one pass of both ConvT over the 4 predicts
    other = 4 * (2 * B * Q * d * 128**2 + 2 * B * Q * d * d * 3)                         # einsum + mask-head MLP (one pass)
    return {"convt_pass_tflop": convt / 1e12, "other_pass_tflop": other / 1e12,
            "floor_ms": (convt / MX8 + 2 * convt / NV4 + 3 * other / BF16) * 1e3, "floor_convt_mx8_ms": convt / MX8 * 1e3, "floor_convt_nv4_ms": 2 * convt / NV4 * 1e3, "floor_other_bf16_ms": 3 * other / BF16 * 1e3,
            "nv4_tflop": 2 * convt / 1e12, "mx8_tflop": convt / 1e12, "bf16_tflop": 3 * other / 1e12}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--which", default="P3", help="label suffix of the stack in the nsys report (L_P3 / B_P3)")
    args = ap.parse_args()
    rep = json.loads(args.report.read_text())
    out = {}
    for model, c in CFG.items():
        key = f"{c['label']}_{args.which}"
        if key not in rep:
            continue
        cats = {k: v["ms_per_step"] for k, v in rep[key]["categories"].items()}

        def cat(prefix):
            return sum(v for k, v in cats.items() if k.startswith(prefix))

        bb, hd = backbone(model), head(model)
        opt_floor = c["opt_floats"] * 36 / BW * 1e3
        gemm_nv4_floor = bb["floor_nv4_part_ms"] + hd["floor_convt_nv4_ms"]
        gemm_mx8_floor = bb["floor_mx8_part_ms"] + hd["floor_convt_mx8_ms"]
        gemm_bf16_floor = bb["floor_bf16_part_ms"] + hd["floor_other_bf16_ms"]
        meas_bf16 = cat("GEMM bf16")
        rows = [
            ("GEMM NVFP4 (backbone dgrad / wgrad + head ConvT dgrad / wgrad)", cat("GEMM NVFP4"), gemm_nv4_floor, "NVFP4 1.2 PFLOPS sustained (peak 2.03 PFLOPS would be %.1f ms)" % (bb["floor_nv4_part_peak_ms"] + hd["floor_convt_nv4_ms"] * NV4 / NV4_PEAK)),
            ("GEMM MXFP8 (backbone proj / fc1 / fc2 forward + head ConvT forward)", cat("GEMM MXFP8"), gemm_mx8_floor, "MXFP8 600 TFLOPS"),
            ("GEMM bf16 (qkv forward, einsum, mask head)", meas_bf16, gemm_bf16_floor, "BF16 230 TFLOPS"),
            ("attention (flash fwd/bwd + RoPE)", cat("attention"), bb["attn_floor_ms"], "BF16 230 TFLOPS, RoPE excluded from the floor"),
            ("FP4 + MXFP8 quantization (act_kernel / w_kernel / quant_dual)", cat("FP4 quantize") + cat("MXFP8 quantize"), bb["quant_floor_ms"], "DRAM 1.55 TB/s on cold first-touch bytes (loose: inputs mostly in L2)"),
            ("r14 fused head kernels (GELU+dwconv+LN2D)", cat("r14 head"), float("nan"), "DRAM-bound"),
            ("optimizer (r15)", cat("optimizer"), opt_floor, "DRAM 1.55 TB/s x 36 B/float"),
            ("LN / row kernels", cat("LN / row"), float("nan"), "DRAM-bound"),
            ("elementwise / copy / other", cat("elementwise"), float("nan"), "DRAM-bound"),
            ("loss / matcher / mask build", cat("loss"), float("nan"), ""),
            ("cuDNN conv (patch embed)", cat("conv"), float("nan"), ""),
        ]
        kern = rep[key]["kernel_ms_per_step"]
        out[model] = {"stack": key, "kernel_ms_per_step": kern, "window_ms_per_step": rep[key]["busy"]["span_ms_per_step"], "rows": [
            {"part": r[0], "measured_ms": None if r[1] != r[1] else r[1], "floor_ms": None if r[2] != r[2] else r[2], "basis": r[3]} for r in rows], "backbone": bb, "head": hd}
        print(f"\n### {model} ({key}): kernel {kern:.1f} ms/step, window {rep[key]['busy']['span_ms_per_step']:.1f} ms/step")
        print("| part | measured ms | % of kernel time | analytic floor ms | basis |\n|---|---:|---:|---:|---|")
        for r in sorted(rows, key=lambda r: -(r[1] if r[1] == r[1] else 0)):
            print(f"| {r[0]} | {'-' if r[1] != r[1] else '%.2f' % r[1]} | {'-' if r[1] != r[1] else '%.1f' % (100 * r[1] / kern)} | {'-' if r[2] != r[2] else '%.2f' % r[2]} | {r[3]} |")
        print(f"backbone Linear {bb['gemm_total_tflop']:.1f} TFLOP/step (bf16 {bb['bf16_fwd_tflop']:.1f}, MXFP8 {bb['mx8_fwd_tflop'] + bb['mx8_bwd_tflop']:.1f}, NVFP4 {bb['nv4_tflop']:.1f}); attention {bb['attn_tflop']:.2f} TFLOP; "
              f"head ConvT {hd['mx8_tflop']:.1f} TFLOP MXFP8 + {hd['nv4_tflop']:.1f} TFLOP NVFP4, bf16 einsum / mask head {hd['bf16_tflop']:.1f} TFLOP; quantizer cold bytes {bb['quant_bytes_gb']:.1f} GB")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
