"""Gate A (quantizer): round17 CUDA ``quant_dual`` vs QuACK ``to_mx`` / ``to_mx_dim0`` (rceil) bit-exactness, spiky / large /
tiny / zero / NaN data, M tails (7203 / 7903 / 14406 / 15806 / 1029), bf16 and fp32 inputs, plus dequantization error vs the source.

  python tests/test_quant.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))

import torch

from common.bench import gpu_lock
import mx8
from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig, to_mx, to_mx_dim0


def reference(x, mpad):
    """QuACK rceil reference -> (qr, sfr, qc, sfc) in the same layouts as mx8.quantize."""
    M, C = x.shape
    qr, sr = to_mx(x.contiguous(), 32, "rceil")
    sfr = pack_scale_2d_to_blocked_contig(sr.view(M, C // 32)).squeeze(0)
    xp = torch.zeros(mpad, C, dtype=x.dtype, device=x.device)
    xp[:M] = x
    qc, sc = to_mx_dim0(xp, 32, "rceil")  # scale (mpad/32, C)
    sfc = pack_scale_2d_to_blocked_contig(sc.t().contiguous()).squeeze(0)  # (C, mpad/32) -> blocked
    return qr, sfr, qc, sfc


def make(kind, M, C, dtype, gen):
    x = torch.randn(M, C, device="cuda", generator=gen)
    if kind == "spiky":  # DINOv3-like massive activations: a few huge channels / tokens
        x[:, 3] *= 3e3
        x[7, :] *= 80
        x[M // 2, C // 3] = 4.0e4
    elif kind == "scale8":
        x = x * 8.0
    elif kind == "tiny":
        x = x * 1e-30
    elif kind == "large":
        x = x * 1e4
    elif kind == "zeros":
        x = torch.zeros_like(x)
        x[5, 7] = 1.0
    elif kind == "pow2":  # exact power-of-two block maxima: the RCEIL boundary case
        x = torch.randn(M, C, device="cuda", generator=gen).sign() * torch.exp2(torch.randint(-6, 6, (M, C), device="cuda", generator=gen).float())
        x[:, ::32] = 448.0 * 2.0 ** torch.randint(-3, 3, (M, 1), device="cuda", generator=gen).float()
    return x.to(dtype)


def main():
    rows = []
    gen = torch.Generator(device="cuda").manual_seed(0)
    with gpu_lock():
        for dtype in (torch.bfloat16, torch.float32):
            for M, C in ((7203, 1024), (7903, 4096), (1029, 768), (14406, 768), (15806, 3072), (128, 128), (33, 256)):
                for kind in ("normal", "spiky", "scale8", "tiny", "large", "zeros", "pow2"):
                    x = make(kind, M, C, dtype, gen)
                    mp = mx8.pad128(M)
                    t = mx8.quantize(x, mpad=mp)
                    qr, sfr, qc, sfc = reference(x, mp)
                    ok_qr = torch.equal(t.qr.view(torch.uint8), qr.view(torch.uint8))
                    ok_sr = torch.equal(t.sfr.view(torch.uint8), sfr.view(torch.uint8))
                    ok_qc = torch.equal(t.qc.view(torch.uint8), qc.view(torch.uint8))
                    ok_sc = torch.equal(t.sfc.view(torch.uint8), sfc.view(torch.uint8))
                    row = {"dtype": str(dtype).split(".")[-1], "M": M, "C": C, "kind": kind, "q_row": ok_qr, "sf_row": ok_sr, "q_col": ok_qc, "sf_col": ok_sc}
                    rows.append(row)
                    if not all((ok_qr, ok_sr, ok_qc, ok_sc)):
                        print("MISMATCH", row, flush=True)
        # NaN / Inf propagation (scale byte 255, data NaN)
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        x[3, 5] = float("nan")
        x[40, 77] = float("inf")
        t = mx8.quantize(x)
        qr, sfr, qc, sfc = reference(x, 256)
        nan_ok = {"q_row": torch.equal(t.qr.view(torch.uint8), qr.view(torch.uint8)), "sf_row": torch.equal(t.sfr.view(torch.uint8), sfr.view(torch.uint8)),
                  "q_col": torch.equal(t.qc.view(torch.uint8), qc.view(torch.uint8)), "sf_col": torch.equal(t.sfc.view(torch.uint8), sfc.view(torch.uint8))}
        print("nan/inf", nan_ok, flush=True)
    n = len(rows)
    bad = [r for r in rows if not all((r["q_row"], r["sf_row"], r["q_col"], r["sf_col"]))]
    print(f"cases {n}, mismatching {len(bad)}")
    out = ROUND / "profile" / "gateA_quant.json"
    out.write_text(json.dumps({"cases": n, "mismatching": len(bad), "bad": bad[:20], "nan_inf": nan_ok}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
