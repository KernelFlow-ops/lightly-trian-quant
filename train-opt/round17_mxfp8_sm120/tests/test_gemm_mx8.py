"""Gate A (GEMM): round17 MXFP8 GEMMs + epilogues vs (a) the fp32 product of the DEQUANTIZED operands (kernel correctness) and
(b) the fp32 product of the original bf16 operands (quantization error), at the real training shapes incl. M tails, spiky data.

  python tests/test_gemm_mx8.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))

import torch
import torch.nn.functional as F

from common.bench import gpu_lock
from round17_mxfp8_sm120.impl import gemm_mx8 as G
from round17_mxfp8_sm120.impl import mx8
from quack.blockscaled.quantize import unpack_scale_blocked_to_2d


def _scale2d(sf, mn, sfk):
    s = unpack_scale_blocked_to_2d(sf.unsqueeze(0).contiguous(), mn, sfk)[0]
    return torch.exp2(s.view(torch.uint8).float() - 127.0)


def deq_row(t: mx8.MX8):
    R, C = t.rows, t.cols
    s = _scale2d(t.sfr, R, C // 32)
    return (t.qr.float().view(R, C // 32, 32) * s[:, :, None]).view(R, C)


def deq_col(t: mx8.MX8):
    """(Rpad, C) dequantized column copy (rows >= R are zero)."""
    s = _scale2d(t.sfc, t.cols, t.mpad // 32)  # (C, Rpad/32)
    return (t.qc.float().view(t.mpad // 32, 32, t.cols) * s.t()[:, None, :]).view(t.mpad, t.cols)


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def make(kind, shape, gen, scale=1.0):
    x = torch.randn(*shape, device="cuda", generator=gen)
    if kind == "spiky":
        x[:, 3] *= 2e3
        x[7, :] *= 60
        x[shape[0] // 2, shape[1] // 3] = 3e4
    elif kind == "scale8":
        x = x * 8
    return (x * scale).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cases = []
    if args.quick:
        cfgs = [("vitl16", 1024, 7203)]
    else:
        cfgs = [("vitl16", 1024, 7203), ("vitl16", 1024, 7903), ("vitb16", 768, 14406), ("vitb16", 768, 15806), ("vitl16", 1024, 1029)]
    gen = torch.Generator(device="cuda").manual_seed(1)
    rows = []
    with gpu_lock():
        for model, d, M in cfgs:
            lin = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (4 * d, d), "fc2": (d, 4 * d)}
            for data in ("normal", "spiky", "scale8") if not args.quick else ("normal", "spiky"):
                for name, (N, K) in lin.items():
                    X = make(data, (M, K), gen)
                    W = make("normal", (N, K), gen, 0.05)
                    dY = make(data, (M, N), gen)
                    Xq, Wq, dYq = mx8.quantize(X), mx8.quantize(W), mx8.quantize(dY)
                    Xr, Wr, Wc, dYr = deq_row(Xq), deq_row(Wq), deq_col(Wq), deq_row(dYq)
                    Xc, dYc = deq_col(Xq), deq_col(dYq)
                    bias = torch.randn(N, device="cuda", generator=gen) * 0.1
                    gamma = torch.rand(N, device="cuda", generator=gen) * 0.5 + 0.1
                    resid = torch.randn(M, N, device="cuda", generator=gen) * 5
                    res = {"model": model, "M": M, "data": data, "linear": name}
                    # ---- fwd
                    y = G.fwd_bias(Xq, Wq, bias)
                    ref_dq = Xr @ Wr.t() + bias
                    ref_fp = X.float() @ W.float().t() + bias
                    res["fwd_vs_deq"] = rel(y, ref_dq)
                    res["fwd_vs_fp32"] = rel(y, ref_fp)
                    res["fwd_cos_fp32"] = cos(y, ref_fp)
                    # ---- epilogues
                    out, yb = G.fwd_res(Xq, Wq, bias, gamma, resid)
                    res["res_vs_deq"] = max(rel(out, resid + gamma * ref_dq), rel(yb, ref_dq))
                    z, h = G.fwd_gelu(Xq, Wq, bias)
                    res["gelu_vs_deq"] = max(rel(z, ref_dq), rel(h, F.gelu(ref_dq)))
                    # ---- dgrad
                    dx = G.dgrad(dYq, Wq)
                    res["dgrad_vs_deq"] = rel(dx, dYr @ Wc)
                    res["dgrad_vs_fp32"] = rel(dx, dY.float() @ W.float())
                    zz = torch.randn(M, K, device="cuda", generator=gen).to(torch.bfloat16) * 2  # z has K columns only for fc2-shaped dgrad
                    if name == "fc2":  # dgrad of fc2: dY [M, d] @ W2 [d, 4d] -> [M, 4d], epilogue GELU'
                        dz, db = G.dgrad_dgelu(dYq, Wq, zz)
                        zf = zz.float().requires_grad_()
                        with torch.enable_grad():
                            dzr = torch.autograd.grad(F.gelu(zf), zf, dYr @ Wc)[0]
                        res["dgelu_vs_deq"] = rel(dz, dzr)
                        res["dgelu_db_vs_deq"] = rel(db, dzr.sum(0))
                    # ---- wgrad (contraction over zero-padded tokens)
                    dw = G.wgrad(dYq, Xq)
                    res["wgrad_vs_deq"] = rel(dw, dYc.t() @ Xc)
                    res["wgrad_vs_fp32"] = rel(dw, dY.float().t() @ X.float())
                    res["wgrad_cos_fp32"] = cos(dw, dY.float().t() @ X.float())
                    rows.append(res)
                    worst = max(v for k, v in res.items() if k.endswith("_vs_deq"))
                    print(f"{model} M={M} {data:6s} {name:4s} worst_vs_deq={worst:.2e} fwd_vs_fp32={res['fwd_vs_fp32']:.3e} dgrad_vs_fp32={res['dgrad_vs_fp32']:.3e} wgrad_vs_fp32={res['wgrad_vs_fp32']:.3e}", flush=True)
    bad = [r for r in rows if max(v for k, v in r.items() if k.endswith("_vs_deq")) > 5e-3]
    print(f"cases {len(rows)}, kernel-vs-dequantized failures (rel L2 > 5e-3): {len(bad)}")
    (ROUND / "profile" / ("gateA_gemm_quick.json" if args.quick else "gateA_gemm.json")).write_text(json.dumps({"rows": rows, "bad": bad}, indent=1))


if __name__ == "__main__":
    main()
