"""Gate A for the round13 elementwise kernels: vs fp32 PyTorch references incl. spiky (massive-activation) rows.

Run:  python tests/test_block_kernels.py [--json profile/gateA_kernels.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "impl"))

import torch
import torch.nn.functional as F

import block_kernels as bk
from common.bench import gpu_lock
from mb import err_stats


def spiky(M, D, dev, seed=0, spike=True):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(M, D, device=dev, generator=g) * 2.0
    if spike:
        # DINOv3 "massive activations": a few channels reach ~1.5e5 from layer 0 on every token, a few tokens are huge
        x[:, 7] = 1.5e5
        x[:, D // 2 + 3] = -8.0e4
        x[::97] *= 30.0
    return x


def rows(name, err, ok_cos=0.999999):
    return {"name": name, **err, "pass": bool(err["cos"] >= ok_cos)}


def run_case(M, D, spike, eps=1e-6):
    dev = "cuda"
    out = []
    torch.manual_seed(1)
    x = spiky(M, D, dev, 0, spike)
    w = torch.rand(D, device=dev) + 0.5
    b = torch.randn(D, device=dev) * 0.1
    y, mean, rstd = bk.ln_fwd(x, w, b, eps)
    xr = x.clone().requires_grad_()
    yref = F.layer_norm(xr, (D,), w, b, eps)
    ybf = yref.detach().to(torch.bfloat16)
    out.append(rows("ln_fwd.y(bf16 vs fp32 ref)", err_stats(y, yref), ok_cos=0.999995))
    mism = float((y != ybf).float().mean())
    out.append({"name": "ln_fwd.y_exact_bf16_mismatch_frac", "mismatch_frac": mism, "pass": mism < 2e-3})
    out.append(rows("ln_fwd.mean", err_stats(mean, x.mean(-1))))
    out.append(rows("ln_fwd.rstd", err_stats(rstd, torch.rsqrt(x.var(-1, unbiased=False) + eps))))

    # ---- LN2 bwd + residual + gamma prep (proj side)
    dy = (torch.randn(M, D, device=dev) * 0.1).to(torch.bfloat16)
    dres = torch.randn(M, D, device=dev) * 0.05
    gamma = torch.rand(D, device=dev) * 1e-2
    y1 = torch.randn(M, D, device=dev).to(torch.bfloat16)
    wr = w.clone().requires_grad_(); br = b.clone().requires_grad_()
    yref = F.layer_norm(xr, (D,), wr, br, eps)
    yref.backward(dy.float())
    dx_ref = xr.grad + dres
    dx, g, sums = bk.ln_bwd_res(x, mean, rstd, w, dy, dres, y=y1, gamma=gamma)
    out.append(rows("rowbwd(ln+gam).dx", err_stats(dx, dx_ref)))
    out.append(rows("rowbwd(ln+gam).g(bf16)", err_stats(g, (gamma * dx_ref).to(torch.bfloat16))))
    out.append(rows("rowbwd(ln+gam).dW", err_stats(sums[0], wr.grad)))
    out.append(rows("rowbwd(ln+gam).dB", err_stats(sums[1], br.grad)))
    out.append(rows("rowbwd(ln+gam).dGamma", err_stats(sums[2], (dx_ref * y1.float()).sum(0))))
    out.append(rows("rowbwd(ln+gam).dBiasBranch", err_stats(sums[3], (gamma * dx_ref).sum(0))))
    # ---- LN1 bwd (no gamma)
    dx2, g2, sums2 = bk.ln_bwd_res(x, mean, rstd, w, dy, dres)
    out.append(rows("rowbwd(ln).dx", err_stats(dx2, dx_ref)))
    out.append(rows("rowbwd(ln).dW", err_stats(sums2[0], wr.grad)))
    out.append(rows("rowbwd(ln).dB", err_stats(sums2[1], br.grad)))
    # ---- fc2 prep
    gg, sums3 = bk.gamma_prep(dres, y1, gamma)
    out.append(rows("gamma_prep.g(bf16)", err_stats(gg, (gamma * dres).to(torch.bfloat16))))
    out.append(rows("gamma_prep.dGamma", err_stats(sums3[0], (dres * y1.float()).sum(0))))
    out.append(rows("gamma_prep.dBiasBranch", err_stats(sums3[1], (gamma * dres).sum(0))))
    # ---- colsum
    C = 3 * D
    z = torch.randn(M, C, device=dev).to(torch.bfloat16)
    out.append(rows("colsum_bf16", err_stats(bk.colsum_bf16(z), z.float().sum(0))))
    # determinism
    _, _, s_a = bk.ln_bwd_res(x, mean, rstd, w, dy, dres, y=y1, gamma=gamma)
    out.append({"name": "determinism(bitwise)", "pass": bool(torch.equal(s_a, sums))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    allres = {}
    ok = True
    with gpu_lock():
        for D in (1024, 768):
            for M in (7203, 7903, 14406, 15806, 129):
                for spike in (False, True):
                    res = run_case(M, D, spike)
                    key = f"D{D}_M{M}_{'spiky' if spike else 'plain'}"
                    allres[key] = res
                    bad = [r for r in res if not r["pass"]]
                    ok &= not bad
                    worst = min((r.get("cos", 1.0) for r in res), default=1.0)
                    print(f"{key}: {'PASS' if not bad else 'FAIL'}  min cos={worst:.9f}", flush=True)
                    for r in bad:
                        print("   FAIL", r, flush=True)
        # one detailed print
        for r in allres["D1024_M7203_spiky"]:
            print({k: (round(v, 9) if isinstance(v, float) else v) for k, v in r.items()})
    if args.json:
        Path(args.json).write_text(json.dumps(allres, indent=1))
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
