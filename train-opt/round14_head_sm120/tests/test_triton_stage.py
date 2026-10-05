"""Unit test + timing of the fused stage kernels (GELU -> dwconv3x3 -> LN2D) against an fp32 torch reference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock  # noqa: E402
from round11_bf16_mixed.tests.microbench import compare  # noqa: E402
from round14_head_sm120.impl import head_cuda, head_triton  # noqa: E402


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def ref_stage(Z, bias, dw, gam, bet, B, h, w, eps, dy):
    """fp32 reference with autograd. Z [B*h*w,4C] (bf16 values, upcast), returns y, dZ, dbias, ddw, dgam, dbet."""
    C = bias.numel()
    Zf = Z.float().detach().requires_grad_(True)
    bias_ = bias.detach().clone().requires_grad_(True)
    dw_ = dw.detach().clone().requires_grad_(True)  # [C,1,3,3]
    gam_ = gam.detach().clone().requires_grad_(True)
    bet_ = bet.detach().clone().requires_grad_(True)
    t = Zf.view(B, h, w, 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, 2 * h, 2 * w, C)  # NHWC (pixel shuffle)
    x = (t + bias_).permute(0, 3, 1, 2)
    x = F.gelu(x)
    x = F.conv2d(x, dw_, None, padding=1, groups=C)
    x = x.permute(0, 2, 3, 1)
    y = F.layer_norm(x, (C,), gam_, bet_, eps)
    g = torch.autograd.grad(y, [Zf, bias_, dw_, gam_, bet_], dy.float().view(B, 2 * h, 2 * w, C))
    return y.detach().reshape(-1, C), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--B", type=int, default=7)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--pw", type=int, default=0)
    ap.add_argument("--nw", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--impl", default="triton", choices=["triton", "cuda"])
    ap.add_argument("--spiky", action="store_true")
    args = ap.parse_args()
    C, B, h = args.C, args.B, args.h
    ht = head_cuda if args.impl == 'cuda' else head_triton
    w = h
    eps = 1e-5
    torch.manual_seed(0)
    with gpu_lock():
        scale = 8.0 if args.spiky else 1.0
        Z = (torch.randn(B * h * w, 4 * C, device="cuda") * 2 * scale).bfloat16()
        bias = torch.randn(C, device="cuda") * 0.1
        dw = torch.randn(C, 1, 3, 3, device="cuda") * 0.3
        gam = 1 + 0.2 * torch.randn(C, device="cuda")
        bet = 0.1 * torch.randn(C, device="cuda")
        dy = torch.randn(B * 2 * h * 2 * w, C, device="cuda").bfloat16()
        dwt = dw.reshape(C, 9).t().contiguous()
        PW = args.pw or None
        nw = args.nw or None  # triton: num_warps ; cuda: channels per thread
        Y, mean, rstd = ht.fwd(Z, bias, dwt, gam, bet, B, h, w, eps, PW, nw)
        du, dgam, dbet = ht.bwd1(Z, bias, dwt, gam, mean, rstd, dy, B, h, w, PW, nw)
        dZ, ddw, dbias = ht.bwd2(Z, bias, dwt, du, B, h, w, PW, nw)
        yr, (dZr, dbr, ddwr, dgr, dbtr) = ref_stage(Z, bias, dw, gam, bet, B, h, w, eps, dy)
        print(f"C={C} B={B} h={h} spiky={args.spiky}")
        print(f"  y     rel_l2={rel(Y, yr):.3e} cos={cos(Y, yr):.7f}")
        print(f"  dZ    rel_l2={rel(dZ, dZr):.3e} cos={cos(dZ, dZr):.7f}")
        print(f"  dbias rel_l2={rel(dbias, dbr):.3e} cos={cos(dbias, dbr):.7f}")
        print(f"  ddw   rel_l2={rel(ddw.t().reshape(C,1,3,3), ddwr):.3e} cos={cos(ddw.t(), ddwr.reshape(C,9)):.7f}")
        print(f"  dgam  rel_l2={rel(dgam, dgr):.3e}  dbet rel_l2={rel(dbet, dbtr):.3e}")
        if args.bench:
            cands = {}
            if args.impl == "cuda":
                for cpt_ in (4, 8):
                    for pw_ in (2 * w, w):
                        cands[f"fwd cpt={cpt_} pw={pw_}"] = (lambda cpt_=cpt_, pw_=pw_: ht.fwd(Z, bias, dwt, gam, bet, B, h, w, eps, pw_, cpt_))
                for cpt_ in (4, 8):
                    for rpb_ in (1, 4):
                        for minb_ in (1, 2):
                            k = f"cpt={cpt_} rpb={rpb_} minb={minb_}"
                            cands[f"bwd1 {k}"] = (lambda cpt_=cpt_, rpb_=rpb_, minb_=minb_: ht.bwd1(Z, bias, dwt, gam, mean, rstd, dy, B, h, w, None, cpt_, rpb_, minb_))
                            cands[f"bwd2 {k}"] = (lambda cpt_=cpt_, rpb_=rpb_, minb_=minb_: ht.bwd2(Z, bias, dwt, du, B, h, w, None, cpt_, rpb_, minb_))
            else:
                for pw_ in (None, w):
                    k = f"pw={pw_ or 2*w},nw=8"
                    cands[f"fwd {k}"] = (lambda pw_=pw_: ht.fwd(Z, bias, dwt, gam, bet, B, h, w, eps, pw_, 8))
                    cands[f"bwd1 {k}"] = (lambda pw_=pw_: ht.bwd1(Z, bias, dwt, gam, mean, rstd, dy, B, h, w, pw_, 8))
                    cands[f"bwd2 {k}"] = (lambda pw_=pw_: ht.bwd2(Z, bias, dwt, du, B, h, w, pw_, 8))
            res = compare(cands, iters=30)
            nbytes = B * (2 * h) * (2 * w) * C * 2
            print(f"  one pass over the feature map = {nbytes/1e6:.0f} MB -> {nbytes/1.79e12*1e3:.3f} ms at 1.79 TB/s")
            for k, v in res.items():
                print(f"  {k:24s}", f"{v['ms']:.3f} ms" if "ms" in v else v["error"])


if __name__ == "__main__":
    main()
