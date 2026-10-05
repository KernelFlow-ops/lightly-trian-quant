"""Candidate matrix for the elementwise part of one ScaleBlock stage (bias add + GELU + depthwise 3x3 + LN2D), fwd and fwd+bwd,
at the real stage-1 / stage-2 shapes of both models.

Candidates: torch bf16 (channels_last conv, bf16-in/out LayerNorm = the TB1 head), Triton v1 (plain), Triton v2 (prefetch + fused reductions),
CUDA (head_kernels.cu).  The torch candidate gets the already pixel-shuffled NHWC tensor (cuDNN ConvT writes it directly); the fused
candidates read the un-shuffled GEMM output.  Backward of the fused path = bwd1 + bwd2 (+ the partial sums).
"""
from __future__ import annotations

import argparse
import json
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

MODELS = {"vitl16": (7, 1024), "vitb16": (14, 768)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {}
    with gpu_lock():
        for model, (B, C) in MODELS.items():
            for stage, h in ((1, 32), (2, 64)):
                H = 2 * h
                gen = torch.Generator(device="cuda").manual_seed(0)
                Z = (torch.randn(B * h * h, 4 * C, device="cuda", generator=gen) * 2).bfloat16()
                t = Z.view(B, h, h, 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, H, C).contiguous()  # NHWC, pixel-shuffled
                bias = torch.randn(C, device="cuda", generator=gen) * 0.1
                dw = torch.randn(C, 1, 3, 3, device="cuda", generator=gen) * 0.3
                gam = 1 + 0.2 * torch.randn(C, device="cuda", generator=gen)
                bet = 0.1 * torch.randn(C, device="cuda", generator=gen)
                dwt = dw.reshape(C, 9).t().contiguous()
                dy = torch.randn(B * H * H, C, device="cuda", generator=gen).bfloat16()
                dy_nchw = dy.view(B, H, H, C).permute(0, 3, 1, 2)
                tt = t.permute(0, 3, 1, 2).requires_grad_(True)  # channels_last NCHW view

                def torch_fwd():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        x = F.gelu(tt + bias.to(torch.bfloat16).view(1, C, 1, 1))
                        x = F.conv2d(x, dw.to(torch.bfloat16), None, padding=1, groups=C)
                        return F.layer_norm(x.permute(0, 2, 3, 1), (C,), gam.to(torch.bfloat16), bet.to(torch.bfloat16), 1e-5)

                bias_r = bias.clone().requires_grad_(True)
                dw_r = dw.clone().requires_grad_(True)
                gam_r = gam.clone().requires_grad_(True)
                bet_r = bet.clone().requires_grad_(True)

                def torch_fwdbwd():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x = F.gelu(tt + bias_r.to(torch.bfloat16).view(1, C, 1, 1))
                        x = F.conv2d(x, dw_r.to(torch.bfloat16), None, padding=1, groups=C)
                        y = F.layer_norm(x.permute(0, 2, 3, 1), (C,), gam_r.to(torch.bfloat16), bet_r.to(torch.bfloat16), 1e-5)
                    torch.autograd.grad(y, [tt, bias_r, dw_r, gam_r, bet_r], dy.view(B, H, H, C))

                cands = {"torch bf16 (TB1)": (torch_fwd, torch_fwdbwd)}
                for name, mod, kw in (("triton v1", head_triton, {"V": 1}), ("triton v2", head_triton, {"V": 2}), ("cuda", head_cuda, {})):
                    def mk(mod=mod, kw=kw):
                        def fwd():
                            if "V" in kw:
                                head_triton.V = kw["V"]
                            return mod.fwd(Z, bias, dwt, gam, bet, B, h, h, 1e-5)

                        def fwdbwd():
                            if "V" in kw:
                                head_triton.V = kw["V"]
                            y, mean, rstd = mod.fwd(Z, bias, dwt, gam, bet, B, h, h, 1e-5)
                            du, _, _ = mod.bwd1(Z, bias, dwt, gam, mean, rstd, dy, B, h, h)
                            mod.bwd2(Z, bias, dwt, du, B, h, h)
                        return fwd, fwdbwd
                    cands[name] = mk()
                res_f = compare({k: v[0] for k, v in cands.items()}, iters=30)
                res_fb = compare({k: v[1] for k, v in cands.items()}, iters=30)
                key = f"{model}/stage{stage}"
                nbytes = B * H * H * C * 2
                report[key] = {"feature_map_MB": nbytes / 1e6, "fwd_ms": {k: v.get("ms") for k, v in res_f.items()}, "fwdbwd_ms": {k: v.get("ms") for k, v in res_fb.items()}}
                print(f"== {key}: B={B} C={C} out {H}x{H}, one pass = {nbytes/1e6:.0f} MB ({nbytes/1.79e12*1e3:.3f} ms @1.79TB/s)")
                for k in cands:
                    print(f"   {k:20s} fwd {res_f[k].get('ms', float('nan')):7.3f} ms   fwd+bwd {res_fb[k].get('ms', float('nan')):7.3f} ms")
                head_triton.V = 2
                del Z, t, tt, dy
                torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
