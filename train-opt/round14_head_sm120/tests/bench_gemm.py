"""round14: candidate matrix for the GEMM parts of the head (ConvT k2s2 as GEMM [B*h*w, C] x [C, 4C], and the mask einsum).

Layout: ConvT as nn.Linear(C -> 4C): Wm [4C, C] row-major (Wm[(dy*2+dx)*C+co, ci] = W[ci, co, dy, dx]).
  fwd   Z  [M,4C] = X [M,C]  @ Wm^T
  dgrad dX [M,C]  = dZ [M,4C] @ Wm
  wgrad dWm[4C,C] = dZ^T @ X   (fp32 out)
Einsum (per batch b): logits[b] [100,HW] = E[b] [100,C] @ Y[b]^T ; dY[b] [HW,C] = dL[b]^T @ E[b] ; dE[b] [100,C] = dL[b] @ Y[b]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round14_head_sm120/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock  # noqa: E402
from round11_bf16_mixed.tests.microbench import compare  # noqa: E402

MODELS = {"vitl16": (7, 1024), "vitb16": (14, 768)}
NQ = 100


def tflops(ms, flops):
    return flops / (ms * 1e-3) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    from quack.gemm_interface import gemm as qgemm

    report = {}
    with gpu_lock():
        for model in args.models:
            B, C = MODELS[model]
            rep = {}
            g = torch.Generator(device="cuda").manual_seed(0)
            for stage, hw in ((1, 32), (2, 64)):
                M = B * hw * hw
                N4 = 4 * C
                X = torch.randn(M, C, device="cuda", generator=g).bfloat16()
                dZ = torch.randn(M, N4, device="cuda", generator=g).bfloat16()
                Wm = (torch.randn(N4, C, device="cuda", generator=g) * 0.03).bfloat16()
                Z = torch.empty(M, N4, device="cuda", dtype=torch.bfloat16)
                dX = torch.empty(M, C, device="cuda", dtype=torch.bfloat16)
                dW32 = torch.empty(N4, C, device="cuda", dtype=torch.float32)
                fl = 2.0 * M * C * N4
                # cuDNN ConvTranspose2d (current baseline path), channels_last bf16
                convt = torch.nn.ConvTranspose2d(C, C, 2, 2).cuda().bfloat16()
                xin = X.view(B, hw, hw, C).permute(0, 3, 1, 2).requires_grad_(True)  # NCHW view of NHWC data = channels_last
                gout = dZ.view(B, hw, hw, 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, 2 * hw, 2 * hw, C).permute(0, 3, 1, 2)

                def cudnn_fwd():
                    with torch.no_grad():
                        return convt(xin)

                def cudnn_fwdbwd():
                    out = convt(xin)
                    torch.autograd.grad(out, [xin, convt.weight], gout)

                cands = {
                    "fwd/torch.mm": lambda: torch.mm(X, Wm.t(), out=Z),
                    "fwd/quack": lambda: qgemm(X, Wm.t(), out=Z),
                    "fwd/cudnn_convT": cudnn_fwd,
                    "dgrad/torch.mm": lambda: torch.mm(dZ, Wm, out=dX),
                    "dgrad/quack": lambda: qgemm(dZ, Wm, out=dX),
                    "wgrad/torch.mm_f32": lambda: torch.mm(dZ.t(), X, out_dtype=torch.float32),
                    "wgrad/quack_f32": lambda: qgemm(dZ.t(), X, out=dW32),
                    "wgrad/quack_f32_splitk_auto": lambda: qgemm(dZ.t(), X, out=dW32, split_k=None),
                    "fwd+bwd/cudnn_convT(total)": cudnn_fwdbwd,
                }
                res = compare(cands, iters=args.iters)
                rep[f"stage{stage}"] = {"M": M, "flops_per_gemm": fl, "ms": {k: v.get("ms") for k, v in res.items()}, "err": {k: v["error"] for k, v in res.items() if "error" in v}}
                print(f"== {model} B={B} C={C} ConvT stage{stage}: M={M} K={C} N={N4} ({fl/1e9:.1f} GF/gemm)")
                for k, v in res.items():
                    if "ms" in v:
                        print(f"   {k:34s} {v['ms']:8.3f} ms  {tflops(v['ms'], fl):6.1f} TFLOPS" if "total" not in k else f"   {k:34s} {v['ms']:8.3f} ms (fwd+dgrad+wgrad, 3 GEMMs: {tflops(v['ms'], 3*fl):.1f} TFLOPS)")
                    else:
                        print(f"   {k:34s} FAIL {v['error']}")
                # accuracy of quack vs torch.mm (fp32 compare)
                ref = (X.float() @ Wm.float().t())
                qgemm(X, Wm.t(), out=Z)
                rep[f"stage{stage}"]["fwd_quack_rel_l2"] = float((Z.float() - ref).norm() / ref.norm())
                del X, dZ, Wm, Z, dX, dW32, convt, xin, gout, ref
                torch.cuda.empty_cache()
            # einsum
            HW = 128 * 128
            Y = torch.randn(B, HW, C, device="cuda", generator=g).bfloat16()
            E = torch.randn(B, NQ, C, device="cuda", generator=g).bfloat16()
            dL = torch.randn(B, NQ, HW, device="cuda", generator=g).bfloat16()
            fl_e = 2.0 * B * NQ * C * HW
            L32 = torch.empty(B, NQ, HW, device="cuda", dtype=torch.float32)
            L16 = torch.empty(B, NQ, HW, device="cuda", dtype=torch.bfloat16)
            dY = torch.empty(B, HW, C, device="cuda", dtype=torch.bfloat16)
            dE = torch.empty(B, NQ, C, device="cuda", dtype=torch.bfloat16)
            dE32 = torch.empty(B, NQ, C, device="cuda", dtype=torch.float32)

            def bmm_f32():
                return torch.bmm(E, Y.transpose(1, 2), out_dtype=torch.float32)

            cands = {
                "e_fwd/bmm_bf16_out": lambda: torch.bmm(E, Y.transpose(1, 2), out=L16),
                "e_fwd/bmm_f32_out": bmm_f32,
                "e_fwd/quack_f32": lambda: qgemm(E, Y.transpose(1, 2), out=L32),
                "e_dY/bmm": lambda: torch.bmm(dL.transpose(1, 2), E, out=dY),
                "e_dY/quack": lambda: qgemm(dL.transpose(1, 2), E, out=dY),
                "e_dE/bmm_bf16": lambda: torch.bmm(dL, Y, out=dE),
                "e_dE/bmm_f32": lambda: torch.bmm(dL, Y, out_dtype=torch.float32),
                "e_dE/quack_f32": lambda: qgemm(dL, Y, out=dE32),
                "e_dE/quack_f32_splitk": lambda: qgemm(dL, Y, out=dE32, split_k=None),
                "e_dE/matmul_flat(reshape)": lambda: torch.matmul(dL.reshape(B * NQ, HW), Y.reshape(B, HW, C)[0]),
            }
            res = compare(cands, iters=args.iters)
            rep["einsum"] = {"flops": fl_e, "ms": {k: v.get("ms") for k, v in res.items()}, "err": {k: v["error"] for k, v in res.items() if "error" in v}}
            print(f"== {model} einsum B={B} Q={NQ} C={C} HW={HW} ({fl_e/1e9:.1f} GF/op; y read = {Y.numel()*2/1e6:.0f} MB)")
            for k, v in res.items():
                if "ms" in v:
                    print(f"   {k:34s} {v['ms']:8.3f} ms  {tflops(v['ms'], fl_e):6.1f} TFLOPS  {Y.numel()*2/1e9/(v['ms']*1e-3):.2f} TB/s(Y)")
                else:
                    print(f"   {k:34s} FAIL {v['error']}")
            report[model] = rep
            del Y, E, dL, L32, L16, dY, dE, dE32
            torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
