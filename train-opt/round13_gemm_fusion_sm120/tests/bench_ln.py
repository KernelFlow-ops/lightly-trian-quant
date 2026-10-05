"""Round13 LayerNorm candidate matrix (fp32 residual stream x in, bf16 out), real training shapes.

fwd   : fp32 x[M,D] -> bf16 y (+stats)
bwd   : LN backward + residual-gradient add + (optional) gamma prep  (the per-block elementwise backward)
Candidates: torch (autograd composite), r5 Triton fused_norm, QuACK layernorm_{fwd,bwd}, round13 CUDA kernels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TMP = Path("/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120")
os.environ.setdefault("QUACK_CACHE_DIR", str(TMP / "quack_cache"))
ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "impl"))

import torch
import torch.nn.functional as F

import block_kernels as bk
from common.bench import gpu_lock
from mb import err_stats, time_us

EPS = 1e-6
SHAPES = [("vitl16", 1024, 7203), ("vitl16", 1024, 7903), ("vitb16", 768, 14406), ("vitb16", 768, 15806)]


def traffic_bytes(kind, M, D):
    if kind == "fwd":  # read x fp32, write y bf16 (+ small stats)
        return M * D * (4 + 2)
    if kind == "bwd_ln":  # read x fp32, dy bf16, dres fp32; write dx fp32
        return M * D * (4 + 2 + 4 + 4)
    if kind == "bwd_ln_gam":  # + read y1 bf16, write g bf16
        return M * D * (4 + 2 + 4 + 2 + 4 + 2)
    if kind == "bwd_gam":  # read dres fp32, y bf16; write g bf16
        return M * D * (4 + 2 + 2)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import triton
    from round5_block_fusion.impl import fused_norm as fn_mod
    from round5_block_fusion.impl.fused_norm import _ResidualNorm
    from quack.rmsnorm import _rmsnorm_fwd, layernorm_bwd
    rows = []
    with gpu_lock():
        for model, D, M in SHAPES:
            dev = "cuda"
            torch.manual_seed(0)
            x = torch.randn(M, D, device=dev) * 2
            x[:, 7] = 1.5e5
            w = torch.rand(D, device=dev) + 0.5
            b = torch.randn(D, device=dev) * 0.1
            dy = (torch.randn(M, D, device=dev) * 0.1).to(torch.bfloat16)
            dres = torch.randn(M, D, device=dev) * 0.05
            gamma = torch.rand(D, device=dev) * 1e-2
            y1 = torch.randn(M, D, device=dev).to(torch.bfloat16)
            yref = F.layer_norm(x, (D,), w, b, EPS)

            # ---------------- forward
            ymine, mean, rstd = bk.ln_fwd(x, w, b, EPS)
            cands_fwd = {}
            cands_fwd["torch(layer_norm+cast)"] = lambda: F.layer_norm(x, (D,), w, b, EPS).to(torch.bfloat16)

            def r5_fwd():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return _ResidualNorm.apply(x, None, None, w, b, EPS)[1]
            cands_fwd["r5_triton"] = r5_fwd
            qo = torch.empty(M, D, device=dev, dtype=torch.bfloat16)
            qr = torch.empty(M, device=dev)
            qm = torch.empty(M, device=dev)
            cands_fwd["quack_layernorm_fwd(bf16 out)"] = lambda: _rmsnorm_fwd(x, w, qo, b, qr, qm, None, None, EPS, True)
            cands_fwd["round13_cuda"] = lambda: bk.ln_fwd(x, w, b, EPS)
            for name, fn in cands_fwd.items():
                try:
                    o = fn()
                    o = o[0] if isinstance(o, tuple) else o
                    if name.startswith("quack"):
                        o = qo
                    e = err_stats(o, yref)
                    tm = time_us(fn)
                    gbs = traffic_bytes("fwd", M, D) / tm["us"] / 1e3
                    rows.append({"model": model, "D": D, "M": M, "kind": "fwd", "cand": name, "us": tm["us"], "GBps": gbs, "err": e})
                    print(f"{model} M={M} fwd  {name:34s} {tm['us']:8.1f} us {gbs:7.0f} GB/s cos={e['cos']:.7f} [{tm['mode']}]", flush=True)
                except Exception as exc:
                    print(f"{model} M={M} fwd  {name} ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                    rows.append({"model": model, "D": D, "M": M, "kind": "fwd", "cand": name, "error": str(exc)[:200]})

            # ---------------- backward (LN bwd + residual add [+ gamma prep])
            xr = x.clone().requires_grad_()
            yr = F.layer_norm(xr, (D,), w, b, EPS)
            dx_ref = torch.autograd.grad(yr, xr, dy.float())[0] + dres

            def torch_bwd(with_gam):
                def fn():
                    with torch.enable_grad():
                        xx = x.detach().requires_grad_()
                        ww = w.detach().requires_grad_()
                        bb = b.detach().requires_grad_()
                        yy = F.layer_norm(xx, (D,), ww, bb, EPS)
                        dxx, dw, db = torch.autograd.grad(yy, (xx, ww, bb), dy.float())
                    dxx = dxx + dres
                    if with_gam:
                        g = (gamma * dxx).to(torch.bfloat16)
                        return dxx, g, (dxx * y1.float()).sum(0), g.float().sum(0), dw, db
                    return dxx, dw, db
                return fn

            def r5_bwd(with_gam):
                def fn():
                    dxo = torch.empty_like(x)
                    dbo = torch.empty_like(x) if with_gam else x
                    groups = triton.cdiv(M, 8)
                    part = torch.empty((groups, 3, D), device=x.device, dtype=torch.float32)
                    fn_mod._bwd[(groups,)](x, x, gamma, w, mean, rstd, dres, dy, dxo, dbo, part, M, D,
                                           triton.next_power_of_2(D), 8, with_gam, num_warps=8, enable_fp_fusion=False)
                    return dxo, dbo, part.sum(0)
                return fn

            for with_gam, kind in ((False, "bwd_ln"), (True, "bwd_ln_gam")):
                cands = {"torch(autograd composite)": torch_bwd(with_gam),
                         "r5_triton(bwd; branch grad fp32 [M,D])": r5_bwd(with_gam),
                         "round13_cuda": (lambda wg=with_gam: bk.ln_bwd_res(x, mean, rstd, w, dy, dres, y=y1 if wg else None, gamma=gamma if wg else None))}
                if not with_gam:
                    cands["quack_layernorm_bwd(core only, no dres/gamma)"] = lambda: layernorm_bwd(x, w, dy.float(), rstd, mean, True)
                for name, fn in cands.items():
                    try:
                        o = fn()
                        first = o[0]
                        e = err_stats(first, dx_ref) if (not name.startswith("quack")) else err_stats(first, dx_ref - dres)
                        tm = time_us(fn)
                        gbs = traffic_bytes(kind, M, D) / tm["us"] / 1e3
                        rows.append({"model": model, "D": D, "M": M, "kind": kind, "cand": name, "us": tm["us"], "GBps": gbs, "err": e})
                        print(f"{model} M={M} {kind:10s} {name:44s} {tm['us']:8.1f} us {gbs:7.0f} GB/s cos={e['cos']:.7f} [{tm['mode']}]", flush=True)
                    except Exception as exc:
                        print(f"{model} M={M} {kind} {name} ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                        rows.append({"model": model, "D": D, "M": M, "kind": kind, "cand": name, "error": str(exc)[:200]})
            # fc2 prep alone
            for name, fn in {"torch(mul+cast+sums)": lambda: ((gamma * dres).to(torch.bfloat16), (dres * y1.float()).sum(0), (gamma * dres).sum(0)),
                             "round13_cuda": lambda: bk.gamma_prep(dres, y1, gamma)}.items():
                tm = time_us(fn)
                gbs = traffic_bytes("bwd_gam", M, D) / tm["us"] / 1e3
                rows.append({"model": model, "D": D, "M": M, "kind": "bwd_gam", "cand": name, "us": tm["us"], "GBps": gbs})
                print(f"{model} M={M} bwd_gam    {name:44s} {tm['us']:8.1f} us {gbs:7.0f} GB/s [{tm['mode']}]", flush=True)
            # bf16 column sum (qkv bias grad)
            z = torch.randn(M, 3 * D, device=dev).to(torch.bfloat16)
            for name, fn in {"torch(float().sum(0))": lambda: z.float().sum(0), "torch(sum(0,dtype=fp32))": lambda: z.sum(0, dtype=torch.float32),
                             "round13_cuda": lambda: bk.colsum_bf16(z)}.items():
                tm = time_us(fn)
                gbs = M * 3 * D * 2 / tm["us"] / 1e3
                rows.append({"model": model, "D": D, "M": M, "kind": "colsum_qkv", "cand": name, "us": tm["us"], "GBps": gbs})
                print(f"{model} M={M} colsum_qkv {name:44s} {tm['us']:8.1f} us {gbs:7.0f} GB/s [{tm['mode']}]", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
