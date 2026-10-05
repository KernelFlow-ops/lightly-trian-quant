"""Round13 fused-epilogue GEMM candidates (fp32 residual stream, erf-GELU) at the real training shapes.

  fc1_gelu   : z = x W^T + b (bf16 pre-activation, kept) and h = GELU_erf(z)
  res        : x_out(fp32) = resid(fp32) + gamma * (x W^T + b), y = x W^T + b (bf16, kept for dGamma)   [proj and fc2]
  fc2_dgelu  : dz = (dy W2) * GELU'_erf(z) (bf16) and db1 = sum_m dz

Candidates: torch (cuBLASLt bias epilogue via F.linear + eager elementwise), r5 lt_epilogue (cuBLASLt BIAS / GELU_AUX_BIAS
epilogues), QuACK stock (gemm_act / gemm_dact / gemm_add: tanh GELU, no gamma), QuACK custom epilogues (round13, erf), Spark hgemm
epilogues (bf16 bias/gelu-tanh/residual, bf16 only).  Error is measured against fp32 with exact erf GELU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TMP = Path("/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120")
os.environ.setdefault("QUACK_CACHE_DIR", str(TMP / "quack_cache"))
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "impl"))
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120"))

import torch
import torch.nn.functional as F

from common.bench import gpu_lock
from mb import err_stats, time_us

SHAPES = {"vitl16": (1024, 7203), "vitb16": (768, 14406)}


def lt_extension():
    from torch.utils.cpp_extension import load

    build = TMP / "build" / "lt_epilogue"
    build.mkdir(parents=True, exist_ok=True)
    src = ROOT / "round13_gemm_fusion_sm120" / "impl" / "third_party" / "r5_lt_epilogue" / "lt_epilogue.cu"
    return load(name="trainopt_r13_lt_epilogue", sources=[str(src)], build_directory=str(build),
                extra_cuda_cflags=["-O3"], extra_ldflags=["-lcublasLt"], verbose=False)


def run(label, rows, fn, ref_fn=None, extra=None, flops=None):
    try:
        out = fn()
        torch.cuda.synchronize()
        err = ref_fn(out) if ref_fn else None
        tm = time_us(fn, iters=30, repeats=3)
        row = {"cand": label, "us": tm["us"], "mode": tm["mode"], "err": err}
        if flops:
            row["tflops"] = flops / tm["us"] / 1e6
        rows.append(row)
        e = f" cos={err['cos']:.6f} relmax={err['max_rel_to_absmax']:.1e}" if err else ""
        print(f"   {label:46s} {tm['us']:8.1f} us{(' %5.0f TF' % row['tflops']) if flops else ''}{e}", flush=True)
    except Exception as exc:
        rows.append({"cand": label, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        print(f"   {label:46s} ERROR {type(exc).__name__}: {str(exc)[:160]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vitl16", "vitb16"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from round13_gemm_fusion_sm120.impl import gemm_ops
    from round5_block_fusion.impl.gelu import _dgelu, gelu as r5_gelu
    import triton

    lt = lt_extension()
    results = {}
    torch.manual_seed(0)
    with gpu_lock():
        for model in args.models:
            d, M = SHAPES[model]
            dev = "cuda"
            gemm_ops.POLICY.update({k: "quack" for k in gemm_ops.POLICY})
            q = gemm_ops._quack()
            from quack.gemm_interface import gemm_act, gemm_add, gemm_dact
            import spark_hgemm_ext
            sp = spark_hgemm_ext.extension()
            ws = torch.empty(32 * 1024 * 1024, device=dev, dtype=torch.uint8)
            results[model] = {}

            # ------------------------------------------------------------ fc1: +bias, GELU(erf)
            N, K = 4 * d, d
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
            b = torch.randn(N, device=dev) * 0.5
            bb = b.to(torch.bfloat16)
            zref = x.float() @ w.float().t() + b
            href = F.gelu(zref)
            flops = 2.0 * M * N * K
            print(f"== {model} fc1 +bias+GELU  M={M} N={N} K={K}", flush=True)
            rows = results[model]["fc1_gelu"] = []
            run("torch F.linear + F.gelu (2 kernels)", rows, lambda: F.gelu(F.linear(x, w, bb)), lambda o: err_stats(o, href), flops=flops)
            plan_b = lt.LinearPlan(M, N, K, True, False)
            plan_t = lt.LinearPlan(M, N, K, True, True)
            zb = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            hb = torch.empty_like(zb)
            aux = torch.empty_like(zb)
            run("r5 lt_epilogue erf (Lt BIAS + Triton GELU)", rows, lambda: (plan_b.run(x, w, bb, zb, zb, ws), r5_gelu(zb))[1], lambda o: err_stats(o, href), flops=flops)
            run("r5 lt_epilogue tanh (Lt GELU_AUX_BIAS, tanh approx)", rows, lambda: (plan_t.run(x, w, bb, hb, aux, ws), hb)[1], lambda o: err_stats(o, href), flops=flops)
            pre = torch.empty_like(zb)
            post = torch.empty_like(zb)
            run("QuACK gemm_act gelu_tanh_approx (stock)", rows, lambda: gemm_act(x, w.t(), bias=b, activation="gelu_tanh_approx", preact_out=pre, postact_out=post)[1],
                lambda o: err_stats(o, href), flops=flops)
            wt = w.t().contiguous()
            run("Spark hgemm v6 bias+gelu(tanh) (h only)", rows, lambda: sp.hgemm(x, wt, hb, 6, bb, 2, None) or hb, lambda o: err_stats(o, href), flops=flops)
            run("round13 QuACK custom epilogue (erf, z+h)", rows, lambda: gemm_ops.linear_gelu(x, w, b)[1], lambda o: err_stats(o, href), flops=flops)
            run("round13 QuACK plain GEMM (reference, no epilogue)", rows, lambda: q["gemm"](x, w.t(), out=zb), None, flops=flops)

            # ------------------------------------------------------------ proj / fc2: +bias, gamma, fp32 residual
            for lname, (Nn, Kk) in (("proj", (d, d)), ("fc2", (d, 4 * d))):
                xin = torch.randn(M, Kk, device=dev, dtype=torch.bfloat16)
                wv = torch.randn(Nn, Kk, device=dev, dtype=torch.bfloat16) * 0.05
                bv = torch.randn(Nn, device=dev) * 0.5
                gv = torch.rand(Nn, device=dev) * 0.5
                resid = torch.randn(M, Nn, device=dev) * 3
                yref = xin.float() @ wv.float().t() + bv
                oref = resid + gv * yref
                flops = 2.0 * M * Nn * Kk
                print(f"== {model} {lname} +bias+gamma+fp32 residual  M={M} N={Nn} K={Kk}", flush=True)
                rows = results[model][f"res_{lname}"] = []
                bvb = bv.to(torch.bfloat16)
                run("torch F.linear + mul + add (3 kernels, autocast sem.)", rows, lambda: resid + gv * F.linear(xin, wv, bvb), lambda o: err_stats(o, oref), flops=flops)
                out32 = torch.empty(M, Nn, device=dev, dtype=torch.float32)
                run("QuACK gemm_add stock (C+bias, no gamma: lower bound)", rows, lambda: gemm_add(xin, wv.t(), resid, out=out32, bias=bv), None, flops=flops)
                run("round13 QuACK custom epilogue (bias+gamma+residual, y)", rows, lambda: gemm_ops.linear_res(xin, wv, bv, gv, resid, lname)[0], lambda o: err_stats(o, oref), flops=flops)
                run("round13 QuACK plain GEMM (reference)", rows, lambda: q["gemm"](xin, wv.t(), out=torch.empty(M, Nn, device=dev, dtype=torch.bfloat16)), None, flops=flops)
                wvt = wv.t().contiguous()
                rb = resid.to(torch.bfloat16)
                outb = torch.empty(M, Nn, device=dev, dtype=torch.bfloat16)
                run("Spark hgemm v6 bias+residual (bf16 stream; info only)", rows, lambda: sp.hgemm(xin, wvt, outb, 6, bvb, 0, rb) or outb, None, flops=flops)

            # ------------------------------------------------------------ fc2 dgrad + GELU'
            N2, K2 = d, 4 * d  # fc2 weight [d, 4d]
            dy = torch.randn(M, N2, device=dev, dtype=torch.bfloat16)
            w2 = torch.randn(N2, K2, device=dev, dtype=torch.bfloat16) * 0.05
            z = (torch.randn(M, K2, device=dev) * 2).to(torch.bfloat16)
            zf = z.float().requires_grad_()
            dh = dy.float() @ w2.float()
            dzref = torch.autograd.grad(F.gelu(zf), zf, dh)[0]
            dbref = dzref.sum(0)
            flops = 2.0 * M * N2 * K2
            print(f"== {model} fc2 dgrad + GELU'  M={M} N={K2} K={N2}", flush=True)
            rows = results[model]["fc2_dgelu"] = []

            def torch_dgelu():
                dh_ = torch.mm(dy, w2)
                dz_ = torch.ops.aten.gelu_backward(dh_, z)
                return dz_, dz_.float().sum(0)
            run("torch mm + gelu_backward + sum (3 kernels)", rows, lambda: torch_dgelu()[0], lambda o: err_stats(o, dzref), flops=flops)
            dz_b = torch.empty(M, K2, device=dev, dtype=torch.bfloat16)
            dh_b = torch.empty_like(dz_b)

            def r5_dgelu():
                torch.mm(dy, w2, out=dh_b)
                _dgelu[(triton.cdiv(dh_b.numel(), 512),)](z, dh_b, dz_b, dh_b.numel(), 512, enable_fp_fusion=False)
                return dz_b
            run("r5 mm + Triton dgelu (2 kernels, no bias grad)", rows, r5_dgelu, lambda o: err_stats(o, dzref), flops=flops)
            run("QuACK gemm_dact gelu_tanh_approx (stock, +postact)", rows, lambda: gemm_dact(dy, w2, z, activation="gelu_tanh_approx")[0], lambda o: err_stats(o, dzref), flops=flops)
            run("round13 QuACK custom epilogue (erf, + db1)", rows, lambda: gemm_ops.dgrad_dgelu(dy, w2, z)[0], lambda o: err_stats(o, dzref), flops=flops)
            _, dbq = gemm_ops.dgrad_dgelu(dy, w2, z)
            print("   db1 error vs fp32:", err_stats(dbq, dbref), flush=True)
            run("round13 QuACK plain GEMM (reference)", rows, lambda: q["gemm"](dy, w2, out=dz_b), None, flops=flops)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
