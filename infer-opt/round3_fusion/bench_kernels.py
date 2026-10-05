"""Microbench K1-K4 on real ViT-L / ViT-B shapes vs the torch (eager) op sequence.

usage: python bench_kernels.py [k1 k2 k3 k4] [--sweep] [--model vitl16]
Timing goes through common/bench.time_fn (GPU lock).  Results -> results.json
"""
import sys

sys.path.insert(0, "/root/lightly-train/infer-opt/common")
sys.path.insert(0, "/root/lightly-train/infer-opt/round3_fusion")
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from bench import time_fn, write_results
from roofline import waves

from impl import ops

PEAKS = json.loads(Path("/root/lightly-train/infer-opt/results/peaks.json").read_text())
args = [a for a in sys.argv[1:] if not a.startswith("--")]
sweep = "--sweep" in sys.argv
models = ["vitl16"] if "--vitl-only" in sys.argv else ["vitl16", "vitb16"]
only = args or ["k1", "k4", "k3", "k2"]
dev, bf = "cuda", torch.bfloat16
torch.manual_seed(0)
RES = Path(__file__).parent / "results.json"
DIMS = {"vitl16": (1024, 4096, 16), "vitb16": (768, 3072, 12)}
# torch GEMM peaks measured on the model shapes (M=1029, vitl): see brief; pick the closest
TORCH_PEAK_TFLOPS = {"qkv": 126.0, "proj": 131.0, "fc1": 168.0, "fc2": 168.0}
MAX_TFLOPS = 232.0
DRAM_GBPS = 1530.0


def rand(*s, scale=1.0):
    return (torch.randn(*s, device=dev) * scale).to(bf)


def kernel_time_us(fn, n=30):
    """pure GPU busy time per call (sum of CUPTI kernel durations, no launch gaps / event overhead), + kernel count."""
    from bench import gpu_lock
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with gpu_lock(), torch.inference_mode():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type == DeviceType.CUDA and "emcpy" not in e.name and "emset" not in e.name]
    return sum(e.device_time for e in ev) / n, len(ev) / n


def t(fn, **kw):
    r = time_fn(fn, warmup=30, iters=100, repeats=3, **kw)
    ku, nk = kernel_time_us(fn)
    return {"gpu_us": r["gpu_ms_p50"] * 1e3, "wall_us": r["wall_ms_p50"] * 1e3, "kernel_us": ku, "kernels": nk}


def rope_eager(q, k, sin, cos):
    # identical math to SelfAttention.apply_rope
    from lightly_train._models.dinov3.dinov3_src.layers.attention import rope_apply

    N = q.shape[-2]
    prefix = N - sin.shape[-2]
    out = []
    for x in (q, k):
        xf = x.float()
        pre = xf[:, :, :prefix]
        xr = rope_apply(xf[:, :, prefix:], sin, cos)
        out.append(torch.cat((pre, xr), dim=-2).to(x.dtype))
    return out


def report(key, ours, ref, flops=None, peak=None, extra=None):
    d = {"ours": ours, "torch": ref, "speedup_gpu": ref["gpu_us"] / ours["gpu_us"],
         "speedup_kernel_time": ref["kernel_us"] / ours["kernel_us"],
         "speedup_wall": ref["wall_us"] / ours["wall_us"]}
    if flops:
        d["ours_tflops"] = flops / ours["gpu_us"] / 1e6
        d["torch_tflops"] = flops / ref["gpu_us"] / 1e6
        d["ours_kernel_tflops"] = flops / ours["kernel_us"] / 1e6
        d["ours_kernel_pct_of_max232"] = 100 * d["ours_kernel_tflops"] / MAX_TFLOPS
        d["ours_pct_of_max232"] = 100 * d["ours_tflops"] / MAX_TFLOPS
        if peak:
            d["ours_pct_of_torch_gemm_peak"] = 100 * d["ours_tflops"] / peak
    if extra:
        d.update(extra)
    print(f"{key:34s} ours {ours['gpu_us']:8.1f}us (wall {ours['wall_us']:8.1f}) | torch {ref['gpu_us']:8.1f}us "
          f"(wall {ref['wall_us']:8.1f}) | x{d['speedup_gpu']:.2f}"
          + f" | kernel-time ours {ours['kernel_us']:.1f}us ({ours['kernels']:.0f} k) torch {ref['kernel_us']:.1f}us ({ref['kernels']:.0f} k)"
          + (f" | {d['ours_tflops']:.0f} TFLOPS {d['ours_pct_of_max232']:.0f}% (kernel-time: {d['ours_kernel_tflops']:.0f}, {d['ours_kernel_pct_of_max232']:.0f}%)" if flops else ""), flush=True)
    write_results(RES, {key: d})


CFGS = [dict(bm=128, bn=128, bk=32, stages=3, threads=128),
        dict(bm=128, bn=128, bk=64, stages=2, threads=128),
        dict(bm=128, bn=128, bk=32, stages=4, threads=256),
        dict(bm=128, bn=64, bk=32, stages=3, threads=128),
        dict(bm=64, bn=128, bk=32, stages=3, threads=128),
        dict(bm=64, bn=64, bk=64, stages=3, threads=128),
        dict(bm=128, bn=64, bk=64, stages=3, threads=128),
        dict(bm=128, bn=256, bk=32, stages=3, threads=256),
        dict(bm=256, bn=128, bk=32, stages=3, threads=256)]


def best_cfg(prefix, M, N, K, mk_fn, cfgs=CFGS):
    """sweep tile configs, install the best in ops.TILE_CFG; returns (cfg, table)."""
    table = []
    for c in cfgs:
        try:
            fn = mk_fn(c)
            fn()
            r = t(fn)["gpu_us"]
        except Exception as e:  # noqa: BLE001
            table.append({"cfg": c, "err": str(e)[:80]})
            continue
        w = waves(M, N, c["bm"], c["bn"])
        table.append({"cfg": c, "gpu_us": r, "tiles": w["tiles"], "waves": w["waves"]})
        print("   sweep", c, f"{r:.1f}us tiles={w['tiles']} waves={w['waves']:.2f}", flush=True)
    ok = [x for x in table if "gpu_us" in x]
    best = min(ok, key=lambda x: x["gpu_us"])
    return best["cfg"], table


for name in models:
    d, ff, H = DIMS[name]
    for (B, N, prefix) in ((2, 1029, 5), (2, 1129, 105)):
        M = B * N
        tag = f"{name}_N{N}"
        if "k1" in only:
            x, r = rand(B, N, d), rand(B, N, d)
            wb, bb = rand(d), rand(d)
            w, b = wb.float(), bb.float()
            ours = t(lambda: ops.add_layernorm(x, r, w, b))
            ref = t(lambda: F.layer_norm(x + r, (d,), wb, bb, 1e-5))
            ours0 = t(lambda: ops.add_layernorm(x, None, w, b))
            ref0 = t(lambda: F.layer_norm(x, (d,), wb, bb, 1e-5))
            xf, rf = x.float(), r.float()  # fp32 residual stream (autocast-like): eager = add + fp32 LN
            ours32 = t(lambda: ops.add_layernorm(xf, r, w, b))
            ref32 = t(lambda: F.layer_norm(xf + r, (d,), w, b, 1e-5))
            report(f"k1_add_ln_fp32res_{tag}", ours32, ref32)
            byt = 4 * M * d * 2  # read x, r; write s, y
            report(f"k1_add_ln_{tag}", ours, ref, extra={"GBps_ours": byt / ours["gpu_us"] / 1e3,
                                                          "pct_dram_bw": 100 * byt / ours["gpu_us"] / 1e3 / DRAM_GBPS})
            report(f"k1_ln_only_{tag}", ours0, ref0)
        if "k4" in only:
            h = rand(M, d); W = rand(ff, d, scale=d ** -0.5); b = rand(ff)
            fl = 2 * M * d * ff
            key = f"gelu_{ff}_{d}"
            if sweep:
                cfg, table = best_cfg("gelu", M, ff, d, lambda c: (lambda: ops.gemm_bias_gelu(h, W, b, cfg=c)))
                ops.TILE_CFG[key] = cfg
                write_results(RES, {f"sweep_k4_{tag}": table})
            ours = t(lambda: ops.gemm_bias_gelu(h, W, b))
            ref = t(lambda: F.gelu(F.linear(h, W, b)))
            gemm_only = t(lambda: F.linear(h, W, b))
            report(f"k4_gemm_gelu_{tag}", ours, ref, fl, TORCH_PEAK_TFLOPS["fc1"],
                   {"torch_linear_only_us": gemm_only["gpu_us"], "torch_linear_only_kernel_us": gemm_only["kernel_us"], "cfg": ops.TILE_CFG.get(key, "default"),
                    "waves": waves(M, ff, **{k: ops.TILE_CFG.get(key, {}).get(k, 128) for k in ("bm", "bn")})})
        if "k3" in only:
            for kname, K_, Nn, pk in (("fc2", ff, d, "fc2"), ("proj", d, d, "proj")):
                a = rand(M, K_); W = rand(Nn, K_, scale=K_ ** -0.5); b = rand(Nn); res = rand(M, Nn)
                g = rand(Nn)
                g32 = g.float()
                fl = 2 * M * K_ * Nn
                key = f"res_{Nn}_{K_}"
                if sweep:
                    cfg, table = best_cfg("res", M, Nn, K_, lambda c: (lambda: ops.gemm_bias_residual(a, W, b, res, g32, cfg=c)))
                    ops.TILE_CFG[key] = cfg
                    write_results(RES, {f"sweep_k3_{kname}_{tag}": table})
                ours = t(lambda: ops.gemm_bias_residual(a, W, b, res, g32))
                ref = t(lambda: res + g * F.linear(a, W, b))  # eager LayerScale form: linear, mul, add
                ref_nols = t(lambda: res + F.linear(a, W, b))
                gemm_only = t(lambda: F.linear(a, W, b))
                report(f"k3_{kname}_{tag}", ours, ref, fl, TORCH_PEAK_TFLOPS[pk],
                       {"torch_no_layerscale_us": ref_nols["gpu_us"], "torch_linear_only_us": gemm_only["gpu_us"], "torch_linear_only_kernel_us": gemm_only["kernel_us"],
                        "cfg": ops.TILE_CFG.get(key, "default"),
                        "waves": waves(M, Nn, **{k: ops.TILE_CFG.get(key, {}).get(k, 128) for k in ("bm", "bn")})})
        if "k2" in only:
            h = rand(B, N, d); Wq = rand(3 * d, d, scale=d ** -0.5); bq = rand(3 * d)
            ang = torch.randn(1024, 32, device=dev)
            sin = torch.sin(ang).repeat(1, 2).float().contiguous(); cos = torch.cos(ang).repeat(1, 2).float().contiguous()
            fl = 2 * M * d * 3 * d
            key = f"qkv_{d}"
            if sweep:
                cs = [c for c in CFGS if c["bn"] in (64, 128) and c["bm"] <= 128 and d % c["bn"] == 0]
                cfg, table = best_cfg("qkv", M, 3 * d, d, lambda c: (lambda: ops.qkv_rope(h, Wq, bq, sin, cos, prefix, H, cfg=c)), cs)
                ops.TILE_CFG[key] = cfg
                write_results(RES, {f"sweep_k2_{tag}": table})

            def eager():
                qkv = F.linear(h, Wq, bq).reshape(B, N, 3, H, 64)
                q, k, v = [z.transpose(1, 2) for z in qkv.unbind(2)]
                q, k = rope_eager(q, k, sin, cos)
                return q.contiguous(), k.contiguous(), v.contiguous()  # attention-ready [B,H,N,64] layout

            def eager_noc():
                qkv = F.linear(h, Wq, bq).reshape(B, N, 3, H, 64)
                q, k, v = [z.transpose(1, 2) for z in qkv.unbind(2)]
                return rope_eager(q, k, sin, cos)

            ours = t(lambda: ops.qkv_rope(h, Wq, bq, sin, cos, prefix, H))
            ref = t(eager_noc)
            gemm_only = t(lambda: F.linear(h, Wq, bq))
            report(f"k2_qkv_rope_{tag}", ours, ref, fl, TORCH_PEAK_TFLOPS["qkv"],
                   {"torch_linear_only_us": gemm_only["gpu_us"], "torch_linear_only_kernel_us": gemm_only["kernel_us"], "cfg": ops.TILE_CFG.get(key, "default"),
                    "waves": waves(M, 3 * d, **{k: ops.TILE_CFG.get(key, {}).get(k, 128) for k in ("bm", "bn")})})

if sweep:
    write_results(RES, {"tile_cfg_selected": {k: v for k, v in ops.TILE_CFG.items()}})
    print(json.dumps(ops.TILE_CFG, indent=1))
