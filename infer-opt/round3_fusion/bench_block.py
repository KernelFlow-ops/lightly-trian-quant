"""Whole-layer microbench: fused_block vs the original eager block (real weights), + kernel counts.

Eager variants (exactly the model's block math via task_model._attn):
   eager_bf16 : block weights bf16, no autocast (bf16-pure baseline)
   eager_ac   : fp32 weights + autocast bf16, fp32 residual stream (main baseline)
Fused: fp32 residual stream (default recommended) and bf16 residual stream.
usage: python bench_block.py [vitl16|vitb16] ; --profile <variant> for nsys (profile_nsys.sh)
"""
import sys

sys.path.insert(0, "/root/lightly-train/infer-opt/common")
sys.path.insert(0, "/root/lightly-train/infer-opt/round3_fusion")
import copy

import model_factory as mf
import torch
from bench import profile_fn, time_fn, write_results
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from impl import ops

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
name = argv[0] if argv else "vitl16"
prof_variant = sys.argv[sys.argv.index("--profile") + 1] if "--profile" in sys.argv else None
model = mf.load_model(name, torch.float32)
model_bf = copy.deepcopy(model).to(torch.bfloat16)
bb = model.backbone
H = bb.blocks[0].attn.num_heads
sin, cos = bb.rope_embed(H=32, W=32)
d = bb.blocks[0].norm1.weight.numel()
B = 2
res = {}


def eager(m, blk, x):
    h = blk.norm1(x)
    x = x + blk.ls1(m._attn(blk.attn, h, rope=(sin, cos), mask=None))
    return x + blk.ls2(blk.mlp(blk.norm2(x)))


def count_kernels(fn):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in p.events() if e.device_type == DeviceType.CUDA]
    bad = ("Memcpy", "memcpy", "Memset", "memset")
    return sum(1 for n in names if not any(b in n for b in bad)), names


def kernel_time_us(fn, n=20):
    """sum of CUPTI kernel durations per call (pure GPU busy time, no launch gaps)."""
    from bench import gpu_lock

    with gpu_lock(), torch.inference_mode():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type == DeviceType.CUDA and "emcpy" not in e.name and "emset" not in e.name]
    return sum(e.device_time for e in ev) / n


for tag, blk_i, N in (("normal", 0, 1029), ("joint", len(bb.blocks) - 1, 1129)):
    blk, blk_bf = bb.blocks[blk_i], model_bf.backbone.blocks[blk_i]
    w = ops.prepare_block_weights(blk)
    x32 = (torch.randn(B, N, d, device="cuda") * 2).contiguous()
    xbf = x32.bfloat16().contiguous()
    prefix = N - 1024

    def f_eager_bf16():
        return eager(model_bf, blk_bf, xbf)

    def f_eager_ac():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return eager(model, blk, x32)

    def f_fused32():
        return ops.fused_block(x32, w, sin, cos, prefix, H)

    def f_fusedbf():
        return ops.fused_block(xbf, w, sin, cos, prefix, H)

    variants = {"eager_bf16": f_eager_bf16, "eager_autocast": f_eager_ac,
                "fused_fp32res": f_fused32, "fused_bf16res": f_fusedbf}
    if prof_variant:
        if tag == "normal":
            with torch.inference_mode():
                profile_fn(variants[prof_variant], warmup=20, iters=10)
        continue
    with torch.inference_mode():
        for v, fn in variants.items():
            fn()
        torch.cuda.synchronize()
    for v, fn in variants.items():
        with torch.inference_mode():
            nk, names = count_kernels(fn)
        t = time_fn(fn, warmup=30, iters=100, repeats=3)
        ku = kernel_time_us(fn)
        res[f"{tag}_N{N}_{v}"] = {"kernels": nk, "gpu_us": t["gpu_ms_p50"] * 1e3, "wall_us": t["wall_ms_p50"] * 1e3, "kernel_us": ku}
        print(f"{name} {tag:6s} N={N} {v:15s} kernels {nk:3d}  kernel-time {ku:7.1f}us  gpu {t['gpu_ms_p50']*1e3:8.1f}us  wall {t['wall_ms_p50']*1e3:8.1f}us", flush=True)
        if v == "fused_fp32res":
            print("   kernels:", [n[:60] for n in names])
    e, f = res[f"{tag}_N{N}_eager_autocast"], res[f"{tag}_N{N}_fused_fp32res"]
    eb, fb = res[f"{tag}_N{N}_eager_bf16"], res[f"{tag}_N{N}_fused_bf16res"]
    print(f"   speedup vs autocast eager: gpu x{e['gpu_us']/f['gpu_us']:.2f} wall x{e['wall_us']/f['wall_us']:.2f}; "
          f"bf16 fused vs bf16 eager: gpu x{eb['gpu_us']/fb['gpu_us']:.2f} wall x{eb['wall_us']/fb['wall_us']:.2f}")
if not prof_variant:
    write_results("/root/lightly-train/infer-opt/round3_fusion/results.json", {"block_bench": {name: res}})
