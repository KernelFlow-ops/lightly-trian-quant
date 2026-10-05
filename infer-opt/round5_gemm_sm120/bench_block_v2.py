"""Whole-layer benchmark (real weights, fp32 residual stream): round3 fused_block (v1) vs round5 v1 (tuned GEMMs) vs round5 v2 (LN folded,
PDL off/on).  Three timings: kernel-time (torch.profiler CUPTI sum = GPU busy time), CUDA-event p50 (bench.time_fn) and a CUDA-Graph replay of
4 consecutive layers with distinct weights (what the Round-9 pipeline does; weights come from DRAM like in the real model).
usage: python bench_block_v2.py [vitl16|vitb16]"""
import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120"); sys.path.insert(0, "/root/lightly-train/infer-opt/round5_gemm_sm120/impl")
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import statistics
import torch
from kt import ab
import model_factory as mf
from bench import gpu_lock, time_fn, write_results
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
import ops as r5
from r3load import load_r3

r3 = load_r3()
name = sys.argv[1] if len(sys.argv) > 1 else "vitl16"
model = mf.load_model(name, torch.float32)
bb = model.backbone
H = bb.blocks[0].attn.num_heads
sin, cos = bb.rope_embed(H=32, W=32)
d = bb.blocks[0].norm1.weight.numel()
B = 2
NB = 4
W3 = [r3.prepare_block_weights(bb.blocks[i]) for i in range(NB)]
W5 = [r5.prepare_block_weights(bb.blocks[i]) for i in range(NB)]
res = {}


def kernel_time_us(fn, n=20):
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


def graph_us(fn, layers=NB, reps=20):
    with gpu_lock(), torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / layers)
    return min(ts), statistics.median(ts)


for tag, N in (("normal", 1029), ("joint", 1129)):
    prefix = N - 1024
    x32 = (torch.randn(B, N, d, device="cuda") * 2).contiguous()
    st = r5.v2_enter(x32)

    def one_v1(ops, Wl, i):
        return lambda: ops.fused_block(x32, Wl[i], sin, cos, prefix, H)

    def chain_v1(ops, Wl):
        def f():
            x = x32
            for i in range(NB):
                x = ops.fused_block(x, Wl[i], sin, cos, prefix, H)
            return x
        return f

    def chain_v2(pdl, pf=False):
        def f():
            s = st
            for i in range(NB):
                s = r5.fused_block_v2(s, W5[i], sin, cos, prefix, H, pdl=pdl, next_w=W5[(i + 1) % NB], prefetch=pf)
            return s.x
        return f

    variants = {
        "r3_v1": (one_v1(r3, W3, 0), chain_v1(r3, W3)),
        "r5_v1_tuned": (one_v1(r5, W5, 0), chain_v1(r5, W5)),
        "r5_v2_nopdl": (lambda: r5.fused_block_v2(st, W5[0], sin, cos, prefix, H, pdl=False, prefetch=False), chain_v2(False)),
        "r5_v2_pdl": (lambda: r5.fused_block_v2(st, W5[0], sin, cos, prefix, H, pdl=True, prefetch=False), chain_v2(True)),
        "r5_v2_pdl_pf": (lambda: r5.fused_block_v2(st, W5[0], sin, cos, prefix, H, pdl=True, next_w=W5[1], prefetch=True), chain_v2(True, True)),
        "r5_v2_nopdl_pf": (lambda: r5.fused_block_v2(st, W5[0], sin, cos, prefix, H, pdl=False, next_w=W5[1], prefetch=True), chain_v2(False, True)),
    }
    # interleaved (round-robin) CUDA-Graph replays of a 4-layer chain per variant: clock/thermal drift hits all variants equally
    gres = ab({v: fc for v, (f1, fc) in variants.items()}, n=1, rounds=9, reps=3)
    for v, (f1, fc) in variants.items():
        with torch.inference_mode():
            f1(); torch.cuda.synchronize()
        ku, nk = kernel_time_us(f1)
        t = time_fn(f1, warmup=30, iters=100, repeats=3)
        gmin, gp50 = gres[v]["min"] / NB, gres[v]["p50"] / NB
        res[f"{tag}_N{N}_{v}"] = {"kernels": nk, "kernel_us": ku, "event_gpu_us": t["gpu_ms_p50"] * 1e3, "wall_us": t["wall_ms_p50"] * 1e3, "graph_us_per_layer_min": gmin, "graph_us_per_layer_p50": gp50}
        print(f"{name} {tag:6s} N={N} {v:15s} kernels {nk:4.1f}  kernel-time {ku:7.1f}us  event gpu {t['gpu_ms_p50']*1e3:7.1f}us  graph/layer (interleaved) min {gmin:7.1f} p50 {gp50:7.1f}us", flush=True)
write_results("/root/lightly-train/infer-opt/round5_gemm_sm120/results.json", {"block_bench": {name: res}})
