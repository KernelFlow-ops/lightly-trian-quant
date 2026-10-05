"""Correctness (vs round7 dw_ln_kernel) + graph-replay microbench of the round-11 dw_ln2 kernel on the real head shapes.

  python tests/test_dwln.py [--cfgs '[{"TH":4,"TW":8,"PPW":4,"MINB":2,"FAST_GELU":0}]'] [--bench-only]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import r11_common as c
from lock2 import gpu_lock2, other_gpu_procs
import head as hd7   # round7 (read-only)

HERE = Path(__file__).resolve().parents[1] / "impl"


def load_ext(**cfg):
    from torch.utils.cpp_extension import load
    tag = hashlib.md5(repr(sorted(cfg.items())).encode()).hexdigest()[:8]
    build = Path("/root/autodl-tmp/infer-opt-tmp/round11_attn_tier0/build_dwln_" + tag)
    build.mkdir(parents=True, exist_ok=True)
    return load(name="r11_dwln_" + tag, sources=[str(HERE / "head_ext11.cu")], build_directory=str(build),
                extra_cuda_cflags=["-O3", "-arch=sm_120a"] + [f"-D{k}={v}" for k, v in cfg.items()], verbose=False)


def bench(fn, iters=100, reps=5):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        for _ in range(iters // 10):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / iters)
    return sorted(ts)[len(ts) // 2]


ap = argparse.ArgumentParser()
ap.add_argument("--cfgs", default='[{"TH":4,"TW":8,"PPW":4,"MINB":2,"FAST_GELU":0}]')
a = ap.parse_args()
cfgs = json.loads(a.cfgs)
torch.manual_seed(0)
old = hd7.ext()
exts = [(cfg, load_ext(**cfg)) for cfg in cfgs]       # compile outside the GPU lock
print("compiled", len(exts), flush=True)
with gpu_lock2(), torch.inference_mode():
    print("other gpu procs:", other_gpu_procs())
    for C in (1024, 768):
        for Hs in (32, 64):
            B = 2
            # realistic scales: GEMM output pre-bias, bf16
            G = (torch.randn(B * Hs * Hs, 4 * C, device="cuda") * 2).bfloat16()
            bias = torch.randn(C, device="cuda").bfloat16().float()
            dww = (torch.randn(9, C, device="cuda") * 0.3).bfloat16()
            gam = (1 + 0.1 * torch.randn(C, device="cuda")).float()
            bet = (0.1 * torch.randn(C, device="cuda")).float()
            o_old = torch.empty(B, 2 * Hs, 2 * Hs, C, device="cuda", dtype=torch.bfloat16)
            old.dw_ln_launch(G, bias, dww, gam, bet, o_old, B, Hs, Hs, 1e-6)
            t_old = bench(lambda: old.dw_ln_launch(G, bias, dww, gam, bet, o_old, B, Hs, Hs, 1e-6))
            line = f"C={C} Hs={Hs}: old {t_old:7.1f} us |"
            for cfg, e in exts:
                o_new = torch.empty_like(o_old)
                e.dw_ln_launch(G, bias, dww, gam, bet, o_new, B, Hs, Hs, 1e-6)
                ne = (o_new != o_old).float().mean().item()
                mx = (o_new.float() - o_old.float()).abs().max().item()
                t_new = bench(lambda: e.dw_ln_launch(G, bias, dww, gam, bet, o_new, B, Hs, Hs, 1e-6))
                line += f" {t_new:7.1f} us (differ {ne:.2e}, max {mx:.3g}) |"
            print(line, flush=True)
    print("cfgs:", cfgs)
