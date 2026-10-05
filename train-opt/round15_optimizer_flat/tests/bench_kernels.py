"""Standalone K1/K2 throughput on a 310M-float flat buffer for kernel variants:  python bench_kernels.py <variant.cu> [<variant2.cu> ...]
The variant files must define the same module interface as impl/flat_adamw.cu."""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/lightly-train/train-opt")
import torch
from torch.utils.cpp_extension import load

from common import bench

N = 310 * 1024 * 1024
CHUNK = 4096
build = Path("/root/autodl-tmp/train-opt-tmp/round15_optimizer_flat/build_var")
build.mkdir(parents=True, exist_ok=True)


def tm(fn, n=40):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in ev)
    return ms[len(ms) // 2]


with bench.gpu_lock():
    p = torch.randn(N, device="cuda"); g = torch.randn(N, device="cuda") * 1e-3
    m = torch.zeros(N, device="cuda"); v = torch.zeros(N, device="cuda")
    nch = N // CHUNK
    gid = torch.zeros(nch, dtype=torch.int32, device="cuda")
    hyp = torch.tensor([[1e-4, 0.05, 0.9, 0.999, 1e-8, 0.0, 0.1, 0.001], [0, 0, 0.9, 0.999, 1e-8, 3.0, 0.1, 0.001]], dtype=torch.float64, device="cuda")
    ext_lr = torch.zeros(2, dtype=torch.float64, device="cuda")
    eff = torch.zeros(8, dtype=torch.float32, device="cuda")
    partial = torch.zeros(4096, device="cuda"); ticket = torch.zeros(1, dtype=torch.int32, device="cuda")
    ctl = torch.zeros(4, dtype=torch.int32, device="cuda"); norm = torch.zeros((), device="cuda"); clip = torch.zeros(1, device="cuda")
    grid = 8 * torch.cuda.get_device_properties(0).multi_processor_count
    for path in sys.argv[1:]:
        name = "var_" + Path(path).stem
        ext = load(name=name, sources=[path], build_directory=str(build), extra_cuda_cflags=["-O3", "--fmad=false", "-lineinfo"], verbose=False)
        k1 = tm(lambda: ext.sumsq_prep(g, partial, ticket, ctl, hyp, ext_lr, eff, norm, clip, grid, 0.01, 500.0, 1000.0, 10000.0, 0.9))
        g.normal_().mul_(1e-3)
        k2 = tm(lambda: ext.update(p, g, m, v, gid, hyp, eff, clip))
        print(f"{Path(path).stem:24s} K1 {k1:.3f} ms {N*4/k1/1e6:.0f} GB/s | K2 {k2:.3f} ms {N*32/k2/1e6:.0f} GB/s | total {k1+k2:.3f} ms")
