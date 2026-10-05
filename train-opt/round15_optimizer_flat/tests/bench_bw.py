"""Practical DRAM ceiling on this GPU for the access patterns of the flat optimizer (copy / read-only reduce / memset / fused AdamW kernels)."""
import sys, os
sys.path.insert(0, "/root/lightly-train/train-opt")
import torch
from common import bench

N = 310 * 1024 * 1024


def tm(fn, n=30):
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
    a = torch.randn(N, device="cuda")
    b = torch.empty_like(a)
    c = torch.randn(N, device="cuda")
    d = torch.randn(N, device="cuda")
    t = tm(lambda: b.copy_(a)); print(f"copy      {t:.3f} ms  {N*8/t/1e6:.0f} GB/s (4B read + 4B write)")
    t = tm(lambda: a.zero_()); print(f"memset    {t:.3f} ms  {N*4/t/1e6:.0f} GB/s (write)")
    a.normal_()
    t = tm(lambda: torch.linalg.vector_norm(a)); print(f"norm      {t:.3f} ms  {N*4/t/1e6:.0f} GB/s (read)")
    t = tm(lambda: torch._foreach_add_([b, c], [a, d])) if False else tm(lambda: b.add_(a)); print(f"add_      {t:.3f} ms  {N*12/t/1e6:.0f} GB/s (2 read + 1 write)")
    t = tm(lambda: torch._fused_adamw_([b], [a], [c], [d], [], [torch.tensor(1.0, device="cuda")], lr=1e-4, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8, amsgrad=False, maximize=False)) if False else None
