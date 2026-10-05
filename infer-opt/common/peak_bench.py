"""Measure achievable peaks on this GPU (replaces ncu's SOL baseline).

Usage: python common/peak_bench.py --out results/peaks.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import env_info, gpu_lock  # noqa: E402


def _cuda_time(fn, iters: int) -> float:
    """Average GPU ms per call over `iters` back-to-back calls."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def gemm_tflops(m: int, n: int, k: int, dtype: torch.dtype, iters: int = 50) -> float:
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(n, k, device="cuda", dtype=dtype).t()
    ms = _cuda_time(lambda: a @ b, iters)
    return 2 * m * n * k / (ms * 1e-3) / 1e12


def fp8_tflops(m: int, n: int, k: int, iters: int = 50) -> float | str:
    try:
        a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).t()
        one = torch.ones((), device="cuda")
        fn = lambda: torch._scaled_mm(a, b, one, one, out_dtype=torch.bfloat16)  # noqa: E731
        ms = _cuda_time(fn, iters)
        return 2 * m * n * k / (ms * 1e-3) / 1e12
    except Exception as ex:  # noqa: BLE001
        return f"unsupported: {type(ex).__name__}: {ex}"[:300]


def copy_gbps(nbytes: int, iters: int) -> float:
    x = torch.empty(nbytes // 2, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    ms = _cuda_time(lambda: y.copy_(x), iters)
    return 2 * nbytes / (ms * 1e-3) / 1e9  # read + write


def eager_launch_us(n: int = 2000) -> float:
    x = torch.zeros(1, device="cuda")
    for _ in range(100):
        x.add_(1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def graph_launch_us(n_kernels: int = 200, replays: int = 200) -> dict[str, float]:
    x = torch.zeros(1, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x.add_(1)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_kernels):
            x.add_(1)
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(replays):
        g.replay()
    torch.cuda.synchronize()
    per_replay = (time.perf_counter() - t0) / replays * 1e6
    return {"per_kernel_in_graph_us": per_replay / n_kernels, "per_replay_us": per_replay}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    res: dict = {"env": env_info()}
    with gpu_lock(), torch.inference_mode():
        res["bf16_gemm_tflops"] = {
            "8192^3": gemm_tflops(8192, 8192, 8192, torch.bfloat16, 20),
            "4096^3": gemm_tflops(4096, 4096, 4096, torch.bfloat16),
            # ViT-L/16 @512 per-block shapes (M=1029 tokens).
            "qkv_1029x3072x1024": gemm_tflops(1029, 3072, 1024, torch.bfloat16, 200),
            "proj_1029x1024x1024": gemm_tflops(1029, 1024, 1024, torch.bfloat16, 200),
            "fc1_1029x4096x1024": gemm_tflops(1029, 4096, 1024, torch.bfloat16, 200),
            "fc2_1029x1024x4096": gemm_tflops(1029, 1024, 4096, torch.bfloat16, 200),
        }
        res["fp16_gemm_tflops_8192^3"] = gemm_tflops(8192, 8192, 8192, torch.float16, 20)
        res["fp8_e4m3_gemm_tflops"] = {
            "8192^3": fp8_tflops(8192, 8192, 8192, 20),
            "fc1_1029x4096x1024": fp8_tflops(1040, 4096, 1024, 200),
        }
        res["dram_copy_gbps_1GiB"] = copy_gbps(1 << 30, 20)
        res["l2_copy_gbps_16MiB"] = copy_gbps(16 << 20, 500)
        res["eager_launch_us_per_op"] = eager_launch_us()
        res["cuda_graph"] = graph_launch_us()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
