"""Micro-benchmark helpers for round13 (CUDA events, optional CUDA-graph replay).

Every call must happen inside ``common.bench.gpu_lock`` (the dual infer+train lock).
Timing protocol (CONTRACTS section 6): real shapes, CUDA events, warmup >= 20, >= 50 timed
calls, median of 3 repeats.  Candidates are measured in the same process.
"""
from __future__ import annotations

import statistics
from typing import Callable

import torch


def _events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def time_us(fn: Callable[[], object], *, warmup: int = 20, iters: int = 50, repeats: int = 3,
            graph: bool = True, inner: int = 5) -> dict:
    """Median-of-repeats time per call in microseconds.

    With ``graph=True`` the callable is captured ``inner`` times into one CUDA graph and the replay
    is timed (removes host launch overhead and proves capturability).  If capture fails the
    eager loop is used and ``mode`` says so.
    """
    mode = "eager"
    replay = None
    if graph:
        try:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                for _ in range(inner):
                    fn()
            replay = g.replay
            mode = f"graph x{inner}"
        except Exception as exc:  # capture not possible -> eager
            replay = None
            mode = f"eager (graph failed: {type(exc).__name__}: {str(exc)[:80]})"
            torch.cuda.synchronize()
    if replay is None:
        for _ in range(warmup):
            fn()
        n_inner = 1
        run = fn
    else:
        for _ in range(max(2, warmup // inner)):
            replay()
        n_inner = inner
        run = replay
    torch.cuda.synchronize()
    reps = []
    calls = max(1, iters // n_inner)
    for _ in range(repeats):
        a, b = _events()
        a.record()
        for _ in range(calls):
            run()
        b.record()
        torch.cuda.synchronize()
        reps.append(a.elapsed_time(b) * 1000.0 / (calls * n_inner))
    return {"us": statistics.median(reps), "reps": reps, "mode": mode}


def err_stats(out: torch.Tensor, ref: torch.Tensor) -> dict:
    o, r = out.detach().float().reshape(-1), ref.detach().float().reshape(-1)
    diff = (o - r).abs()
    denom = r.abs().max().clamp_min(1e-12)
    cos = torch.dot(o, r) / (o.norm() * r.norm()).clamp_min(1e-30)
    return {
        "max_abs": float(diff.max()),
        "max_rel_to_absmax": float(diff.max() / denom),
        "rel_l2": float((o - r).norm() / r.norm().clamp_min(1e-30)),
        "cos": float(cos),
    }
