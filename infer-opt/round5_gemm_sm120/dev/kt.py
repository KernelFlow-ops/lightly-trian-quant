"""Kernel-time helper (Round 5): time n back-to-back calls captured in one CUDA graph (0.75us/kernel gaps,
no per-call event/launch overhead).  Holds the GPU lock.  Returns per-call microseconds."""
import sys
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import statistics
import torch
from bench import gpu_lock


def kt(fn, n=20, reps=7, warm=3):
    """fn: callable (may be a list of callables that are cycled to defeat L2 reuse). returns dict(min, p50) in us/call."""
    fns = fn if isinstance(fn, (list, tuple)) else [fn]
    with gpu_lock(), torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for f in fns:
                for _ in range(warm):
                    f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(n):
                fns[i % len(fns)]()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / n)
    return {"min": min(ts), "p50": statistics.median(ts)}


def ab(fns: dict, n=10, rounds=7, reps=3, warm=3):
    """Interleaved A/B kernel timing: every candidate is graph-captured (n calls) and replayed round-robin, so slow drifts
    of the GPU clock affect all candidates equally.  Returns {name: dict(min, p50)} in us/call.  Holds the GPU lock once."""
    res = {k: [] for k in fns}
    with gpu_lock(), torch.inference_mode():
        graphs = {}
        s = torch.cuda.Stream()
        for k, f in fns.items():
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(warm):
                    f()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(n):
                    f()
            graphs[k] = g
            g.replay()
        torch.cuda.synchronize()
        for _ in range(rounds):
            for k, g in graphs.items():
                for _ in range(reps):
                    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                    a.record(); g.replay(); b.record(); b.synchronize()
                    res[k].append(a.elapsed_time(b) * 1e3 / n)
    return {k: {"min": min(v), "p50": statistics.median(v)} for k, v in res.items()}
