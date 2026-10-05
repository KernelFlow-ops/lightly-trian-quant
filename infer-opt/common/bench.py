"""Shared timing / profiling / result helpers for every infer-opt round.

All timing and profiling must go through `gpu_lock()` so that parallel workers
never benchmark on the GPU at the same time (timings would be meaningless).
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

TMP_ROOT = Path("/root/autodl-tmp/infer-opt-tmp")
LOCK_PATH = TMP_ROOT / "gpu.lock"


@contextlib.contextmanager
def gpu_lock() -> Iterator[None]:
    """Exclusive, cross-process lock around any GPU timing or profiling run.

    Skipped when a parent (e.g. profile_nsys.sh) already holds it.
    """
    if os.environ.get("INFER_OPT_GPU_LOCK_HELD") == "1":
        yield
        return
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def nvtx(name: str) -> Iterator[None]:
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _stats(values: list[float]) -> dict[str, float]:
    s = sorted(values)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]

    return {
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "mean": statistics.fmean(s),
        "std": statistics.pstdev(s),
        "min": s[0],
        "n": len(s),
    }


def time_fn(
    fn: Callable[[], Any],
    *,
    warmup: int = 50,
    iters: int = 200,
    repeats: int = 3,
    lock: bool = True,
) -> dict[str, Any]:
    """Latency of a batch-1 call, synchronized per iteration.

    Returns wall-clock latency (host view, includes CPU launch overhead) and GPU
    stream time between CUDA events. Reported per repeat plus the median over
    repeats of each repeat's p50.
    """
    ctx = gpu_lock() if lock else contextlib.nullcontext()
    with ctx, torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        runs = []
        for _ in range(repeats):
            wall, gpu = [], []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                start.record()
                fn()
                end.record()
                end.synchronize()
                wall.append((time.perf_counter() - t0) * 1e3)
                gpu.append(start.elapsed_time(end))
            runs.append({"wall_ms": _stats(wall), "gpu_ms": _stats(gpu)})
    return {
        "wall_ms_p50": statistics.median(r["wall_ms"]["p50"] for r in runs),
        "gpu_ms_p50": statistics.median(r["gpu_ms"]["p50"] for r in runs),
        "runs": runs,
    }


def profile_fn(fn: Callable[[], Any], *, warmup: int = 20, iters: int = 10) -> None:
    """Run `fn` under cudaProfilerStart/Stop for nsys --capture-range=cudaProfilerApi."""
    with gpu_lock(), torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for i in range(iters):
            with nvtx(f"iter_{i:02d}"):
                fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


def gpu_snapshot() -> dict[str, str]:
    q = "utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        out = f"unavailable: {e}"
    return {"query": q, "value": out}


def env_info() -> dict[str, Any]:
    p = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": p.name,
        "sm": f"{p.major}{p.minor}",
        "num_sms": p.multi_processor_count,
        "gpu_snapshot": gpu_snapshot(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def peak_mem_mb() -> float:
    return torch.cuda.max_memory_allocated() / 2**20


def write_results(path: str | Path, payload: dict[str, Any]) -> None:
    """Merge `payload` into the JSON file at `path` (keyed by model name etc.)."""
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text())
    _deep_update(data, payload)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def set_quiet_env() -> None:
    """Disable LightlyTrain telemetry for fair timing."""
    os.environ.setdefault("LIGHTLY_TRAIN_EVENTS_DISABLED", "1")
