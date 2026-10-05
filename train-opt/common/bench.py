"""Locked training measurements and result persistence.

The lock, environment snapshot, and merge writer adapt infer-opt/common/bench.py.
"""
from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch

from .config import PHASE_WEIGHTS

INFER_LOCK = Path("/root/autodl-tmp/infer-opt-tmp/gpu.lock")
TRAIN_LOCK = Path("/root/autodl-tmp/train-opt-tmp/gpu.lock")
LOCK_ENV = "TRAIN_OPT_GPU_LOCK_HELD"


@contextlib.contextmanager
def gpu_lock() -> Iterator[None]:
    """Acquire the inference lock, then the training lock, for GPU measurements."""
    if os.environ.get(LOCK_ENV) == "1":
        yield
        return
    INFER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with INFER_LOCK.open("a") as infer, TRAIN_LOCK.open("a") as train:
        fcntl.flock(infer, fcntl.LOCK_EX)
        try:
            fcntl.flock(train, fcntl.LOCK_EX)
            old = os.environ.get(LOCK_ENV)
            os.environ[LOCK_ENV] = "1"
            try:
                yield
            finally:
                if old is None:
                    os.environ.pop(LOCK_ENV, None)
                else:
                    os.environ[LOCK_ENV] = old
                fcntl.flock(train, fcntl.LOCK_UN)
        finally:
            fcntl.flock(infer, fcntl.LOCK_UN)


@contextlib.contextmanager
def nvtx(name: str) -> Iterator[None]:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("No measurements")
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def assert_gpu_exclusive() -> dict[str, str]:
    """Refuse measurements while an unrelated GPU compute process is present."""
    snapshot = gpu_snapshot()
    allowed = {os.getpid()}
    ancestor = os.getppid()
    while ancestor > 1 and ancestor not in allowed:
        allowed.add(ancestor)
        try:
            line = next(line for line in Path(f"/proc/{ancestor}/status").read_text().splitlines() if line.startswith("PPid:"))
            ancestor = int(line.split()[1])
        except (OSError, StopIteration, ValueError):
            break
    for item in os.environ.get("TRAIN_OPT_GPU_ALLOWED_PIDS", "").split(","):
        if item.strip().isdigit():
            allowed.add(int(item.strip()))
    others = []
    for line in snapshot["compute_apps"].splitlines():
        try:
            pid = int(line.split(",", 1)[0].strip())
        except ValueError:
            continue
        if pid not in allowed:
            others.append(pid)
    if others:
        raise RuntimeError(f"GPU has other compute processes {others}; measurement aborted")
    return snapshot


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _one_timed_step(ctx: Any, batch: Any, step: int, step_fn: Callable[[Any, Any, int], Any], *, segments: bool) -> tuple[Any, dict[str, torch.cuda.Event]]:
    marks: dict[str, torch.cuda.Event] = {}

    def mark(name: str) -> None:
        marks[name] = _event()
        marks[name].record()

    if not segments:
        mark("start")
        out = step_fn(ctx, batch, step)
        mark("end")
        return out, marks

    loss_fn = ctx.ops.loss_all_layers
    backward_fn = ctx.fabric.backward
    opt_fn = ctx.ops.opt_step

    def loss_wrapped(*args: Any, **kwargs: Any) -> Any:
        mark("loss_start")
        result = loss_fn(*args, **kwargs)
        mark("loss_end")
        return result

    def backward_wrapped(*args: Any, **kwargs: Any) -> Any:
        mark("bwd_start")
        result = backward_fn(*args, **kwargs)
        mark("bwd_end")
        return result

    def opt_wrapped(*args: Any, **kwargs: Any) -> Any:
        mark("opt_start")
        result = opt_fn(*args, **kwargs)
        mark("opt_end")
        return result

    ctx.ops.loss_all_layers = loss_wrapped
    ctx.fabric.backward = backward_wrapped
    ctx.ops.opt_step = opt_wrapped
    try:
        mark("start")
        out = step_fn(ctx, batch, step)
        mark("end")
    finally:
        ctx.ops.loss_all_layers = loss_fn
        ctx.fabric.backward = backward_fn
        ctx.ops.opt_step = opt_fn
    return out, marks


def _event_ms(marks: dict[str, torch.cuda.Event], start: str, end: str) -> float | None:
    if start not in marks or end not in marks:
        return None
    return marks[start].elapsed_time(marks[end])


def time_steps(
    ctx: Any,
    batches: Sequence[Any],
    phases: Sequence[str] = ("early", "mid", "late"),
    *,
    warmup: int = 20,
    iters: int = 50,
    repeats: int = 3,
    segments: bool = True,
    step_fn: Callable[[Any, Any, int], Any] | None = None,
) -> dict[str, Any]:
    """Time actual training steps on preloaded real batches, with no per-step sync."""
    if not torch.cuda.is_available():
        raise RuntimeError("time_steps requires CUDA events")
    if not batches or warmup < 0 or iters <= 0 or repeats <= 0:
        raise ValueError("Need nonempty batches and positive iterations/repeats")
    from .harness import set_phase, train_step

    measured_step = step_fn or ctx.meta.get("train_step_fn") or train_step
    batch_size = int(ctx.meta["batch_size"])
    result: dict[str, Any] = {"fixed_phase": True}
    old_override = ctx.meta.get("phase_override")
    old_host_probs = list(ctx.host_probs)
    old_device_probs = [float(value.detach().item()) for value in ctx.model.attn_mask_probs]
    ctx.meta["phase_override"] = True
    counter = 0
    with gpu_lock():
        try:
            result["gpu_snapshot_before"] = assert_gpu_exclusive()
            for phase in phases:
                set_phase(ctx, phase)
                for _ in range(warmup):
                    measured_step(ctx, batches[counter % len(batches)], counter)
                    counter += 1
                _sync()
                torch.cuda.reset_peak_memory_stats()
                repetitions: list[dict[str, Any]] = []
                all_ms: list[float] = []
                stage_ms: dict[str, list[float]] = {key: [] for key in ("fwd", "loss", "bwd", "opt", "other")}
                for _ in range(repeats):
                    marks_by_step = []
                    _sync()
                    wall_start = time.perf_counter()
                    for _ in range(iters):
                        _, marks = _one_timed_step(ctx, batches[counter % len(batches)], counter, measured_step, segments=segments)
                        marks_by_step.append(marks)
                        counter += 1
                    _sync()
                    wall_s = time.perf_counter() - wall_start
                    times = [_event_ms(m, "start", "end") for m in marks_by_step]
                    all_ms.extend(float(v) for v in times if v is not None)
                    repetitions.append({"step_ms": _stats([float(v) for v in times if v is not None]), "wall_s": wall_s})
                    if segments:
                        for marks in marks_by_step:
                            parts = {
                                "fwd": _event_ms(marks, "start", "loss_start"),
                                "loss": _event_ms(marks, "loss_start", "loss_end"),
                                "bwd": _event_ms(marks, "bwd_start", "bwd_end"),
                                "opt": _event_ms(marks, "opt_start", "opt_end"),
                            }
                            for key, value in parts.items():
                                if value is not None:
                                    stage_ms[key].append(value)
                            whole = _event_ms(marks, "start", "end") or 0.0
                            stage_ms["other"].append(max(0.0, whole - sum(v or 0.0 for v in parts.values())))
                mean = statistics.fmean(all_ms)
                result[phase] = {
                    "step_ms": _stats(all_ms),
                    "img_s": batch_size * 1000 / mean,
                    "repeats": repetitions,
                    "breakdown_ms": {key: statistics.fmean(values) if values else None for key, values in stage_ms.items()},
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                }
            result["gpu_snapshot_after"] = assert_gpu_exclusive()
        finally:
            if old_override is None:
                ctx.meta.pop("phase_override", None)
            else:
                ctx.meta["phase_override"] = old_override
            ctx.host_probs[:] = old_host_probs
            for tensor, value in zip(ctx.model.attn_mask_probs, old_device_probs):
                tensor.fill_(value)
    if set(phases) == {"early", "mid", "late"}:
        weighted_ms = sum(float(PHASE_WEIGHTS[p]) * result[p]["step_ms"]["mean"] for p in PHASE_WEIGHTS)
        result["weighted_img_s"] = batch_size * 1000 / weighted_ms
    else:
        result["weighted_img_s"] = None
    return result


def time_e2e(ctx: Any, loader: Any, *, steps: int = 100, start_step: int = 0, step_fn: Callable[[Any, Any, int], Any] | None = None) -> dict[str, Any]:
    """Measure real loading plus training without forcing a sync between steps."""
    from .harness import train_step

    measured_step = step_fn or ctx.meta.get("train_step_fn") or train_step
    with gpu_lock():
        before = assert_gpu_exclusive()
        iterator = iter(loader)
        wait_ms: list[float] = []
        images = 0
        _sync()
        start = time.perf_counter()
        for i in range(steps):
            t0 = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            wait_ms.append((time.perf_counter() - t0) * 1e3)
            images += len(batch["image"])
            measured_step(ctx, batch, start_step + i)
        _sync()
        elapsed = time.perf_counter() - start
        after = assert_gpu_exclusive()
    return {"img_s": images / elapsed, "dataload_wait_ms": statistics.fmean(wait_ms), "steps": steps, "wall_s": elapsed, "gpu_snapshot_before": before, "gpu_snapshot_after": after}


def time_epoch(train_fn: Callable[[], Any], val_fn: Callable[[], Any], ckpt_fn: Callable[[], Any]) -> dict[str, Any]:
    """Measure a complete real epoch; callbacks must do the actual train/val/save."""
    timings: dict[str, Any] = {}
    with gpu_lock():
        timings["gpu_snapshot_before"] = assert_gpu_exclusive()
        for name, fn in (("train", train_fn), ("val", val_fn), ("ckpt", ckpt_fn)):
            _sync()
            start = time.perf_counter()
            fn()
            _sync()
            timings[name] = time.perf_counter() - start
        timings["gpu_snapshot_after"] = assert_gpu_exclusive()
    timings["total"] = sum(timings[name] for name in ("train", "val", "ckpt"))
    return timings


def profile_steps(step_fn: Callable[[int], Any], *, warmup: int = 20, steps: int = 5) -> None:
    """Capture only steady training steps using cudaProfilerStart/Stop."""
    if warmup < 0 or steps <= 0:
        raise ValueError("Invalid profile step counts")
    with gpu_lock():
        assert_gpu_exclusive()
        for i in range(warmup):
            step_fn(i)
        _sync()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            for i in range(steps):
                with nvtx(f"step_{i:02d}"):
                    step_fn(warmup + i)
            _sync()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        assert_gpu_exclusive()


def gpu_snapshot() -> dict[str, str]:
    query = "name,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu"
    gpu = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    if gpu.returncode or apps.returncode:
        raise RuntimeError(f"nvidia-smi snapshot failed: {gpu.stderr.strip()} {apps.stderr.strip()}")
    return {"gpu": gpu.stdout.strip(), "compute_apps": apps.stdout.strip()}


def env_info() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_snapshot": gpu_snapshot(),
        "time": datetime.now(timezone.utc).isoformat(),
    }


def peak_mem() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated_gb": None, "reserved_gb": None}
    return {
        "allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
    }


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def write_results(path: str | Path, payload: dict[str, Any]) -> None:
    """Merge model/variant results atomically without overwriting other agents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        _merge(data, payload)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as tmp:
            json.dump(data, tmp, indent=2, ensure_ascii=False, allow_nan=False)
            tmp.write("\n")
            name = tmp.name
        os.replace(name, path)


def empty_result(*, round_name: str, variant: str, model: str, batch: int, precision: str, patches: Sequence[str], desc: str) -> dict[str, Any]:
    """Create the RULES section 8 schema with honest nulls for missing measurements."""
    phase = {"step_ms": {"p50": None, "p90": None, "mean": None, "std": None}, "img_s": None}
    return {
        "round": round_name, "variant": variant, "model": model, "batch": batch,
        "precision": precision, "patches": list(patches), "desc": desc,
        "gpu_step": {"early": copy.deepcopy(phase), "mid": copy.deepcopy(phase), "late": copy.deepcopy(phase), "weighted_img_s": None},
        "e2e": {"img_s": None, "dataload_wait_ms": None},
        "epoch_s": {"train": None, "val": None, "ckpt": None, "total": None},
        "breakdown_ms": {"fwd": None, "loss": None, "bwd": None, "opt": None, "other": None},
        "kernels_per_step": None, "syncs_per_step": None, "gpu_busy_ratio": None,
        "peak_mem_gb": None, "max_batch": None,
        "accuracy": {
            "level_b": {"loss_rel": None, "grad_cos_min": None, "grad_norm_rel": None, "update_cos_min": None, "pass": None},
            "level_c": {"curve300_max_rel": None, "pass": None},
            "level_d": {"budget_epochs": None, "val_miou": None, "val_head_iou": None,
                        "ratio_vs_baseline": {"miou": None, "head_iou": None}, "pass": None},
        },
        "env": {"torch": None, "gpu_snapshot": None, "time": None}, "logs": [],
    }
