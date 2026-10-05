"""Operator-level timing helper for round11: CUDA events, candidates alternated in one process.

Protocol (CONTRACTS section 6): real shapes, warmup >= 20, >= 50 timed iterations, 3 repeats,
median of the repeats; candidates are interleaved repeat by repeat so that clock/thermal drift
hits every candidate equally.  A candidate that raises is recorded as a failure with the error text.
"""
from __future__ import annotations

import statistics
import traceback
from typing import Callable

import torch


def _time_once(fn: Callable[[], object], iters: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def compare(candidates: dict[str, Callable[[], object]], *, warmup: int = 20, iters: int = 50, repeats: int = 3) -> dict[str, dict]:
    """Return {name: {"ms": median, "runs": [...]}} or {name: {"error": text}}."""
    out: dict[str, dict] = {}
    alive: dict[str, Callable[[], object]] = {}
    for name, fn in candidates.items():
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            alive[name] = fn
            out[name] = {"runs": []}
        except Exception as exc:  # unsupported backend/shape combination is a recorded result
            torch.cuda.synchronize()
            out[name] = {"error": f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ''}"}
    for _ in range(repeats):
        for name, fn in alive.items():
            out[name]["runs"].append(_time_once(fn, iters))
    for name in alive:
        out[name]["ms"] = statistics.median(out[name]["runs"])
    return out


def fmt(results: dict[str, dict]) -> str:
    lines = []
    for name, item in results.items():
        if "error" in item:
            lines.append(f"  {name:44s} FAIL  {item['error']}")
        else:
            lines.append(f"  {name:44s} {item['ms']:9.3f} ms   runs={[round(v, 3) for v in item['runs']]}")
    return "\n".join(lines)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def errors(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    out, ref = out.detach().float(), ref.detach().float()
    diff = (out - ref).abs()
    return {
        "max_abs": float(diff.max()),
        "max_rel": float((diff / ref.abs().clamp_min(1e-6)).max()),
        "rel_l2": float(diff.norm() / ref.norm().clamp_min(1e-30)),
        "cos": cosine(out, ref),
    }
