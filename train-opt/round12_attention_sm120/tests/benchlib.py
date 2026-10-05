"""Shared helpers for the round12 attention benchmarks and correctness tests.

Every GPU timing must run under ``common.bench.gpu_lock`` (the dual infer/train lock).
"""
from __future__ import annotations

import math
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]  # .../train-opt
for p in (str(ROOT), str(ROOT.parent / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

SHAPES = {
    # model: (B, H)
    "vitl16": (7, 16),
    "vitb16": (14, 12),
}
D = 64
N_PLAIN, N_JOINT = 1029, 1129
Q_ROWS = 100


def flops_fwd(b: int, h: int, n: int, d: int = D) -> float:
    return 4.0 * b * h * n * n * d


def flops_bwd(b: int, h: int, n: int, d: int = D) -> float:
    return 10.0 * b * h * n * n * d  # 5 matmuls (S recompute, dP, dV, dQ, dK)


def event_us(fn: Callable[[], object], warmup: int = 20, iters: int = 50) -> float:
    """Mean per-call microseconds over ``iters`` launches between CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000.0 / iters


def warm_gpu(seconds: float = 1.5) -> None:
    """Ramp the SM clock before timing (the card idles at ~200 MHz)."""
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    import time
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()


def interleaved(cands: dict[str, Callable[[], object]], warmup: int = 20, iters: int = 50, repeats: int = 3) -> dict[str, float]:
    """Alternate candidates in one process; median of ``repeats`` per-candidate means (CONTRACTS section 6)."""
    for fn in cands.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    runs: dict[str, list[float]] = {k: [] for k in cands}
    for _ in range(repeats):
        for name, fn in cands.items():
            runs[name].append(event_us(fn, warmup=2, iters=iters))
    return {k: statistics.median(v) for k, v in runs.items()}


def require_lock() -> None:
    assert os.environ.get("TRAIN_OPT_GPU_LOCK_HELD") == "1", "run under gpu_run.sh / common.bench.gpu_lock"


def make_rope(n_patch: int = 1024, device="cuda", jitter: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """DINOv3 RopePositionEmbedding tables (fp32 [1024,64], halves duplicated) with random coordinate rescale."""
    import math

    h = w = int(round(n_patch ** 0.5))
    ch = (torch.arange(0.5, h, device=device) / h) * 2 - 1
    cw = (torch.arange(0.5, w, device=device) / w) * 2 - 1
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).flatten(0, 1)  # [HW, 2]
    if jitter:
        coords = coords * torch.empty(2, device=device).uniform_(-0.8, 0.8).exp() + torch.empty(2, device=device).uniform_(-0.1, 0.1)
        coords = coords * torch.empty(1, device=device).uniform_(-0.7, 0.7).exp()
    periods = 100.0 ** (2 * torch.arange(16, device=device, dtype=torch.float32) / 32)
    angles = (2 * math.pi * coords[:, :, None] / periods[None, None, :]).flatten(1, 2)  # [HW, 32]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.sin().contiguous(), angles.cos().contiguous()


def dense_mask(logits: torch.Tensor, n_tokens: int, disabled: torch.Tensor | None) -> torch.Tensor:
    """Source semantics: bool [B,N,N]; True = allowed. logits fp32/bf16 [B,100,128,128]."""
    import torch.nn.functional as F

    b, q = logits.shape[:2]
    prefix = q + 1 + 4
    mask = torch.ones(b, n_tokens, n_tokens, dtype=torch.bool, device=logits.device)
    interp = F.interpolate(logits, size=(32, 32), mode="bilinear").view(b, q, -1)
    mask[:, :q, prefix:] = interp > 0
    if disabled is not None:
        mask[:, :q, prefix:][disabled] = True
    return mask


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double().reshape(-1)
    b = b.detach().double().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def err_stats(x: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    x = x.detach().double()
    ref = ref.detach().double()
    d = (x - ref).abs()
    return {
        "max_abs": float(d.max()),
        "rel_l2": float((x - ref).norm() / ref.norm().clamp_min(1e-30)),
        "cos": cos_sim(x, ref),
    }
