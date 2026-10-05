"""Flat fp32 parameter / gradient / Adam-state buffers with a two-kernel clip + AdamW (round15).

``FlatParams`` owns the flat buffers (contract CONTRACTS section 4): every Parameter's ``.data`` is a view of
``master`` and (after the first optimizer call) its ``.grad`` is a view of ``grad``; Parameter identity, state_dict keys
and shapes do not change.  ``FlatAdamW`` runs

    K1  deterministic global sum of squares  (+ clip coefficient, step counters, per-group lr table, all on the device)
    K2  AdamW update + gradient zeroing      (one block per 4096-float chunk, chunk -> group table)

with no host synchronisation, so the whole thing is CUDA-graph capturable.  lr / weight decay come from a per-group
device table; the TwoStageWarmupPoly learning-rate schedule is evaluated on the device from a device step counter.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

CHUNK = 4096
HYP = 8
EFF = 4
TMP = Path("/root/autodl-tmp/train-opt-tmp/round15_optimizer_flat")
_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("MAX_JOBS", "6")
        build = TMP / "build"
        build.mkdir(parents=True, exist_ok=True)
        src = Path(__file__).resolve().parent / "flat_adamw.cu"
        _EXT = load(name="trainopt_flat_adamw", sources=[str(src)], build_directory=str(build),
                    extra_cuda_cflags=["-O3", "--fmad=false", "-lineinfo"], extra_cflags=["-O3"], verbose=False)
    return _EXT


def _dense(t: torch.Tensor) -> bool:
    """True if ``t`` occupies a dense, non-overlapping block of storage (any permutation of dims)."""
    if t.numel() == 0:
        return False
    dims = sorted((s, st) for s, st in zip(t.shape, t.stride()) if s != 1)
    expect = 1
    for s, st in sorted(dims, key=lambda x: x[1]):
        if st != expect:
            return False
        expect *= s
    return True


def _round_up(n: int, m: int = CHUNK) -> int:
    return (n + m - 1) // m * m


class FlatParams:
    """Flat fp32 ``master`` / ``grad`` (plus Adam ``m`` / ``v``) storage for a list of Parameters."""

    def __init__(self, params: Sequence[torch.nn.Parameter], names: Sequence[str] | None = None) -> None:
        if not params:
            raise ValueError("no parameters")
        device = params[0].device
        self.params = list(params)
        self.names = list(names) if names is not None else [f"p{i}" for i in range(len(params))]
        self.device = device
        self.offsets: list[int] = []
        self.numels: list[int] = []
        total = 0
        for p in self.params:
            if p.device != device or p.dtype != torch.float32:
                raise ValueError("flat storage requires fp32 parameters on one device")
            if not _dense(p.data):
                raise ValueError("parameter is not dense")
            self.offsets.append(total)
            self.numels.append(p.numel())
            total += _round_up(p.numel())
        self.total = total
        self.n_chunks = total // CHUNK
        self.master = torch.zeros(total, dtype=torch.float32, device=device)
        self.grad = torch.zeros(total, dtype=torch.float32, device=device)
        self.m = torch.zeros(total, dtype=torch.float32, device=device)
        self.v = torch.zeros(total, dtype=torch.float32, device=device)
        self.writers: list[Callable[["FlatParams"], None]] = []
        with torch.no_grad():
            for i, p in enumerate(self.params):
                view = self.view(self.master, i)
                view.copy_(p.data)
                p.data = view

    def view(self, buf: torch.Tensor, i: int) -> torch.Tensor:
        p = self.params[i]
        return buf.as_strided(p.shape, p.stride(), self.offsets[i])

    def register_weight_writer(self, fn: Callable[["FlatParams"], None]) -> None:
        """``fn(flat)`` runs on the current stream right after every optimizer update (also inside a CUDA graph)."""
        self.writers.append(fn)


class FlatAdamW:
    """Clip + AdamW over a ``FlatParams`` (torch AdamW semantics, decoupled weight decay)."""

    def __init__(self, params: Sequence[torch.nn.Parameter], groups: Sequence[dict[str, Any]], group_of: Sequence[int],
                 *, names: Sequence[str] | None = None, sched: dict[str, float] | None = None, max_norm: float = 0.0) -> None:
        """``groups[g]``: dict(base_lr, wd, betas, eps, kind) with kind 0 = non-ViT, 1 = ViT, 2 = external lr.
        ``group_of[i]`` is the group index of parameter i.  ``sched``: nw, vw, total, power (TwoStageWarmupPoly)."""
        self.flat = FlatParams(params, names)
        self.ext = extension()
        device = self.flat.device
        self.max_norm = float(max_norm)
        s = sched or dict(nw=0.0, vw=0.0, total=1.0, power=1.0)
        self.nw, self.vw, self.total, self.power = float(s["nw"]), float(s["vw"]), float(s["total"]), float(s["power"])
        self.groups = [dict(g) for g in groups]
        self.group_of = list(group_of)
        n_groups = len(self.groups)
        rows = []
        for g in self.groups:
            b1, b2 = g["betas"]
            rows.append([float(g["base_lr"]), float(g["wd"]), float(b1), float(b2), float(g["eps"]), float(g["kind"]),
                         1.0 - float(b1), 1.0 - float(b2)])
        rows.append([0.0, 0.0, 0.9, 0.999, 1e-8, 3.0, 0.1, 0.001])          # dead row (no gradient: frozen)
        self.dead_gid = n_groups
        self.hyp = torch.tensor(rows, dtype=torch.float64, device=device)
        self.ext_lr = torch.zeros(n_groups + 1, dtype=torch.float64, device=device)
        self.eff = torch.zeros((n_groups + 1) * EFF, dtype=torch.float32, device=device)
        self.grid = 8 * torch.cuda.get_device_properties(device).multi_processor_count
        self.partial = torch.zeros(max(self.grid, 1), dtype=torch.float32, device=device)
        self.ticket = torch.zeros(1, dtype=torch.int32, device=device)
        self.ctl = torch.zeros(4, dtype=torch.int32, device=device)           # [adam_t, sched_step, lr_step_used, -]
        self.norm = torch.zeros((), dtype=torch.float32, device=device)
        self.clip = torch.zeros(1, dtype=torch.float32, device=device)
        self.dead = [False] * len(self.flat.params)
        self.adopted = False
        self.graph_mode = False
        self.external_lr = any(g["kind"] == 2 for g in self.groups)
        self._write_chunk_table()

    # ------------------------------------------------------------------ tables
    def _write_chunk_table(self) -> None:
        table = [0] * self.flat.n_chunks
        for i, off in enumerate(self.flat.offsets):
            gid = self.dead_gid if self.dead[i] else self.group_of[i]
            first = off // CHUNK
            for c in range(first, first + _round_up(self.flat.numels[i]) // CHUNK):
                table[c] = gid
        self.chunk_gid = torch.tensor(table, dtype=torch.int32, device=self.flat.device)

    def set_external_lr(self, lrs: Sequence[float]) -> None:
        """Fallback for schedulers other than TwoStageWarmupPoly (host values, NOT graph capturable)."""
        self.ext_lr[: len(lrs)].copy_(torch.tensor(list(map(float, lrs)), dtype=torch.float64), non_blocking=False)

    # ------------------------------------------------------------------ gradients
    @torch.no_grad()
    def adopt_grads(self) -> list[str]:
        """First eager step: move the autograd gradients into the flat buffer and make ``p.grad`` views of it.
        Parameters that received no gradient are frozen (lr = wd = 0, state untouched).  Returns their names."""
        dead = []
        for i, p in enumerate(self.flat.params):
            g = p.grad
            if g is None:
                self.dead[i] = True
                dead.append(self.flat.names[i])
                continue
            view = self.flat.view(self.flat.grad, i)
            view.copy_(g)
            p.grad = view
        if dead:
            self._write_chunk_table()
        self.adopted = True
        return dead

    def zero_grad(self) -> None:
        self.flat.grad.zero_()

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, max_norm: float | None = None) -> torch.Tensor:
        if not self.adopted:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("FlatAdamW must take one eager step before CUDA graph capture")
            self.adopt_grads()
        mx = self.max_norm if max_norm is None else float(max_norm)
        f = self.flat
        self.ext.sumsq_prep(f.grad, self.partial, self.ticket, self.ctl, self.hyp, self.ext_lr, self.eff, self.norm, self.clip,
                            self.grid, mx, self.nw, self.vw, self.total, self.power)
        self.ext.update(f.master, f.grad, f.m, f.v, self.chunk_gid, self.hyp, self.eff, self.clip)
        for w in f.writers:
            w(f)
        return self.norm

    # ------------------------------------------------------------------ host mirrors
    def device_counters(self) -> tuple[int, int]:
        c = self.ctl.cpu().tolist()
        return int(c[0]), int(c[1])

    def set_counters(self, adam_t: int, sched_step: int) -> None:
        self.ctl[0] = int(adam_t)
        self.ctl[1] = int(sched_step)

    def lr_table(self) -> list[float]:
        """Effective lr of the last update per group (host copy; synchronises)."""
        e = self.eff.view(-1, EFF)[:, 3].cpu().tolist()
        return e[:-1]

    # ------------------------------------------------------------------ state
    def live(self) -> list[int]:
        return [i for i in range(len(self.flat.params)) if not self.dead[i]]

    @torch.no_grad()
    def load_param_state(self, i: int, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor) -> None:
        self.flat.view(self.flat.m, i).copy_(exp_avg)
        self.flat.view(self.flat.v, i).copy_(exp_avg_sq)

    def param_state(self, i: int, *, clone: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        m, v = self.flat.view(self.flat.m, i), self.flat.view(self.flat.v, i)
        return (m.clone(), v.clone()) if clone else (m, v)
