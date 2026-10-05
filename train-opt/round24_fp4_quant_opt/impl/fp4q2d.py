"""Round21 fused FP4 quantizers (NVFP4 ``nv4`` / MXFP4 ``mx4``) for sm_120: python interface of ``fp4_kernels.cu``.

Formats / layouts are the ``FP4Tensor`` contract of CONTRACTS 11.2 / 12.2 (bit-exact reference: ``round20_fp4_probe/impl/fp4_ref.py``)::

    q   [R, C/2]  u8 (e2m1, even index = low nibble, K-major), sf  [ceil(R/128), rk, 32, 4, 4] u8 blocked scales (e4m3 for nv4, ue8m0 for mx4)
    qt  [C, Kt/2] u8 (transposed copy, blocks along the token axis; Kt = Rpad, zero padded), sft [ceil(C/128), Kt/(4 vec), 32, 4, 4]
    ts  fp32 [1] per-tensor scale of the row copy (nv4 only), tst fp32 [1] of the transposed copy (separate: RHT / prefix change the amax)

Interface
    quant_act(x, fmt, row=, col=, rht_row=, rht_col=, sr_row=, sr_col=, prefix=(B, N, npre), amax="delayed"|"exact"|"ts", state=, site=, ts=, ...) -> FP4Q
    quant_weight(w, fmt, amax=..., state=, site=, ts=) -> FP4Q      (2D scales: nv4 16x16, mx4 32x32; q along K, qt = W^T along N; RN only; both copies identical numerics)
    FP4State(n_sites): amax slots [2 (prev, next), n_sites, 2 (row copy, col copy)], device step / seed counters; ``state.roll()`` once per step (graph capturable)

``prefix=(B, N, npre)``: tokens are ``b*N + t``; the first ``npre`` tokens of every image are *prefix* tokens: they are not part of the transposed copy (``Kt = B*(N-npre)``, token
``b*N+npre+p`` -> column ``b*(N-npre)+p``, per-copy amax of the transposed copy = patch tokens only) and are gathered as bf16 ``xp [B*npre, C]`` for the bf16 side GEMM.  The row copy
covers all R tokens.

amax modes (nv4)
    "delayed" (default): ts = fl(headroom * prev_amax / 2688), headroom = 2.0 by default (``headroom=`` of quant_act / quant_weight; recorded per site in ``state.headroom`` for the monitor of
                         ``state.roll()`` / ``state.stats()``) with prev = the same site's amax of the previous step (``state.roll()``); the current amax of the (rotated) values is published with
                         atomicMax into the next slot.  The first call of a site (python flag, run eagerly before graph capture) initialises prev with an exact amax pass.
    "exact"            : separate amax kernel (rotated values, patch tokens only for the col copy), then the quantizer; device-side ts = amax/2688 (see ``div_true``).
    "ts"               : explicit ``ts`` tensor [2] (row, col) / (ts_row, ts_col); used by the gate-A tests.
``div_true``: False (default) = "/6" and "/2688" are reciprocal multiplications, the behaviour of fp4_ref on CUDA tensors (torch divides by a python scalar with a reciprocal multiply);
True = IEEE fp32 division (fp4_ref on CPU tensors = literal CONTRACTS 12.2).

SR (Philox4x32-10, counter = (flat output index / 4, 2*sr_offset + copy, 0, 0), key = device int64 seed): ``seed`` tensor or ``state.seed`` (+1 per ``roll``); ``sr_offset`` is the fixed per-call-site
offset.  ``u_row [R, C]`` / ``u_col [C, Kt]`` fp32 uniforms in [0, 1) override the generator (bit-exact tests vs ``fp4_ref.e2m1_sr``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_BUILD = Path("/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/build/fp4_kernels2d")
_VARIANT = os.environ.get("R24_VARIANT", "")  # experiments only: R24_VARIANT=name R24_DEFS="-DPHILOX_ROUNDS=7 ..." (own build dir / module name)
_DEFS = os.environ.get("R24_DEFS", "").split()
_EXT = None
NPART = 16  # = NPART of fp4_kernels2.cu
# Philox rounds of the in-kernel SR stream: 10 (standard Philox4x32-10, bit-identical to round21's ``fp4q``) or 7 (Philox4x32-7: BigCrush-clean per Salmon et al. 2011, ~4-6 % faster in SR mode).
# Default from the env var FP4Q2_PHILOX, per call ``philox_rounds=``.  Explicit ``u_row`` / ``u_col`` paths are unaffected.
PHILOX_ROUNDS = int(os.environ.get("FP4Q2_PHILOX", "10"))
# quant_weight(row=...) default.  round22's fp4_block22 drops the weight row copy (q / sf) right after the call, so a stack that does so may set ``fp4q2.WEIGHT_ROW = False``
# (run.py of round24 does) to skip writing it.  True = round21 behaviour.
# SR rule of the in-kernel (Philox) stochastic rounding: "exact" = u < p with u = (2h+1)/2^17 (round21, bit-identical), "dither" = cheaper statistically equivalent rule (fast kernels only), see fp4_kernels2.cu
SR_MODE = os.environ.get("FP4Q2_SR", "exact")
WEIGHT_ROW = os.environ.get("FP4Q2_WEIGHT_ROW", "1") != "0"

FMT = {"nv4": 0, "mx4": 1}
VEC = {"nv4": 16, "mx4": 32}
_RHT_SIGNS = (1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1)  # fp4_ref.rht_signs() (seed 20251003); asserted equal in tests/test_fp4q.py
RHT_SIGN_MASK = sum(1 << j for j, s in enumerate(_RHT_SIGNS) if s < 0)


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("MAX_JOBS", "8")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        variant, extra_defs = _VARIANT, _DEFS
        build = _BUILD if not variant else _BUILD.parent / f"fp4_kernels2d_{variant}"
        build.mkdir(parents=True, exist_ok=True)
        _EXT = load(
            name="trainopt_r24_fp4_kernels2d" + (f"_{variant}" if variant else ""),
            sources=[(os.environ.get("R24_SRC") if variant else None) or str(_HERE / "fp4_kernels2d.cu")],  # R24_SRC: experiments only (a scratch copy of the .cu; always together with R24_VARIANT)
            build_directory=str(build),
            extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-Xptxas", "-v"] + extra_defs,
            verbose=bool(int(os.environ.get("R24_VERBOSE_BUILD", "0"))),
        )
    return _EXT


def pad128(n: int) -> int:
    return -(-n // 128) * 128


class FP4State:
    """Per-model amax / RNG state: ``amax [2, n_sites, 2]`` (prev, next) x site x (row copy, col copy), device ``seed`` and ``step`` counters, per-site ``headroom`` (device fp32 [n_sites],
    recorded by the delayed-mode quantizer calls, default 2.0) and the saturation monitor counters updated by ``roll()``."""

    def __init__(self, n_sites: int, seed: int = 20251003, device="cuda"):
        self.n_sites = n_sites
        self.amax = torch.zeros(2, n_sites, 2, dtype=torch.float32, device=device)
        self.seed = torch.tensor([seed], dtype=torch.int64, device=device)
        self.step = torch.zeros(1, dtype=torch.int64, device=device)
        self.inited = [False] * n_sites
        self.headroom = torch.full((n_sites,), 2.0, dtype=torch.float32, device=device)
        self.headroom_host = [2.0] * n_sites
        # spread amax publication of the fast kernels: [site * 2 + copy, NPART, 8] (one 32 B sector per sub-slot), folded into amax[1] by roll() / fold()
        self.parts = torch.zeros(2 * n_sites, NPART, 8, dtype=torch.float32, device=device)
        self.mon_cnt = torch.zeros(2, 2 * n_sites, dtype=torch.int32, device=device)  # [0] comparisons, [1] exceeded
        self.mon_maxr = torch.zeros(2 * n_sites, dtype=torch.float32, device=device)  # max of next / prev

    def roll(self):
        """Once per step (before the quantizers of the step): prev <- next, next <- 0, seed += 1, step += 1, plus the saturation monitor (counts the slots where the amax that
        was just produced exceeded ``headroom * prev``, the value its ts was built from).  Pure device work, CUDA graph capturable, no host sync."""
        extension().amax_roll_mon(self.amax, self.headroom, self.mon_cnt, self.mon_maxr, self.seed, self.step, self.parts)

    def fold(self):
        """Fold the spread publication of the fast kernels into the ``next`` amax slot (amax[1]) and clear it (what roll() does inside the graph; for tests / inspection)."""
        m = self.parts[:, :, 0].amax(dim=1).view(self.n_sites, 2)
        self.amax[1].copy_(torch.maximum(self.amax[1], m))
        self.parts.zero_()

    def set_headroom(self, site: int, headroom: float):
        """Record the per-site headroom used by the monitor (called by quant_act / quant_weight in delayed mode; a device write only when the value changes)."""
        h = float(headroom)
        if self.headroom_host[site] != h:
            self.headroom_host[site] = h
            self.headroom[site] = h

    def reset_stats(self):
        self.mon_cnt.zero_()
        self.mon_maxr.zero_()

    def stats(self) -> dict:
        """Saturation statistics on the host (device -> host copy; NOT for use inside a CUDA graph).  Per site and copy (``row`` = row copy / weight, ``col`` = transposed copy):
        ``calls`` (roll comparisons), ``exceed`` (next > prev * headroom: the nv4 block scale saturated at 448 in that step), ``max_ratio`` (max next / prev; inf when prev was 0)."""
        cnt = self.mon_cnt.cpu().view(2, self.n_sites, 2)
        mr = self.mon_maxr.cpu().view(self.n_sites, 2)
        out = {"n_sites": self.n_sites, "step": int(self.step.item()), "headroom": [float(v) for v in self.headroom.cpu()]}
        for c, name in enumerate(("row", "col")):
            out[name] = {"calls": [int(v) for v in cnt[0, :, c]], "exceed": [int(v) for v in cnt[1, :, c]],
                         "max_ratio": [float("inf") if float(v) >= 3.0e38 else float(v) for v in mr[:, c]]}
        out["total_calls"] = int(cnt[0].sum())
        out["total_exceed"] = int(cnt[1].sum())
        out["exceed_rate"] = out["total_exceed"] / max(1, out["total_calls"])
        return out

    def reset(self):
        self.amax.zero_()
        self.parts.zero_()
        self.step.zero_()
        self.inited = [False] * self.n_sites
        self.reset_stats()


@dataclass
class FP4Q:
    fmt: str
    rows: int
    cols: int
    kt: int = 0
    q: Optional[torch.Tensor] = None
    sf: Optional[torch.Tensor] = None
    qt: Optional[torch.Tensor] = None
    sft: Optional[torch.Tensor] = None
    ts: Optional[torch.Tensor] = None  # [1] fp32 (row copy) -- nv4 only
    tst: Optional[torch.Tensor] = None  # [1] fp32 (transposed copy)
    xp: Optional[torch.Tensor] = None  # bf16 [B*npre, C] prefix rows (prefix mode)

    def row_triple(self):
        """(q, sf, ts) as consumed by ``round20 fp4_cands.operand``."""
        return self.q, self.sf, self.ts

    def col_triple(self):
        return self.qt, self.sft, self.tst


def _site_amax_init(x, state, site, fmt, row, col, kw):
    prev = state.amax[0, site]
    prev.zero_()
    extension().quant_act(x, 0, row, col, kw["rpad"], kw["B"], kw["N"], kw["npre"], kw["rht_r"], kw["rht_c"], RHT_SIGN_MASK, False, False, None, None, None, 0, 1, None, 1.0, prev, False,
                          True, 10, None, 0)
    state.inited[site] = True


def quant_act(x: torch.Tensor, fmt: str = "nv4", *, row: bool = True, col: bool = True, rht_row: bool = False, rht_col: bool = False, sr_row: bool = False, sr_col: bool = False,
              prefix: Optional[Tuple[int, int, int]] = None, rpad: Optional[int] = None, amax: str = "delayed", state: Optional[FP4State] = None, site: int = 0,
              ts=None, seed: Optional[torch.Tensor] = None, sr_offset: int = 0, u_row: Optional[torch.Tensor] = None, u_col: Optional[torch.Tensor] = None,
              div_true: bool = False, headroom: float = 2.0, philox_rounds: Optional[int] = None, sr_mode: Optional[str] = None) -> FP4Q:
    """Quantize an activation / gradient ``x [R, C]`` (contiguous bf16 / fp32, C % 64 == 0) into the requested FP4 copies with one read of ``x``."""
    ext = extension()
    rounds = PHILOX_ROUNDS if philox_rounds is None else int(philox_rounds)
    srm = {"exact": 0, "dither": 1}[SR_MODE if sr_mode is None else sr_mode] if (sr_row or sr_col) else 0  # the SR rule only matters for SR copies
    R, C = x.shape
    B, N, npre = prefix if prefix is not None else (0, 0, 0)
    kw = dict(rpad=0 if rpad is None else rpad, B=B, N=N, npre=npre, rht_r=rht_row, rht_c=rht_col)
    f = FMT[fmt]
    if (sr_row or sr_col) and seed is None and state is not None:
        seed = state.seed
    tsrc, ts_mode, mul, nxt, parts = None, 0, 1.0, None, None
    if f == 0:
        if amax == "ts":
            tsrc = torch.as_tensor(ts, dtype=torch.float32, device=x.device).reshape(-1)
            if tsrc.numel() == 1:
                tsrc = tsrc.repeat(2)
            tsrc = tsrc.contiguous()
        elif amax == "exact":
            tsrc = torch.zeros(2, dtype=torch.float32, device=x.device)
            ext.quant_act(x, 0, row, col, kw["rpad"], B, N, npre, rht_row, rht_col, RHT_SIGN_MASK, False, False, None, None, None, 0, 1, None, 1.0, tsrc, div_true, True, 10, None, 0)
            ts_mode = 1
        elif amax == "delayed":
            assert state is not None, "amax='delayed' needs an FP4State"
            if not state.inited[site]:
                _site_amax_init(x, state, site, fmt, row, col, kw)
            assert headroom > 0, headroom
            state.set_headroom(site, headroom)
            tsrc, ts_mode, mul, nxt = state.amax[0, site], 1, float(headroom), state.amax[1, site]
            parts = state.parts[2 * site:2 * site + 2]
        else:
            raise ValueError(amax)
    q, sf, qt, sft, xp, tso = ext.quant_act(x, f, row, col, kw["rpad"], B, N, npre, rht_row, rht_col, RHT_SIGN_MASK, sr_row, sr_col, u_row, u_col, seed, sr_offset, ts_mode, tsrc, mul, nxt,
                                            div_true, False, rounds, parts, srm)
    kt = qt.shape[1] * 2 if col else 0
    return FP4Q(fmt, R, C, kt, q if row else None, sf if row else None, qt if col else None, sft if col else None, tso[0:1] if f == 0 and row else None,
                tso[1:2] if f == 0 and col else None, xp if npre > 0 else None)


def quant_weight(w: torch.Tensor, fmt: str = "nv4", *, amax: str = "delayed", state: Optional[FP4State] = None, site: int = 0, ts=None, div_true: bool = False,
                 headroom: float = 2.0, row: Optional[bool] = None) -> FP4Q:
    """Weight ``w [N, K]`` (fp32 master or bf16, N and K multiples of 128): ``q [N, K/2]`` along K (fwd B operand) and ``qt [K, N/2]`` = W^T along N (dgrad B operand), 2D block scales."""
    ext = extension()
    row = WEIGHT_ROW if row is None else bool(row)  # row=False: only the W^T copy (qt, sft, ts) is produced (q / sf = None): the row copy is the forward operand, unused by an mx8 / bf16 forward
    N, K = w.shape
    f = FMT[fmt]
    tsrc, ts_mode, mul, nxt = None, 0, 1.0, None
    if f == 0:
        if amax == "ts":
            tsrc = torch.as_tensor(ts, dtype=torch.float32, device=w.device).reshape(-1)[:1].contiguous()
        elif amax == "exact":
            tsrc = torch.zeros(1, dtype=torch.float32, device=w.device)
            ext.amax_abs(w, tsrc)
            ts_mode = 1
        elif amax == "delayed":
            assert state is not None
            if not state.inited[site]:
                prev = state.amax[0, site]
                prev.zero_()
                ext.amax_abs(w, prev)
                state.inited[site] = True
            assert headroom > 0, headroom
            state.set_headroom(site, headroom)
            tsrc, ts_mode, mul, nxt = state.amax[0, site], 1, float(headroom), state.amax[1, site]
        else:
            raise ValueError(amax)
    q, sf, qt, sft, tso = ext.quant_weight(w, f, ts_mode, tsrc, mul, nxt, div_true, False, row)
    ts_t = tso[0:1] if f == 0 else None
    return FP4Q(fmt, N, K, N, q if row else None, sf if row else None, qt, sft, ts_t, ts_t)


def philox_uniform(n: int, seed: Optional[torch.Tensor], word1: int, device="cuda", rounds: Optional[int] = None) -> torch.Tensor:
    """The uniform stream the SR path uses: u[i], counter (i/4, word1, 0, 0), word1 = 2*sr_offset (row copy) or 2*sr_offset+1 (col copy)."""
    return extension().philox_uniform(n, seed, word1, torch.device(device), PHILOX_ROUNDS if rounds is None else int(rounds))
