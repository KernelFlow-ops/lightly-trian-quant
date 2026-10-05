"""Round20b FP4 simulation quantizers: ``fp4_ref`` semantics (bit-exact when ``amax="exact"``) plus (a) a dedicated SR generator that never touches the
torch default generators, (b) shared RHT signs, (c) the nv4 per-tensor amax policy ``exact | delayed`` (6a 12.3).

Everything is plain torch ops on device tensors: no host sync, no data-dependent Python branch (the delayed-amax state is selected with
``torch.where`` on a device flag), so the simulated step stays capturable (it is only run eagerly in R20b).

``Sim`` owns the per-process state: SR generator, RHT sign vector, delayed-amax table keyed by call site.  Call-site keys are plain strings
(``"b3.fc1.xrow"``, ``"h2.c1.dz.col"`` ...) and must be deterministic per step.

Semantics frozen in fp4_ref are reused verbatim through its private helpers (``_nv4_sf``, ``_codes_nv4``, ``_mx4_byte``, ``_codes_mx4``,
``rht_apply``, ``pack_nibbles``, ``to_blocked``); only the per-tensor scale source differs.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import fp4_ref as R

NV4_TS_DIV = 6.0 * 448.0
DIAG: dict = {}   # delayed-amax diagnostics (0-d device tensors), reset by Sim.reset; read with ``diag_snapshot()``
SITE: dict = {}   # call site -> max over calls of (current amax / previous amax)


def diag_snapshot(top: int = 64) -> dict:
    """One host sync: counts, the largest cur/prev amax ratio (delayed amax saturates iff ratio > 2) and the worst call sites."""
    out = {k: float(v) for k, v in DIAG.items()}
    if SITE:
        names = list(SITE)
        vals = torch.stack([SITE[n] for n in names]).tolist()
        out["worst_sites"] = sorted(([n, round(v, 4)] for n, v in zip(names, vals)), key=lambda kv: -kv[1])[:top]
        out["n_sites"] = len(names)
        out["n_sites_ratio_gt_2"] = sum(v > 2.0 for v in vals)
    return out


class Sim:
    def __init__(self, device="cuda", seed: int = 777, amax: str = "delayed"):
        self.device = device
        self.signs = R.rht_signs(device=device)
        self.reset(seed, amax)

    def reset(self, seed: int = 777, amax: str = "delayed") -> None:
        assert amax in ("exact", "delayed")
        self.amax = amax
        self.seed = seed
        self.gen = R.make_gen(self.device, seed)  # dedicated generator: the default torch generators are never used for SR
        self.state: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        DIAG.clear()
        SITE.clear()

    # ------------------------------------------------------------------ amax
    def amax_used(self, key: str, cur: torch.Tensor) -> torch.Tensor:
        """exact: the current amax.  delayed: 2 x amax of the previous call at ``key`` (first call: exact).  Always records the current amax."""
        if self.amax == "exact":
            return cur
        st = self.state.get(key)
        if st is None:
            st = (torch.zeros((), dtype=torch.float32, device=cur.device), torch.zeros((), dtype=torch.bool, device=cur.device))
            self.state[key] = st
        prev, init = st
        used = torch.where(init, prev * 2.0, cur)
        # diagnostics (device-side, no host sync): how often the current amax exceeds the 2x headroom (the values above it would saturate)
        zero = torch.zeros_like(cur)
        ratio = torch.where(init & (prev > 0), cur / prev.clamp_min(1e-30), zero)
        d = DIAG
        if not d:
            d.update(n_calls=zero.clone(), n_delayed=zero.clone(), n_over=zero.clone(), max_ratio=zero.clone())
        d["n_calls"] += 1.0
        d["n_delayed"] += init.float()
        d["n_over"] += (init & (cur > used)).float()
        d["max_ratio"] = torch.maximum(d["max_ratio"], ratio)
        SITE[key] = torch.maximum(SITE[key], ratio) if key in SITE else ratio
        prev.copy_(cur)
        init.fill_(True)
        return used

    def ts_of(self, key: str, xf: torch.Tensor) -> torch.Tensor:
        """nv4 per-tensor scale [1] fp32 (fp4_ref formula, amax source per policy)."""
        amax = self.amax_used(key, xf.abs().amax())
        return torch.where(amax > 0, amax / NV4_TS_DIV, torch.ones_like(amax)).reshape(1).float()


# ------------------------------------------------------------------------------------------------ matrices
def q_matrix(sim: Sim, key: str, x: torch.Tensor, fmt: str, *, rht: bool = False, sr: bool = False, u: Optional[torch.Tensor] = None):
    """Quantize ``x [R, C]`` along C.  Returns (q [R, C/2] u8, blocked sf u8, ts [1] fp32 | None).  == ``fp4_ref.quantize_matrix`` for amax='exact'."""
    xf = x.float().contiguous()
    Rr, C = xf.shape
    vec = R.VEC[fmt]
    assert C % vec == 0, (C, vec)
    if rht:
        xf = R.rht_apply(xf, sim.signs)
    if sr and u is None:
        u = torch.rand(Rr, C, generator=sim.gen, device=xf.device)
    if fmt == "nv4":
        ts = sim.ts_of(key, xf)
        sf8 = R._nv4_sf(xf.reshape(Rr, C // 16, 16).abs().amax(-1), ts)
        codes = R._codes_nv4(xf, sf8, ts, sr, u)
        return R.pack_nibbles(codes), R.to_blocked(sf8.view(torch.uint8)), ts
    byte = R._mx4_byte(xf.reshape(Rr, C // 32, 32).abs().amax(-1))
    codes = R._codes_mx4(xf, byte, sr, u)
    return R.pack_nibbles(codes), R.to_blocked(byte.to(torch.uint8)), None


def q_matrix_2d(sim: Sim, key: str, x: torch.Tensor, fmt: str):
    """2D (vec x vec tile) scale weight quantization; (q, sf, ts, qt, sft, tst) as ``fp4_ref.quantize_matrix_2d`` (amax source per policy)."""
    xf = x.float().contiguous()
    Rr, C = xf.shape
    v = R.VEC[fmt]
    assert Rr % v == 0 and C % v == 0
    amax_t = xf.reshape(Rr // v, v, C // v, v).abs().amax((1, 3))
    if fmt == "nv4":
        ts = sim.ts_of(key, xf)
        sf8 = R._nv4_sf(amax_t, ts)
        sfr = sf8.repeat_interleave(v, 0)
        sfc = sf8.t().contiguous().repeat_interleave(v, 0)
        codes = R._codes_nv4(xf, sfr, ts, False, None)
        return (R.pack_nibbles(codes), R.to_blocked(sfr.view(torch.uint8)), ts, R.pack_nibbles(codes.t().contiguous()), R.to_blocked(sfc.view(torch.uint8)), ts)
    byte = R._mx4_byte(amax_t)
    bytr = byte.repeat_interleave(v, 0)
    bytc = byte.t().contiguous().repeat_interleave(v, 0)
    codes = R._codes_mx4(xf, bytr, False, None)
    return (R.pack_nibbles(codes), R.to_blocked(bytr.to(torch.uint8)), None, R.pack_nibbles(codes.t().contiguous()), R.to_blocked(bytc.to(torch.uint8)), None)


# ------------------------------------------------------------------------------------------------ operands
def q_row(sim: Sim, key: str, x: torch.Tensor, fmt: str, *, rht: bool = False, sr: bool = False) -> R.FP4Tensor:
    """Row copy of an activation / gradient (blocks along C = the GEMM's K axis)."""
    t = R.FP4Tensor(fmt, x.shape[0], x.shape[1], R.pad128(x.shape[0]), rht="row" if rht else "", sr=sr)
    t.q, t.sf, t.ts = q_matrix(sim, key, x, fmt, rht=rht, sr=sr)
    return t


def q_col(sim: Sim, key: str, x: torch.Tensor, fmt: str, *, rht: bool = False, sr: bool = False, rpad: Optional[int] = None) -> R.FP4Tensor:
    """Transposed copy ``[C, Rpad/2]`` (blocks along the zero-padded token axis).  ``x [R, C]``."""
    Rr, C = x.shape
    rp = R.pad128(Rr) if rpad is None else rpad
    xt = torch.nn.functional.pad(x.float(), (0, 0, 0, rp - Rr)).t().contiguous()
    t = R.FP4Tensor(fmt, Rr, C, rp, rht="col" if rht else "", sr=sr)
    t.qt, t.sft, t.tst = q_matrix(sim, key, xt, fmt, rht=rht, sr=sr)
    return t


def q_weight(sim: Sim, key: str, w: torch.Tensor, fmt: str, *, two_d: bool = True, rht_row: bool = False, rht_col: bool = False,
             want_row: bool = True, want_col: bool = True) -> R.FP4Tensor:
    """Weight ``w [N, K]``: ``q`` = W along K (fwd B operand), ``qt`` = W^T [K, N] along N (dgrad B operand).  RN.  Mirrors ``fp4_ref.quantize_weight``."""
    N, K = w.shape
    wf = w.float()
    t = R.FP4Tensor(fmt, N, K, N, rht="+".join(n for n, f in (("row", rht_row), ("col", rht_col)) if f))
    if two_d and not (rht_row or rht_col):
        q, sf, ts, qt, sft, tst = q_matrix_2d(sim, key + ".2d", wf, fmt)
        if want_row:
            t.q, t.sf, t.ts = q, sf, ts
        if want_col:
            t.qt, t.sft, t.tst = qt, sft, tst
        return t
    if want_row:
        a = R.rht_apply(wf, sim.signs) if rht_row else wf
        if two_d:
            t.q, t.sf, t.ts = q_matrix_2d(sim, key + ".r2d", a, fmt)[:3]
        else:
            t.q, t.sf, t.ts = q_matrix(sim, key + ".r1d", a, fmt)
    if want_col:
        bt = wf.t().contiguous()
        if rht_col:
            bt = R.rht_apply(bt, sim.signs)
        if two_d:
            t.qt, t.sft, t.tst = q_matrix_2d(sim, key + ".c2d", bt, fmt)[:3]
        else:
            t.qt, t.sft, t.tst = q_matrix(sim, key + ".c1d", bt, fmt)
    return t
