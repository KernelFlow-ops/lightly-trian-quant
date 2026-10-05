"""Optional variant fp4q2d (sr_mode="dither"): (1) sr_mode="exact" of fp4q2d == fp4q2 == round21 fp4q bit for bit; (2) dither: scale bytes / ts / RN unchanged, only the SR decisions differ;
(3) in-kernel unbiasedness (mean of the dequantised SR value over 256 seeds vs x, normalised by the expected noise; same statistic as test_fp4q2.py::part_sr) for exact and dither, row / col copy,
and the fraction of elements rounding up vs the ideal p; (4) Philox-7 + dither.     python tests/test_q2d.py  -> profile/test_q2d.json
(The exhaustive 2^16-values-of-h check of the two decision rules is tests/sr_dither_check.py.)"""
import json
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
sys.path.insert(0, str(ROUND.parent / "round20_fp4_probe" / "impl"))
import torch

import fp4_ref as R
import fp4q2
import fp4q2d
from common.bench import assert_gpu_exclusive, gpu_lock
from round21_fp4_quant_head.impl import fp4q as A

dev = "cuda"


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    va = a.view(torch.int32) if a.dtype == torch.float32 else a.view(torch.uint8)
    vb = b.view(torch.int32) if b.dtype == torch.float32 else b.view(torch.uint8)
    return a.shape == b.shape and torch.equal(va, vb)


def main():
    out = {"fails": []}
    t0 = time.time()
    with gpu_lock():
        for _ in range(60):
            try:
                assert_gpu_exclusive()
                break
            except RuntimeError:
                time.sleep(20)
        seed = torch.tensor([424242], dtype=torch.int64, device=dev)
        g = torch.Generator(device=dev).manual_seed(3)
        # (1) exact mode == fp4q2 == r21a, (2) dither differs only in the SR decisions
        n_cmp = 0
        for (Rr, C) in ((7203, 1024), (1029, 768), (257, 192), (14406, 3072)):
            x = (torch.randn(Rr, C, generator=g, device=dev) * 3).bfloat16()
            x[::97] *= 8
            for (row, col) in ((True, True), (True, False), (False, True)):
                kw = dict(row=row, col=col, sr_row=True, sr_col=True, amax="exact", seed=seed, sr_offset=5)
                a = A.quant_act(x, "nv4", **kw)
                b = fp4q2.quant_act(x, "nv4", **kw)
                c = fp4q2d.quant_act(x, "nv4", sr_mode="exact", **kw)
                d = fp4q2d.quant_act(x, "nv4", sr_mode="dither", **kw)
                for f in ("q", "sf", "qt", "sft", "ts", "tst"):
                    if not (same(getattr(a, f), getattr(b, f)) and same(getattr(a, f), getattr(c, f))):
                        out["fails"].append(f"exact-mode differs {Rr}x{C} row={row} col={col} {f}")
                    if f in ("sf", "sft", "ts", "tst") and not same(getattr(a, f), getattr(d, f)):
                        out["fails"].append(f"dither changed scales {Rr}x{C} {f}")
                rn = fp4q2d.quant_act(x, "nv4", row=row, col=col, amax="exact")
                rn2 = fp4q2d.quant_act(x, "nv4", row=row, col=col, amax="exact", sr_mode="dither")
                for f in ("q", "qt", "sf", "sft"):
                    if not same(getattr(rn, f), getattr(rn2, f)):
                        out["fails"].append(f"sr_mode changed the RN output {f}")
                n_cmp += 1
        out["cases_bitexact"] = n_cmp
        x = (torch.randn(2048, 1024, generator=g, device=dev) * 3).bfloat16()
        a = A.quant_act(x, "nv4", row=True, col=False, sr_row=True, amax="exact", seed=seed, sr_offset=5)
        d = fp4q2d.quant_act(x, "nv4", row=True, col=False, sr_row=True, amax="exact", seed=seed, sr_offset=5, sr_mode="dither")
        out["q_bytes_differ_exact_vs_dither_frac"] = float((a.q != d.q).float().mean())
        # (3) unbiasedness
        stats = {}
        S = 256
        for mode, rounds in (("exact", 10), ("dither", 10), ("dither", 7)):
            for copy in ("row", "col"):
                Rr, C = 128, 1024
                gg = torch.Generator(device=dev).manual_seed(11)
                xg = torch.randn(Rr, C, generator=gg, device=dev)
                xg[:, ::97] *= 8.0
                x_ = xg.bfloat16()
                rr0 = R.quantize(x_, "nv4", row=True, col=False)
                rc0 = R.quantize(x_, "nv4", row=False, col=True)
                ts = torch.cat([rr0.ts.reshape(1), rc0.tst.reshape(1)])
                acc = None
                upfrac = 0.0
                for sidx in range(S):
                    sd = torch.tensor([1000 + sidx], dtype=torch.int64, device=dev)
                    k = fp4q2d.quant_act(x_, "nv4", row=(copy == "row"), col=(copy == "col"), sr_row=True, sr_col=True, amax="ts", ts=ts, seed=sd, sr_offset=1, sr_mode=mode, philox_rounds=rounds)
                    dq = R.dequant_copy(k.q, k.sf, k.ts, "nv4", Rr, C) if copy == "row" else R.dequant_copy(k.qt, k.sft, k.tst, "nv4", C, k.kt)[:, :Rr]
                    acc = dq.double() if acc is None else acc + dq.double()
                mean = (acc / S).float()
                xe = x_.float() if copy == "row" else x_.float().t()
                base = rr0 if copy == "row" else rc0
                sfm = R.from_blocked(base.sf if copy == "row" else base.sft, Rr if copy == "row" else C, (C if copy == "row" else base.qt.shape[1] * 2) // 16)
                sv = sfm.view(torch.float8_e4m3fn).float()
                scale = sv.repeat_interleave(16, 1)[:, :xe.shape[1]] * (rr0.ts if copy == "row" else rc0.tst)
                v = (xe / scale.clamp_min(1e-38)).abs()
                ok = (v <= 6.0) & (scale > 0)
                grid = R._grid(dev)
                a_ = v.clamp(max=6.0)
                ix = (torch.bucketize(a_, grid, right=True) - 1).clamp(0, 6)
                lo, hi = grid[ix], grid[ix + 1]
                p = (a_ - lo) / (hi - lo)
                var = ((hi - lo) ** 2 * p * (1 - p)) * scale ** 2
                exp_rms = float(torch.sqrt(var[ok].mean() / S))
                bias_rms = float(torch.sqrt(((mean - xe)[ok] ** 2).mean()))
                stats[f"{mode}_r{rounds}_{copy}"] = {"bias_rms_over_expected_noise": round(bias_rms / exp_rms, 4), "seeds": S, "saturated_frac_excluded": round(float((~ok).float().mean()), 4)}
                if not (0.9 < bias_rms / exp_rms < 1.12):
                    out["fails"].append(f"unbiasedness {mode} r{rounds} {copy}: {stats[f'{mode}_r{rounds}_{copy}']}")
        out["unbiasedness"] = stats
    out["fail"] = len(out["fails"])
    out["seconds"] = round(time.time() - t0, 1)
    (ROUND / "profile" / "test_q2d.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


main()
