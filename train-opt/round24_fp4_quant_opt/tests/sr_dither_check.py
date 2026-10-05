"""Exhaustive check of the two in-kernel SR rules over ALL 65536 values of the 16 random bits h (no sampling noise):
  exact   : round up iff u < p, u = (2h+1)/2^17  (round21 / fp4_ref.e2m1_sr given that u)
  dither  : |v| >= 1 carry trick, |v| < 1 RN(|v| + (2h - 65535) 2^-18)
For a set of |v| (grid points, midpoints, +-1 ulp around them, random, tiny, subnormal, > 6): mean of the produced magnitude over h vs |v| (bias, in units of the cell spacing), the
variance vs the ideal p (1 - p) s^2 and P(up) vs p.   python tests/sr_dither_check.py  -> profile/sr_dither_check.json
"""
import json
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "tests"))
import torch

from bench_hot import load_mod
from common.bench import assert_gpu_exclusive, gpu_lock

GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def main():
    m = load_mod("srd", "")
    ext = m.extension()
    g = torch.Generator().manual_seed(1)
    vals = []
    for a, b in zip(GRID[:-1], GRID[1:]):
        vals += [a, b, (a + b) / 2, a + (b - a) * 0.1, a + (b - a) * 0.9, a + (b - a) / 3]
    vals += [6.0, 6.5, 100.0, 1e-3, 1e-6, 1e-30, 1e-40, 0.2499999, 0.25, 0.2500001, 0.75, 0.7499999, 0.7500001, 0.4999999, 0.5000001, 0.9999999, 1.0000001, 1.9999999, 2.0000001, 3.9999998, 4.0000005]
    import numpy as np

    vals_t = torch.tensor(vals, dtype=torch.float32)
    # next / previous float around every value
    ar = vals_t.numpy().view(np.int32)
    vals_t = torch.cat([vals_t, torch.from_numpy((ar + 1).view(np.float32)), torch.from_numpy((ar - 1).view(np.float32))])
    vals_t = torch.cat([vals_t, torch.rand(3000, generator=g) * 6.0, torch.rand(500, generator=g) * 0.5])
    vals_t = vals_t[(vals_t >= 0) & torch.isfinite(vals_t)]
    n = vals_t.numel()
    with gpu_lock():
        for _ in range(60):
            try:
                assert_gpu_exclusive()
                break
            except RuntimeError:
                import time

                time.sleep(20)
        v = vals_t.cuda().repeat_interleave(65536)
        h = torch.arange(65536, dtype=torch.int32, device="cuda").repeat(n)
        ce, cd = ext.sr_rules(v.contiguous(), h.contiguous())
        grid = torch.tensor(GRID[1:] and [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0], device="cuda")
        out = {}
        vv = vals_t.cuda().clamp(max=6.0).double()
        for name, c in (("exact", ce), ("dither", cd)):
            val = grid[(c.long() & 7)].view(n, 65536).double()
            mean = val.mean(1)
            # cell of each v
            gp = torch.tensor(GRID, device="cuda", dtype=torch.float64)
            idx = torch.bucketize(vv, gp, right=True).clamp(1, len(GRID) - 1)
            lo, hi = gp[idx - 1], gp[idx]
            sp = hi - lo
            pf = (vv - lo) / sp
            up = ((val > vv.unsqueeze(1)).double().mean(1))  # fraction of h that rounded above v
            var = ((val - vv.unsqueeze(1)) ** 2).mean(1)
            ideal_var = pf * (1 - pf) * sp * sp
            bias = (mean - vv) / sp
            # P(up) vs p only meaningful off the grid points
            offgrid = (pf > 1e-6) & (pf < 1 - 1e-6)
            out[name] = {"max_abs_bias_over_spacing": float(bias.abs().max()), "mean_abs_bias_over_spacing": float(bias.abs().mean()),
                         "max_abs_Pup_minus_p": float((up - pf)[offgrid].abs().max()) if offgrid.any() else None,
                         "max_abs_var_minus_ideal_over_spacing2": float(((var - ideal_var) / (sp * sp)).abs().max()),
                         "n_values": n, "worst_bias_value": float(vv[bias.abs().argmax()]), "worst_Pup_value": float(vv[offgrid][(up - pf)[offgrid].abs().argmax()]) if offgrid.any() else None}
        # grid points (and clamped > 6) must map to themselves
        exact_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
        vg = exact_grid.repeat_interleave(65536).contiguous()
        hg = torch.arange(65536, dtype=torch.int32, device="cuda").repeat(8)
        ceg, cdg = ext.sr_rules(vg, hg)
        for name, c in (("exact", ceg), ("dither", cdg)):
            val = grid[(c.long() & 7)].view(8, 65536)
            out[name]["grid_points_map_to_themselves_frac"] = float((val == exact_grid.unsqueeze(1)).double().mean())
        out["note"] = "bias / variance are over all 65536 values of h per |v| (exhaustive); spacing = width of the e2m1 cell containing |v| (clamped to 6); 2^-17 = 7.6e-6"
    p = ROUND / "profile" / "sr_dither_check.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))


main()
