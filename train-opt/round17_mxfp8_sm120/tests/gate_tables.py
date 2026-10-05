"""Render gate B / gate S result JSONs (gate_b_multi.py / gate_s_multi.py) as markdown tables.

  python tests/gate_tables.py B <gateB.json> [cand]   |   python tests/gate_tables.py S <gateS.json>
"""
from __future__ import annotations

import json
import sys


def table_b(path, cand=None):
    d = json.load(open(path))
    seeds = d["seeds"]
    cands = list(next(iter(seeds.values()))["cands"])
    out = []
    for c in cands if cand is None else [cand]:
        kind = next(iter(seeds.values()))["cands"][c]["kind"]
        out.append(f"\n**{d['model']} {c}** ({kind}); per seed, vs R0 (fp32) and vs the bf16 reference\n")
        out.append("| seed | loss_rel vs R0 | grad cos (global) vs R0 | grad_norm_rel vs R0 | loss_rel vs ref | grad cos (global) vs ref | grad_norm_rel vs ref | update cos (global) vs ref | per-param cos min vs ref | per-param offenders (calibrated) | pass |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for s, r in seeds.items():
            x = r["cands"][c]
            a, b = x["vs_r0"], x["vs_ref"]
            off = x["per_param_offenders_vs_r0"]
            offs = ", ".join(f"{k.replace('backbone.', '')}: cos(cand,R0) {v[1]:.3f} vs cos(TB1,R0) {v[0]:.3f}, cos(cand,TB1) {x['per_param_cos_vs_ref'][k]:.4f}" for k, v in off.items()) or "0"
            out.append(f"| {s} | {a['loss_rel']:.1e} | {a['grad_cos_global']:.5f} | {a['grad_norm_rel']:.4f} | {b['loss_rel']:.1e} | {b['grad_cos_global']:.5f} | {b['grad_norm_rel']:.4f} | {b['update_cos_global']:.5f} | {b['grad_cos_min']:.4f} | {offs} | {x['pass']} |")
        refrow = [f"{s}: loss_rel {r['ref_vs_r0']['loss_rel']:.1e}, gcos {r['ref_vs_r0']['grad_cos_global']:.5f}, pmin {r['ref_vs_r0']['grad_cos_min']:.3f}" for s, r in seeds.items()]
        out.append("\nTB1 vs R0 (bf16 self-noise): " + "; ".join(refrow))
    return "\n".join(out)


def table_s(path):
    d = json.load(open(path))
    out = [f"\n**{d['model']}**: 50 steps, 2 batches, gate_c(interval=10, max_rel=0.05); max relative EMA-loss deviation over checkpoints 10..50, grad-norm ratio range\n",
           "| pair | schedule | fixed mid | fixed late |", "|---|---|---|---|"]
    pairs = sorted({k.rsplit("_", 1)[0] for k in d["gates"]})
    for p in pairs:
        cells = []
        for ph in ("schedule", "mid", "late"):
            g = d["gates"].get(f"{p}_{ph}")
            if g is None:
                cells.append("-")
                continue
            gn = [v for v in g["grad_norm_ratios"].values() if v is not None]
            cells.append(f"{g['curve300_max_rel']:.4f} / [{min(gn):.3f},{max(gn):.3f}] / {'PASS' if g['pass'] else 'FAIL'}")
        out.append(f"| {p.replace('_vs_', ' vs ')} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    if sys.argv[1] == "B":
        print(table_b(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
    else:
        print(table_s(sys.argv[2]))
