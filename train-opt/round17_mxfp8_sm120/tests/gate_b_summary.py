"""Compact matrix of gate_b_multi.py results: per candidate x seed, global grad cos / loss_rel vs the bf16 reference (and vs R0), pass flag.

  python tests/gate_b_summary.py profile/gateB_*.json [--md]
"""
from __future__ import annotations

import json
import sys


def main():
    md = "--md" in sys.argv
    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
        d = json.load(open(path))
        seeds = sorted(d["seeds"])
        cands = list(d["seeds"][seeds[0]]["cands"])
        print(f"\n{d['model']} ({path.split('/')[-1]}): global grad cos vs TB1 [vs R0] / loss_rel vs TB1 / grad_norm_rel vs TB1")
        if md:
            print("| candidate | " + " | ".join(f"seed {s}" for s in seeds) + " | min gcos vs TB1 | pass |\n|---|" + "---|" * (len(seeds) + 2))
        for c in cands:
            cells, mn, ok = [], 1.0, True
            for s in seeds:
                x = d["seeds"][s]["cands"][c]
                a, b = x["vs_ref"], x["vs_r0"]
                cells.append(f"{a['grad_cos_global']:.4f} [{b['grad_cos_global']:.4f}] / {a['loss_rel']:.1e} / {a['grad_norm_rel']:.3f}")
                mn = min(mn, a["grad_cos_global"])
                ok &= x["pass"]
            if md:
                print(f"| {c} | " + " | ".join(cells) + f" | {mn:.4f} | {ok} |")
            else:
                print(f"{c:36s} " + " | ".join(cells) + f" | min {mn:.4f} pass={ok}")


if __name__ == "__main__":
    main()
