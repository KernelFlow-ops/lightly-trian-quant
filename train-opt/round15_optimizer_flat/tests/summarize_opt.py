"""Compact optimizer-benchmark table (round15): python summarize_opt.py"""
import glob
import json
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
print("model   variant        n_params   numel        eager_med  eager_min  graph_med  note")
for f in sorted(glob.glob(str(ROUND / "profile" / "opt_bench_*.json"))):
    r = json.load(open(f))["result"]
    e, g = r["eager_ms"], r["graph_replay_ms"]
    note = (r["graph_capture_error"] or "")[:70]
    km = r.get("kernels_ms")
    if km:
        note = f"K1 {km['sumsq_prep']:.3f} ms ({km['sumsq_GBs']:.0f} GB/s) K2 {km['update']:.3f} ms ({km['update_GBs']:.0f} GB/s) floats {km['total_floats']}"
    print(f"{r['model']:7s} {r['variant']:14s} {r['n_params']:8d} {r['numel']:11d}  {e['median']:8.2f}  {e['min']:8.2f}  {'-' if g is None else format(g['median'], '8.2f'):>8s}   {note}")
