"""Compact tables for NOTES: bench jsons + nsys analysis jsons of round16 (and round15 opt bench).  python summarize.py [--write-results]"""
import glob
import json
import statistics
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
TMP = Path("/root/autodl-tmp/train-opt-tmp") / ROUND.name
sys.path.insert(0, str(ROUND.parent))


def bench_rows():
    rows = []
    for f in sorted(glob.glob(str(ROUND / "profile" / "bench_*.json"))):
        d = json.load(open(f))["summary"]
        rows.append((d["model"], d["name"], d["weighted_img_s"], d["step_ms_mean"], d.get("graph"), Path(f).name))
    return rows


def nsys_rows():
    rows = []
    for f in sorted(glob.glob(str(TMP / "nsys" / "*_analysis.json"))):
        d = json.load(open(f))
        s = d.get("sqlite", {})
        per = s.get("per_step", [])
        wall = statistics.fmean(p["wall_ms"] for p in per) if per else None
        rows.append((Path(f).name.replace("_analysis.json", ""), d["kernel_ms_per_step"], d["kernels_per_step"], d["gpu_busy_ratio"],
                     s.get("gap_count_per_step"), s.get("gap_ms_per_step"), wall, s.get("gap_histogram")))
    return rows


if __name__ == "__main__":
    print("== bench (weighted img/s; mean step ms early/mid/late)")
    for m, n, w, ms, g, f in bench_rows():
        extra = ""
        if g:
            extra = f" captures={g['captures']} capture_s={[round(x, 2) for x in g['capture_s']]} warm_s={g['warm_s']:.1f} metrics={g.get('metrics')}"
        print(f"{m:7s} {n:34s} W={(w if w is not None else 0.0):7.2f}  " + "/".join(f"{ms[p]:6.1f}" for p in ("early", "mid", "late") if p in ms) + extra)
    print("== nsys (ViT mid, 5 steps)")
    for name, k, n, busy, gc, gm, wall, hist in nsys_rows():
        print(f"{name:52s} kernel {k:7.2f} ms  {n:6.0f} kernels  busy {busy:.3f}  gaps>20us {gc} / {gm:.2f} ms  nsys wall {wall:.1f} ms  hist {hist}")
