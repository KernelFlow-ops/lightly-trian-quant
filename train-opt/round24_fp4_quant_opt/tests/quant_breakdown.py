"""Per-step FP4-quantizer time breakdown by call site / shape from an nsys GPU trace (CPU only).
  python tests/quant_breakdown.py <trace.csv> <label> [--steps 5] [--json out.json]
Kernel name -> site: act_kernel<T,0,ROW,COL> grid (C/64, ceil(rows/64)); w_kernel grid (K/64, N/64); amax_roll*."""
import argparse, collections, csv, json, re
ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("label"); ap.add_argument("--steps", type=int, default=5); ap.add_argument("--json")
a = ap.parse_args()
rows = collections.defaultdict(lambda: [0, 0.0])
other = collections.defaultdict(lambda: [0, 0.0])
with open(a.trace) as f:
    for d in csv.DictReader(f):
        nm = d["Name"]
        m = re.match(r"void act_kernel<[^,]+, \(int\)(\d), \(bool\)(\d), \(bool\)(\d)>", nm)
        dur = float(d["Duration (ns)"]) / 1e3
        m2 = re.match(r"void actf_kernel<\(bool\)(\d), \(bool\)(\d), \(bool\)(\d), \(int\)(\d+)>", nm)
        if m:
            fmt, r, c = map(int, m.groups())
            kind = {(1, 1): "dual", (1, 0): "row", (0, 1): "col"}[(r, c)]
            key = (f"act_{kind}", int(d["GrdX"]) * 64, int(d["GrdY"]))
        elif m2:
            r, c, sr, nr = map(int, m2.groups())
            kind = {(1, 1): "dual", (1, 0): "row", (0, 1): "col"}[(r, c)]
            key = (f"actf_{kind}{'_sr' if sr else ''}", int(d["GrdX"]) * 64, int(d["GrdY"]))
        elif nm.startswith("void w_kernel"):
            key = ("w_kernel", int(d["GrdX"]) * 64, int(d["GrdY"]) * 64)  # (K, N)
        elif "amax_roll" in nm or "amax_abs" in nm:
            key = (nm.split("(")[0].replace("void ", "")[:30], 0, 0)
        else:
            continue
        rows[key][0] += 1; rows[key][1] += dur
out = []
tot = 0.0
print(f"== {a.label} (per step; {a.steps} steps in window)")
for k, (n, us) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
    out.append({"kernel": k[0], "C_or_K": k[1], "tiles_or_N": k[2], "calls_per_step": n / a.steps, "avg_us": round(us / n, 2), "ms_per_step": round(us / a.steps / 1e3, 3)})
    tot += us / a.steps / 1e3
    print(f"  {k[0]:12s} C/K={k[1]:5d} y={k[2]:5d} calls/step={n / a.steps:5.1f} avg={us / n:7.2f} us  -> {us / a.steps / 1e3:6.3f} ms/step")
print(f"  total FP4 quantizer {tot:.3f} ms/step")
if a.json:
    json.dump({"label": a.label, "rows": out, "total_ms_per_step": round(tot, 3)}, open(a.json, "w"), indent=1)
