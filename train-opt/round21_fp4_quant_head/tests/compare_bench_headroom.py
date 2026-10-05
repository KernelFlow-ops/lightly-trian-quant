"""Post-change full bench (profile/bench_fp4q_headroom.json) vs the pre-change full bench (profile/bench_fp4q.json): per-mode ratio over all 28 shapes -> merged into profile/bench_headroom.json["vs_prechange"]."""
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "profile"
pre = json.loads((P / "bench_fp4q.json").read_text())
post = json.loads((P / "bench_fp4q_headroom.json").read_text())
res = {}
for mode in ("nv4_row", "nv4_col", "nv4_dual", "nv4_dual_rhtcol", "nv4_dual_sr", "nv4_dual_rhtcol_sr", "nv4_full_prefix_rhtcol_sr", "mx4_dual", "r17_mx8_dual"):
    rs = []
    for m in pre["act"]:
        for name, r in pre["act"][m].items():
            a, b = r["cold_us"].get(mode), post["act"][m][name]["cold_us"].get(mode)
            if a and b and "us" in a and "us" in b:
                rs.append(b["us"] / a["us"])
    res[mode] = {"n": len(rs), "post_over_pre_median": round(statistics.median(rs), 4), "min": round(min(rs), 4), "max": round(max(rs), 4)}
    print(mode, res[mode])
f = P / "bench_headroom.json"
d = json.loads(f.read_text())
d["vs_prechange"] = {"note": "profile/bench_fp4q_headroom.json (after) / profile/bench_fp4q.json (before the change, earlier session); r17_mx8_dual is an untouched control", "modes": res}
f.write_text(json.dumps(d, indent=1))
