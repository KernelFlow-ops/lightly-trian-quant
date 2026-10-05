"""Assemble results.json of R21a from profile/gateA_fp4q.json, profile/bench_fp4q.json, profile/per_step_quant_cost.json and the ptxas / cuobjdump static analysis."""
import json
import re
import statistics
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
P = ROUND / "profile"


def load(n):
    f = P / n
    return json.loads(f.read_text()) if f.exists() else None


gate = load("gateA_fp4q.json")
bench = load("bench_fp4q.json")
step = load("per_step_quant_cost.json")
static = load("static_summary.json")
head_bench = load("bench_headroom.json")
base_hash = load("headroom_baseline_hashes.json")
res = {"round": "round21_fp4_quant_head", "part": "R21a"}
if gate:
    res["gate_A"] = {"file": "profile/gateA_fp4q.json", "all_pass": gate["all_pass"],
                     "parts": {k: {"cases": (len(v["cases"]) if isinstance(v.get("cases"), list) else v.get("cases")), "fail": v.get("fail")} for k, v in gate["parts"].items()}}
if bench:
    summ = {}
    for model, shapes in bench["act"].items():
        for name, r in shapes.items():
            summ[f"{model}/{name}"] = {m: {"us": v["us"], "x_floor": v.get("x_floor")} for m, v in r["cold_us"].items() if m != "torch_x.t().contiguous()"}
            summ[f"{model}/{name}"]["warm_us"] = r["warm_us"]
            summ[f"{model}/{name}"]["torch_x.t().contiguous()_us"] = r["cold_us"]["torch_x.t().contiguous()"]["us"]
    dual_x = [r["cold_us"]["nv4_dual"]["x_floor"] for s in bench["act"].values() for r in s.values()]
    big = [r["cold_us"]["nv4_dual"]["x_floor"] for s in bench["act"].values() for r in s.values() if r["R"] * r["C"] >= 2.5e7]
    res["bench"] = {"file": "profile/bench_fp4q.json", "hbm_copy_GBps": bench["hbm_copy_GBps"], "act_cold_us": summ, "weight": bench.get("weight"),
                    "nv4_dual_x_floor_median_all": round(statistics.median(dual_x), 3), "nv4_dual_x_floor_max_all": max(dual_x),
                    "nv4_dual_x_floor_median_large(R*C>=2.5e7)": round(statistics.median(big), 3)}
res["per_step_quant_cost"] = step
if gate and "headroom" in gate["parts"]:
    h = gate["parts"]["headroom"]
    res["headroom_followup"] = {"status": "complete (READY_headroom written)", "interface": "quant_act(..., headroom=2.0) / quant_weight(..., headroom=2.0) (delayed mode); FP4State.roll() = roll + saturation monitor; FP4State.stats(); FP4State.set_headroom(site, h); FP4State.reset_stats()",
                                "gate_A_headroom_part": {"cases": h["cases"], "fail": h["fail"], "identical_to_prechange_default": h["identical_to_prechange_default"],
                                                         "identical_to_prechange_headroom=2.0": h["identical_to_prechange_headroom=2.0"], "graph": h["graph"], "headroom_per_site_recorded": h["headroom_per_site_recorded"]},
                                "bench": head_bench, "baseline_hash_table": "profile/headroom_baseline_hashes.json (%d outputs of the pre-change build)" % (len(base_hash["hashes"]) if base_hash else 0)}
res["static"] = static
res["not_measured"] = ["head / backbone integration (R21b, R22)", "B gate", "S gate", "graph_equiv / export / resume (no params or training state touched)", "step bench", "nsys"]
(ROUND / "results.json").write_text(json.dumps(res, indent=1))
print("results.json written", list(res))
