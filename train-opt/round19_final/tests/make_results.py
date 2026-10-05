"""Build round19_final/results.json (schema of bench.empty_result, null for anything unmeasured) and summary.md (the full waterfall) from profile/*.json.
  python tests/make_results.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
from common import bench  # noqa: E402

PROF = ROUND / "profile"
MODELS = ("vitl16", "vitb16")
LAB = {"vitl16": "L", "vitb16": "B"}


def load(name):
    p = PROF / name
    return json.loads(p.read_text()) if p.exists() else None


def bench_json(model, stack, runner):
    return load(f"bench_{model}_{stack}_{runner}_all.json")


def fill_bench(model, stack, runner, desc):
    d = bench_json(model, stack, runner)
    if d is None:
        return None
    m = d["measurement"]
    cmd = d["command"]
    r = bench.empty_result(round_name="round19_final", variant=f"{stack}|{runner}", model=model, batch=int(d["summary"].get("batch", 7 if model == "vitl16" else 14)),
                           precision="bf16-mixed", patches=[], desc=desc)
    for ph in ("early", "mid", "late"):
        r["gpu_step"][ph] = {"step_ms": m[ph]["step_ms"], "img_s": m[ph]["img_s"]}
    r["gpu_step"]["weighted_img_s"] = m["weighted_img_s"]
    r["peak_mem_gb"] = max(m[p]["peak_allocated_gb"] for p in ("early", "mid", "late"))
    r["peak_reserved_gb"] = max(m[p]["peak_reserved_gb"] for p in ("early", "mid", "late"))
    if runner == "graph":      # peak allocated does not contain the graph's private pool (round16 NOTES 5.5): record the reserved memory instead
        r["peak_mem_gb"] = r["peak_reserved_gb"]
    r["breakdown_ms"].update({k: v for k, v in (m["mid"].get("breakdown_ms") or {}).items()})
    r["env"] = d["summary"].get("env") or {"gpu_snapshot": m.get("gpu_snapshot_before")}
    r["command"] = cmd
    r["graph"] = d["summary"].get("graph")
    return r


def main():
    results: dict = {}
    descs = {("R0", "eager"): "original Lightly step (fp32/TF32), same harness", ("FINAL", "eager"): "P2+r14+r15, eager (no graph)",
             ("FINAL", "graph"): "FINAL = TB1 BF16 + r13 + r17 MXFP8 + r14 head + r15 flat opt, whole-step CUDA graph",
             ("P2R15", "graph"): "previous best (P2+r15, graph), same-session drift check", ("TB1", "eager"): "TB1/P1 torch BF16 reference, eager, same-session",
             ("FINAL_fc", "graph"): "FINAL with r17 policy fc (safe MXFP8 tier), graph", ("FINAL_noMX", "graph"): "FINAL without MXFP8 (r13 bf16 block + r14 + r15), graph"}
    for model in MODELS:
        results[model] = {}
        for (stack, runner), desc in descs.items():
            r = fill_bench(model, stack, runner, desc)
            if r is not None:
                results[model][f"{stack}|{runner}"] = r
        fin = results[model].get("FINAL|graph")
        if fin is None:
            continue
        gb = load(f"gateB_{model}_FINAL.json")
        if gb and "summary" in gb:
            s = gb["summary"]["FINAL"]
            fin["accuracy"]["level_b"] = {"loss_rel": s["vs_r0"]["loss_rel_max"], "grad_cos_min": s["vs_r0"]["grad_cos_min"], "grad_norm_rel": s["vs_r0"]["grad_norm_rel_max"], "update_cos_min": None,
                                          "pass": s["pass"], "grad_cos_median": s["vs_r0"]["grad_cos_median"], "n_batches": s["n_batches"],
                                          "vs_p1": {k: s["vs_ref"][k] for k in ("grad_cos_median", "grad_cos_min", "loss_rel_max", "pass")}}
        gs = load(f"gateS_{model}_FINAL_graph_vs_R0.json")
        if gs:
            runs = gs["runs"]
            fin["accuracy"]["level_c"] = {"curve300_max_rel": max(v["gate"]["curve300_max_rel"] for v in runs.values()), "pass": all(v["gate"]["pass"] for v in runs.values()),
                                          "per_scenario": {k: v["gate"]["curve300_max_rel"] for k, v in runs.items()}}
        e2e = load(f"e2e_{model}_scan.json")
        if e2e:
            best = max(e2e["runs"], key=lambda r: r.get("img_s_steady", 0))
            fin["e2e"] = {"img_s": best.get("img_s_steady"), "dataload_wait_ms": best.get("wait_ms_mean"), "setting": best["setting"], "img_s_incl_startup": best.get("img_s_total")}
        nr = load("nsys_report.json")
        if nr and LAB[model] in nr:
            x = nr[LAB[model]]
            fin["kernels_per_step"] = x["kernels_per_step"]
            fin["gpu_busy_ratio"] = x["busy"]["busy_ratio"]
            fin["syncs_per_step"] = (x.get("sync_calls_total", 1) - 1) / 5 if "sync_calls_total" in x else None
        if model == "vitl16":
            best_b = None
            for p in sorted(PROF.glob("maxbatch_vitl16_FINAL_graph_B*.json")):
                r = json.loads(p.read_text())["result"]
                if r.get("ok"):
                    best_b = max(best_b or 0, r["batch"])
            fin["max_batch"] = best_b
    bench.write_results(ROUND / "results.json", results)
    print("wrote results.json:", {m: list(v) for m, v in results.items()})


if __name__ == "__main__":
    main()
