"""Merge measured gates / nsys data into round17 results.json (bench.py already wrote gpu_step / breakdown / peak mem per model+variant).

  python tests/finalize_results.py
Writes accuracy.level_b / level_c, kernels_per_step, gpu_busy_ratio, nsys category table, quantization overhead per variant that has data.
Everything not measured stays null.  Also prints a compact policy table (gate B pass counts) for NOTES.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

R = Path("/root/lightly-train/train-opt/round17_mxfp8_sm120")
P = R / "profile"
NS = Path("/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/nsys")


def variant_key(c: str) -> str:
    return c  # run.py names variants by the stack token (MX8_all, MX8_fc, TB1_R13 ...)


def gate_b_stats():
    """{model: {cand: {"n", "pass_n", "gcos": [...], "loss_rel_ref": [...], "loss_rel_r0": [...], "gn_rel": [...], "update_cos": [...]}}} over all gateB_<model>_mx8*.json."""
    out = {}
    for f in sorted(glob.glob(str(P / "gateB_*_mx8*.json"))):
        d = json.loads(Path(f).read_text())
        m = d["model"]
        for s, r in d["seeds"].items():
            for c, x in r["cands"].items():
                e = out.setdefault(m, {}).setdefault(c, {"n": 0, "pass_n": 0, "gcos": [], "gcos_r0": [], "loss_rel_ref": [], "loss_rel_r0": [], "gn_rel": [], "update_cos": [], "files": set()})
                e["n"] += 1
                if x["kind"] == "prec":
                    ok = x["vs_r0"]["loss_rel"] <= 2e-2 and x["vs_r0"]["grad_cos_global"] >= 0.99 and x["vs_ref"]["loss_rel"] <= 2e-2 and x["vs_ref"]["grad_cos_global"] >= 0.99
                else:
                    ok = x["pass"]
                e["pass_n"] += int(ok)
                e["gcos"].append(x["vs_ref"]["grad_cos_global"])
                e["gcos_r0"].append(x["vs_r0"]["grad_cos_global"])
                e["loss_rel_ref"].append(x["vs_ref"]["loss_rel"])
                e["loss_rel_r0"].append(x["vs_r0"]["loss_rel"])
                e["gn_rel"].append(x["vs_ref"]["grad_norm_rel"])
                e["update_cos"].append(x["vs_ref"]["update_cos_global"])
                e["files"].add(Path(f).name)
    return out


def gate_s():
    out = {}
    for f in sorted(g for g in glob.glob(str(P / "gateS_*_mx8*.json")) if "INVALID" not in g):
        d = json.loads(Path(f).read_text())
        for k, g in d["gates"].items():
            cand, rest = k.split("_vs_")
            ref, phase = rest.rsplit("_", 1)
            out.setdefault(d["model"], {}).setdefault(cand, {})[f"vs_{ref}_{phase}"] = {"max_rel": g["curve300_max_rel"], "pass": g["pass"], "finite": g["finite"]}
    return out


def main():
    res = json.loads((R / "results.json").read_text()) if (R / "results.json").exists() else {}
    gb, gs = gate_b_stats(), gate_s()
    table = []
    for model, cands in gb.items():
        for c, e in cands.items():
            table.append((model, c, e["n"], e["pass_n"], min(e["gcos"]), sum(e["gcos"]) / len(e["gcos"]), max(e["loss_rel_ref"]), max(e["gn_rel"])))
            ent = res.setdefault(model, {}).get(c)
            if ent is None or not c.startswith("MX8_"):
                continue
            ent["accuracy"]["level_b"] = {
                "criterion": "precision change (Amendment 1 9.4): loss_rel<=2e-2 and global grad cos>=0.99 vs R0 and vs TB1; per-param recorded",
                "n_batches": e["n"], "n_pass": e["pass_n"], "pass": e["pass_n"] == e["n"],
                "loss_rel": max(max(e["loss_rel_ref"]), max(e["loss_rel_r0"])), "grad_cos_global_min_vs_TB1": min(e["gcos"]), "grad_cos_global_min_vs_R0": min(e["gcos_r0"]),
                "grad_norm_rel": max(e["gn_rel"]), "update_cos_min": min(e["update_cos"]), "grad_cos_min": None,
            }
    for model, cands in gs.items():
        for c, ph in cands.items():
            ent = res.get(model, {}).get(c)
            if ent is None or not c.startswith("MX8_"):
                continue
            ent["accuracy"]["level_c"] = {"steps": 50, "pass": all(v["pass"] for v in ph.values()), "curve300_max_rel": max(v["max_rel"] for v in ph.values()), "detail": ph}
    for f in sorted(glob.glob(str(P / "nsys_cat_step_*_v2.json"))):
        d = json.loads(Path(f).read_text())
        name = Path(f).stem.replace("nsys_cat_step_", "").replace("_v2", "")
        model, _, stack = name.partition("_")
        import re
        ent = next((v for k, v in res.get(model, {}).items() if re.sub(r"[^A-Za-z0-9_]", "_", k) == stack), None)
        ana = NS / f"step_{name}_v2_analysis.json"
        a = json.loads(ana.read_text()) if ana.exists() else {}
        if ent is not None:
            ent["kernels_per_step"] = d["kernels_per_step"]
            ent["gpu_busy_ratio"] = a.get("gpu_busy_ratio")
            ent["syncs_per_step"] = a.get("syncs_per_step")
            ent["nsys_categories_ms_per_step"] = {k: round(v["ms_per_step"], 2) for k, v in d["categories"].items()}
            ent["nsys_kernel_ms_per_step"] = d["kernel_ms_per_step"]
            ent["quant_ms_per_step"] = d["categories"].get("quantize MX8 (own CUDA)", {}).get("ms_per_step")
    for model, v in res.items():
        for k, ent in v.items():
            if k.startswith("MX8_"):
                ent["policy"] = k[4:]
                ent["stack"] = "TB1_STACK,r13:full,r17:" + k[4:]
                ent["quant_ms_per_step"] = ent.get("quant_ms_per_step")
    (R / "results.json").write_text(json.dumps(res, indent=1))
    print("model cand n pass gcos_min gcos_mean loss_rel_max gn_rel_max")
    for t in sorted(table):
        print("%s %-34s n=%2d pass=%2d  gcos min %.4f mean %.4f  loss_rel max %.1e  gn_rel max %.3f" % t)


if __name__ == "__main__":
    main()
