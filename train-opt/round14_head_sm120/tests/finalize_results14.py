"""Merge measured gates / nsys data into round14 results.json (run.py bench already wrote gpu_step / breakdown / peak mem per model+variant).

  python tests/finalize_results14.py
Gate B: all `profile/gateB_*_r14*.json` files are pooled per (model, candidate) (a batch is a (file, seed) pair: batches differ between processes);
the gate-B-only MXFP8 head candidates (MXH_R14, MXH_R14_bwd, reference TB1_R13, no round17) get their own entries with null perf fields.
Gate S: `profile/gateS_*_r14*.json`.  nsys: `profile/nsys_step_<model>_<stack>.json` (tests/nsys_step_summary.py --json).
Everything not measured stays null.
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
from common.bench import empty_result  # noqa: E402

R = ROOT / "round14_head_sm120"
P = R / "profile"
GATE_B_ONLY = {"MXH_R14": "TB1+r13+r14:mx8 (ConvT fwd/dgrad/wgrad MXFP8), reference TB1_R13", "MXH_R14_bwd": "TB1+r13+r14:mx8_bwd (ConvT dgrad/wgrad MXFP8), reference TB1_R13"}


def main():
    res = json.loads((R / "results.json").read_text())
    pooled: dict = {}
    for f in sorted(glob.glob(str(P / "gateB_*_r14*.json"))):
        d = json.loads(Path(f).read_text())
        for seed, rep in d["seeds"].items():
            for cand, x in rep["cands"].items():
                if cand.startswith("CTRL_"):
                    continue
                pooled.setdefault((d["model"], cand), []).append((Path(f).name, seed, x))
    for (model, cand), items in pooled.items():
        xs = [x for _, _, x in items]
        ent = res.setdefault(model, {}).get(cand)
        if ent is None:
            if cand not in GATE_B_ONLY:
                continue
            ent = empty_result(round_name="round14_head_sm120", variant=cand, model=model, batch=7 if model == "vitl16" else 14, precision="bf16-mixed",
                               patches=[], desc=GATE_B_ONLY[cand] + " (gate B only; no step timing for this exact stack)")
            res[model][cand] = ent
        g_ref = [x["vs_ref"]["grad_cos_global"] for x in xs]
        ent["accuracy"]["level_b"] = {
            "criterion": xs[0]["criterion"], "n_batches": len(xs), "n_pass": sum(bool(x["pass"]) for x in xs), "pass": all(bool(x["pass"]) for x in xs),
            "loss_rel_max": max(x["vs_ref"]["loss_rel"] for x in xs),
            "grad_cos_global_min": min(g_ref), "grad_cos_global_median": statistics.median(g_ref),
            "grad_cos_global_min_vs_R0": min(x["vs_r0"]["grad_cos_global"] for x in xs),
            "grad_cos_min": min(x["vs_ref"]["grad_cos_min"] for x in xs),
            "grad_norm_rel_max": max(x["vs_ref"]["grad_norm_rel"] for x in xs),
            "update_cos_min": min(x["vs_ref"]["update_cos_min"] for x in xs),
            "per_param_offenders_vs_r0_per_batch": [len(x["per_param_offenders_vs_r0"]) for x in xs],
            "sources": sorted({n for n, _, _ in items}),
        }
    for f in sorted(glob.glob(str(P / "gateS_*_r14*.json"))):
        d = json.loads(Path(f).read_text())
        model = d["model"]
        for k, g in d["gates"].items():
            cand, rest = k.split("_vs_")
            ref, phase = rest.rsplit("_", 1)
            ent = res.get(model, {}).get(cand)
            if ent is None:
                continue
            lc = ent["accuracy"].setdefault("level_c", {})
            det = lc.setdefault("detail", {})
            det[f"vs_{ref}_{phase}"] = {"max_rel": g["curve300_max_rel"], "pass": g["pass"], "finite": g["finite"]}
            lc["steps"] = 50
            lc["pass"] = all(v["pass"] for v in det.values())
            lc["curve300_max_rel"] = max(v["max_rel"] for v in det.values())
    for f in sorted(glob.glob(str(P / "nsys_step_*.json"))):
        d = json.loads(Path(f).read_text())
        model, stack = Path(f).stem[len("nsys_step_"):].split("_", 1)
        ent = res.get(model, {}).get(stack)
        if ent is None:
            continue
        ent["kernels_per_step"] = d["kernels_per_step"]
        ent["syncs_per_step"] = d.get("syncs_per_step")
        ent["gpu_busy_ratio"] = d.get("gpu_busy_ratio")
        ent["nsys"] = {"kernel_ms_per_step": d["kernel_ms_per_step"], "categories": d["categories"], "nvtx_gpu_ms": d.get("nvtx_gpu_ms"),
                       "source": f"nsys/step_{model}_{stack} (mid phase, warmup 20, 5 captured steps)"}
    (R / "results.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    for model, vs in res.items():
        for v, e in vs.items():
            lb, lc = e["accuracy"].get("level_b", {}), e["accuracy"].get("level_c", {})
            print(f"{model:7s} {v:12s} w_img/s {e['gpu_step'].get('weighted_img_s')} | B {lb.get('n_pass')}/{lb.get('n_batches')} | S {lc.get('pass')} | kernels/step {e.get('kernels_per_step')} busy {e.get('gpu_busy_ratio')}")


if __name__ == "__main__":
    main()
