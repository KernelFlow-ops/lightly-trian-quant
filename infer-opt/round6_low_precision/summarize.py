"""Fold sens_*.jsonl / greedy_*.json / bench results into results.json (format of previous rounds) and print tables."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from bench import write_results
R = HERE / "results.json"
res = json.loads(R.read_text())
sens = {}
for m in ("vitl16", "vitb16"):
    d = {}
    for l in open(HERE / f"sens_{m}.jsonl"):
        r = json.loads(l); d[(r["tag"], r["gran"])] = r
    sens[m] = {f"{t}|{g}": {k: r[k] for k in ("n_units", "pooled_iou", "min_iou", "conf_rel_max", "logit_mae", "pass")}
               for (t, g), r in d.items() if t.startswith(("A:", "B:", "baseline"))}
    single = {f"{t}|{g}": {k: r[k] for k in ("pooled_iou", "conf_rel_max", "logit_mae")} for (t, g), r in d.items() if t.startswith("C:")}
    sens[m]["per_layer_single_unit"] = single
    # greedy summaries
    for g in ("row", "b32", "b128", "row_noproj"):
        p = HERE / f"greedy_{m}_{g}.json"
        if p.exists():
            gj = json.loads(p.read_text())
            sens[m][f"greedy_{g}"] = {"chosen_n": gj["chosen_n"], "chosen_counts": gj.get("chosen_counts"),
                                     "heldout_baseline_bf16_vs_fp32_126f": gj.get("heldout_baseline_bf16_vs_fp32"),
                                     "heldout_sim_fp8_vs_fp32_126f": gj.get("heldout_fp8_vs_fp32"),
                                     "frontier": [{k: f[k] for k in ("n_units", "pooled_iou", "min_iou", "conf_rel_max", "logit_mae", "ok")} for f in gj["frontier"]]}
write_results(R, {"sensitivity": sens})
# variants in previous-rounds format
for m in ("vitl16", "vitb16"):
    for tag, v in list(res.get(m, {}).items()):
        if tag.startswith("fp8-row-") and "bench" in v:
            a = v["bench"]["accuracy_8f_real_fp8"]
            v["accuracy_vs_fp32"] = {"gate": {"pooled_head_iou": a["pooled_iou"], "min_frame_head_iou": a["min_iou"],
                                              "max_conf_rel_diff": a["conf_rel_max"], "pass": a["pass"]}, "note": "8 gate frames, real FP8 + fused head, CUDA graph"}
            b = v["bench"]
            v["fps_e2e"] = None
            v["speedup_graph_model_vs_bf16fusedhead"] = b["bf16+fusedhead graph"]["gpu_p50"] / b["fp8+fusedhead graph"]["gpu_p50"]
R.write_text(json.dumps(res, indent=2))
for m in ("vitl16", "vitb16"):
    for tag, v in res.get(m, {}).items():
        if "bench" in v:
            b = v["bench"]
            print(m, tag, {k: round(b[k]["gpu_p50"], 3) for k in b if isinstance(b[k], dict) and "gpu_p50" in b[k]}, v["accuracy_vs_fp32"]["gate"])
