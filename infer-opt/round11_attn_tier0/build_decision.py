"""Assemble profile/attn_decision.json (+ markdown tables for NOTES) from the measured files.  Nothing is estimated: missing inputs -> null.

  python build_decision.py
Inputs: profile/attn_speed_{A,B,C}_{b2,b4}.json, profile/attn_layer_scan_<model>.json, profile/attn_operand_scan.json,
        results.json["bench11"] (tags attn_swap, attn_swap2, attn_perf2, paired*, ...), profile/attn16_cfg_sweep.json
"""
import json
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / "profile"
res = json.loads((HERE / "results.json").read_text())["bench11"]


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- speed
speed = {}
for g in "ABC":
    for b in ("b2", "b4"):
        d = jload(P / f"attn_speed_{g}_{b}.json")
        if not d:
            continue
        for sh, row in d["speed"].items():
            for cand, v in row.items():
                if cand == "flash" and cand in speed.get(sh, {}):
                    speed[sh][cand]["flash_repeat_us"] = speed[sh][cand].get("flash_repeat_us", []) + [v["kernel_us"]]
                    continue
                speed.setdefault(sh, {})[cand] = v
CANDS = ["flash", "tl", "a16k0", "a16k1", "a16k2", "a16k3", "a16k4", "sage2_fp8", "sage2_fp8_f32", "sage2_pv16", "sage2_pv16_f32", "sage3", "spark_fp8"]


def per_frame_saving_us(model, cand, field="kernel_us"):
    """sum over layers of (flash - cand); B=2; ViT-L: 21 x N1029 + 3 x N1129 at H16; ViT-B: 9 + 3 at H12."""
    H, n_plain, n_joint = (16, 21, 3) if model == "vitl16" else (12, 9, 3)
    try:
        s = lambda N: speed[f"2x{H}x{N}"]  # noqa: E731
        return n_plain * (s(1029)["flash"]["kernel_us"] - s(1029)[cand][field]) + n_joint * (s(1129)["flash"]["kernel_us"] - s(1129)[cand][field])
    except KeyError:
        return None


# ---------------------------------------------------------------- operator accuracy (real qkv, all layers, 2 frames)
opacc = {}
for m in ("vitl16", "vitb16"):
    d = jload(P / f"attn_layer_scan_{m}.json")
    if not d:
        continue
    cands = [k for k in next(iter(d.values())) if k not in ("S_scaled_absmax_x8", "N")]
    opacc[m] = {}
    for c in cands:
        vals = [(k, v[c]["rel_l2"]) for k, v in d.items() if "rel_l2" in v.get(c, {}) and v[c].get("finite", True)]
        nonfin = [k for k, v in d.items() if c in v and not v[c].get("finite", True)]
        opacc[m][c] = {"mean_rel_l2": statistics.mean(x[1] for x in vals), "max_rel_l2": max(x[1] for x in vals), "worst_layer": max(vals, key=lambda x: x[1])[0],
                       "nonfinite_layer_frames": nonfin, "n": len(vals)}
    opacc[m]["_S_absmax_by_layer_frame"] = {k: v["S_scaled_absmax_x8"] for k, v in d.items() if v["S_scaled_absmax_x8"] > 3000}

# ---------------------------------------------------------------- model-level gates + fps
gates = {}
fps = {}
for key, v in res.items():
    model, vars_, tag = key.split("|")
    for var, r in v["results"].items():
        if "M_8frame" in r:
            M, H = r["M_8frame"], r["H_126frame"]
            gates.setdefault(model, {}).setdefault(var, {})[tag] = {
                "M": {"pooled_iou": M["pooled_head_iou"], "min_frame_iou": M["min_frame_head_iou"], "conf_rel_max": M["max_conf_rel_diff"], "pass": M["pass"]},
                "H": {"pooled_iou": H["pooled_iou"], "min_frame_iou": H["min_frame_iou"], "conf_rel_max": H["conf_rel_max"],
                      "frames_conf_gt3pct": H["frames_conf_gt3pct"], "pass": H["pass"], "score_mae": H.get("score_mae")},
                "G_graph_eq_eager": r["G_model_fn_graph_vs_eager"]["mask_logits"]["identical"] and r["G_model_fn_graph_vs_eager"]["class_logits"]["identical"],
                "G_pipeline_identical_masks": r["G_pipeline_graph_vs_nograph"]["identical_masks"]}
        if "fps_median" in r:
            fps.setdefault(model, {}).setdefault(tag, {})[var] = {"fps_median": r["fps_median"], "vs_first_pct": r["fps_vs_first_pct"], "rounds": r["fps_rounds"],
                                                                  "first": vars_.split(",")[0]}

out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": "RTX 5090 sm_120a", "note": "all numbers measured in this round; null = not measured",
       "speed_us_graph_replay": speed, "operator_accuracy_real_qkv_all_layers_frames0_7": opacc, "model_level_gates": gates, "end_to_end_fps": fps,
       "per_frame_attention_saving_us_vs_flash_kernel_only": {m: {c: per_frame_saving_us(m, c) for c in CANDS if c != "flash"} for m in ("vitl16", "vitb16")},
       "per_frame_attention_saving_us_vs_flash_full_path": {m: {c: per_frame_saving_us(m, c, "full_us") for c in CANDS if c != "flash"} for m in ("vitl16", "vitb16")}}
rec = jload(HERE / "profile" / "attn_recommendation.json")
if rec:
    out["recommendation"] = rec
(P / "attn_decision.json").write_text(json.dumps(out, indent=1))

# ---------------------------------------------------------------- markdown tables for NOTES
print("### speed (us, kernel-only / full path)")
for sh in sorted(speed, key=lambda s: (int(s.split("x")[0]), -int(s.split("x")[1]), int(s.split("x")[2]))):
    row = speed[sh]
    print(f"{sh}: " + " | ".join(f"{c} {row[c]['kernel_us']:.1f}/{row[c]['full_us']:.1f} ({row[c]['kernel_TFLOPS']:.0f}TF)" for c in CANDS if c in row))
print("### per-frame saving (us) kernel-only / full")
for m in ("vitl16", "vitb16"):
    print(m, {c: (None if per_frame_saving_us(m, c) is None else round(per_frame_saving_us(m, c))) for c in CANDS if c != "flash"})
    print(m, "full", {c: (None if per_frame_saving_us(m, c, "full_us") is None else round(per_frame_saving_us(m, c, "full_us"))) for c in CANDS if c != "flash"})
print("### operator accuracy")
for m, d in opacc.items():
    print(m, {c: f"{v['mean_rel_l2']:.2e}/{v['max_rel_l2']:.2e}" for c, v in d.items() if not c.startswith("_")})
