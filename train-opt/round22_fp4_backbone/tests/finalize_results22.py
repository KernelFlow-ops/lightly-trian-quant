"""Add the hand-written headline / decision block to results.json (after tests/summarize_r22.py).  Every number is read from the measured json already merged into results.json."""
import json
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
p = ROUND / "results.json"
r = json.loads(p.read_text())
s0 = r.get("bench_sessions", {}).get("session0", {})
head = {}
for model, cand in (("vitl16", "P3B"), ("vitb16", "P3B_B")):
    m = s0.get(model, {})
    if "FINAL" in m and cand in m:
        head[model] = {"candidate_stack": cand, "FINAL_img_s": m["FINAL"]["weighted_img_s_mean"], "candidate_img_s": m[cand]["weighted_img_s_mean"], "ratio": m[cand]["vs_FINAL_img_s"],
                       "FINAL_mid_ms": m["FINAL"]["mid_ms_mean"], "candidate_mid_ms": m[cand]["mid_ms_mean"], "mid_ms_delta": m[cand]["vs_FINAL_mid_ms_delta"],
                       "publish_threshold_2pct_met": bool(m[cand]["vs_FINAL_img_s"] >= 1.02)}
r["headline"] = {
    "policy": {"vitl16": "all.d=nv4+sr+all.w=nv4+sr+nofirst  (preset P3B)", "vitb16": "all.d=nv4+sr+all.w=nv4+sr  (preset P3B_B)"},
    "release_stack": 'BEST_STACK_V2 + ("round22_fp4_backbone",), variant = policy above; head untouched (FINAL r14 bf16)',
    "amax": "delayed, headroom X=2 weights=2 dY=8 (READY_headroom existed before every run)",
    "speed": head,
    "vs_R20a_estimate_ms": {"vitl16": {"estimate": 6.5, "nsys_kernel_ms": 4.99, "bench_mid_ms": -head.get("vitl16", {}).get("mid_ms_delta")}, "vitb16": {"estimate": 3.8, "nsys_kernel_ms": 2.62, "bench_mid_ms": -head.get("vitb16", {}).get("mid_ms_delta")}},
    "nsys_categories_ms_per_step": "profile/nsys_report22.json (labels L_FINAL L_P3B B_FINAL B_P3B_B)",
    "requests": ["R21: act_kernel latency for L2-resident small shapes (dY dual+SR 47 us, X col 17.8 us in ViT-L)", "R21: one kernel emitting mx8 row + nv4 transposed copy of X", "R21: weight quantizer row=False / fused mx8 row output", "R21: Philox 7 rounds option"],
    "not_measured": ["stacking with the R21b head patch", "ViT-L S gate without nofirst", "peak step memory FINAL vs FINAL+r22"],
}
p.write_text(json.dumps(r, indent=1, default=str))
print(json.dumps(r["headline"]["speed"], indent=1))
