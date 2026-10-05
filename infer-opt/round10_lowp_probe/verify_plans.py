"""GPU verification of the post-go-rule plans (profile/final_*.json): gate M + gate H (+ alt-calibration robustness when SmoothQuant / GPTQ is used) -> profile/verify_<name>.json
   python verify_plans.py [name ...]"""
import json
import os
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "impl"))
import emu_eval as ee
from emu import make_plan

REAL = "--real" in sys.argv                    # int8 -> int8k: the real mg_i8 kernel (exact s32 accumulation) instead of the bf16-rounded dequantised operands (GPTQ entries stay emulated)
args = [a for a in sys.argv[1:] if a != "--real"]
names = args or [p.stem[len("final_"):] for p in sorted((HERE / "profile").glob("final_*.json"))]
by_model = {}
for n in names:
    d = json.loads((HERE / "profile" / f"final_{n}.json").read_text())
    if REAL:
        for e in d["entries_after_go_rule"]:
            if e["spec"].get("fmt") == "int8" and not e["spec"].get("gptq"):
                e["spec"] = dict(e["spec"], fmt="int8k")
    by_model.setdefault(d["model"], []).append((n, d))
for model, lst in by_model.items():
    h = ee.Harness(model)
    nb = h.net.nb
    need = any(s["spec"].get("sq") or s["spec"].get("gptq") for _, d in lst for s in d["entries_after_go_rule"])
    if need:
        h.calibrate(hess_kinds=("qkv", "proj", "fc1", "fc2"))
        import gptq
        gptq.install(h.net)
    for n, d in lst:
        t0 = time.time()
        res = h.eval_plan(make_plan(d["entries_after_go_rule"], nb), "full")
        out = {k: v for k, v in res.items() if k != "per_frame"}
        out["per_frame_conf_rel"] = res["per_frame"]["conf_rel"]
        out["per_frame_iou"] = res["per_frame"]["iou"]
        uses_cal = any(s["spec"].get("sq") or s["spec"].get("gptq") for s in d["entries_after_go_rule"])
        if uses_cal:
            h.calibrate(hess_kinds=("qkv", "proj", "fc1", "fc2"), idxs=ee.CAL_ALT)
            r2 = h.eval_plan(make_plan(d["entries_after_go_rule"], nb), "full")
            out["alt_calibration"] = {k: v for k, v in r2.items() if k != "per_frame"}
            h.calibrate(hess_kinds=("qkv", "proj", "fc1", "fc2"))      # restore the primary calibration
        out["calib_frames"] = ee.CAL_IDX
        (HERE / "profile" / f"verify_{n}{'_realkernel' if REAL else ''}{('_' + Path(os.environ['MG16_TUNED']).stem) if os.environ.get('MG16_TUNED') else ''}.json").write_text(json.dumps(out, indent=1))
        print(n, "M", round(out["M"]["pooled_iou"], 4), f"{out['M']['conf_max'] * 100:.2f}%", out["M"]["pass"], "| H", round(out["H"]["pooled_iou"], 4), "min", round(out["H"]["min_iou"], 3),
              f"conf {out['H']['conf_max'] * 100:.2f}% >3%:{out['H']['n_conf_gt3']} mae {out['H']['logit_mae']:.2e}", out["H"]["pass"], f"({time.time() - t0:.0f}s)",
              ("| alt-cal H " + f"{out['alt_calibration']['H']['pooled_iou']:.4f}/{out['alt_calibration']['H']['conf_max'] * 100:.2f}% M {out['alt_calibration']['M']['pooled_iou']:.4f}") if uses_cal else "", flush=True)
