"""Noise-realisation robustness of candidate plans.  The conf rel diff of ViT-L is chaotic: the same scheme with a different rounding realisation moves a fragile frame by several %.
Each candidate is evaluated for seeds 0..4 (seed 0 = plain block Hadamard, seed k = randomised signs: same statistics, independent error) and, for int8, on two paths:
  emu  = dequantised operands rounded to bf16 (what the sweeps used)      real = the real mg_i8 kernel (exact s32 accumulation)
  python robust_plans.py --model vitl16 [--set core|name1,name2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "impl"))
import numpy as np  # noqa: E402

import emu_eval as ee  # noqa: E402
from emu import make_plan  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--set", default="core")
ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--tag", default="")
a = ap.parse_args()
seeds = [int(x) for x in a.seeds.split(",")]
K4 = ["qkv", "proj", "fc1", "fc2"]


def int8(kinds, layers="all", had=16, pfx=False, sq=0.0, fmt="int8"):
    return {"kinds": kinds, "layers": layers, "spec": dict(fmt=fmt, had=had, pfx=pfx, **({"sq": sq} if sq else {}))}


def fp16(kinds):
    return {"kinds": kinds, "layers": "all", "spec": {"fmt": "fp16"}}


CAND = {
    "int8_all4_had16": [int8(K4)],
    "int8_all4_had32": [int8(K4, had=32)],
    "int8_all4_had16_pfx": [int8(K4, pfx=True)],
    "int8_mlp_had16": [int8(["fc1", "fc2"])],
    "int8_qkv_had16": [int8(["qkv"])], "int8_proj_had16": [int8(["proj"])], "int8_fc1_had16": [int8(["fc1"])], "int8_fc2_had16": [int8(["fc2"])],
    "int8_fc1fc2qkv_had16": [int8(["qkv", "fc1", "fc2"])],
    "int8_mlp_had16+fp16_qkvproj": [int8(["fc1", "fc2"]), fp16(["qkv", "proj"])],
    "int8_all4_sq70had16": [int8(K4, sq=0.7)],
    "fp16_all4": [fp16(K4)],
    "fp16_fc2": [fp16(["fc2"])], "fp16_qkv": [fp16(["qkv"])], "fp16_proj": [fp16(["proj"])], "fp16_fc1": [fp16(["fc1"])],
    "bf16_P0": [],
}
if a.set == "core":
    names = ["bf16_P0", "fp16_all4", "int8_all4_had16", "int8_mlp_had16", "int8_qkv_had16", "int8_proj_had16", "int8_fc1_had16", "int8_fc2_had16", "int8_all4_had32", "int8_all4_had16_pfx",
             "int8_fc1fc2qkv_had16", "int8_mlp_had16+fp16_qkvproj", "int8_all4_sq70had16"]
else:
    names = a.set.split(",")
h = ee.Harness(a.model)
nb = h.net.nb
if any("sq70" in n for n in names):
    h.calibrate()
out = {}
t0 = time.time()
for n in names:
    ents = CAND[n]
    has_i8 = any(e["spec"]["fmt"] == "int8" for e in ents)
    for path in (["emu", "real"] if has_i8 else ["emu"]):
        recs = []
        for sd in seeds:
            if sd != 0 and not any(e["spec"].get("had") for e in ents):
                continue
            es = []
            for e in ents:
                sp = dict(e["spec"])
                if sp["fmt"] == "int8" and path == "real":
                    sp["fmt"] = "int8k"
                if sp.get("had"):
                    sp["seed"] = sd
                es.append({"kinds": e["kinds"], "layers": e["layers"], "spec": sp})
            r = h.eval_plan(make_plan(es, nb), "full")
            recs.append({"seed": sd, "M": r["M"], "H": r["H"]})
        key = f"{n}|{path}"
        if not recs:
            continue
        passM = sum(1 for x in recs if x["M"]["pass"])
        passMH = sum(1 for x in recs if x["M"]["pass"] and x["H"]["pass"])
        out[key] = {"n_seeds": len(recs), "pass_M": passM, "pass_M_and_H": passMH, "M_conf_max_pct_per_seed": [round(x["M"]["conf_max"] * 100, 2) for x in recs],
                    "H_conf_max_pct_per_seed": [round(x["H"]["conf_max"] * 100, 2) for x in recs], "H_iou_per_seed": [round(x["H"]["pooled_iou"], 4) for x in recs],
                    "M_iou_per_seed": [round(x["M"]["pooled_iou"], 4) for x in recs], "H_gt3_frames_per_seed": [x["H"]["n_conf_gt3"] for x in recs],
                    "H_logit_mae_median": float(np.median([x["H"]["logit_mae"] for x in recs])), "entries": ents}
        print(f"{key:40s} seeds {len(recs)}: M pass {passM}/{len(recs)}  M+H pass {passMH}/{len(recs)}  Mconf% {out[key]['M_conf_max_pct_per_seed']}  Hconf% {out[key]['H_conf_max_pct_per_seed']}  "
              f"H>3%frames {out[key]['H_gt3_frames_per_seed']}  mae {out[key]['H_logit_mae_median']:.2e}  ({time.time() - t0:.0f}s)", flush=True)
        (HERE / "profile" / f"robust_{a.model}{a.tag}.json").write_text(json.dumps(out, indent=1))
