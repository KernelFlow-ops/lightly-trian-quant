import json, statistics, sys
from pathlib import Path
P = Path('/root/lightly-train/train-opt/round20_fp4_probe/profile')
out = {"models": {}}
for model in ("vitl16", "vitb16"):
    d = json.load(open(P / f"fp4_error_table_{model}.json"))
    out["models"][model] = {"capture": d["capture"], "layers": d["layers"]}
    summ = {}
    groups = {"backbone_all": lambda k: k.startswith("block"), "backbone_joint": None, "head": lambda k: k.startswith("head")}
    joint = {"vitl16": "block22", "vitb16": "block10"}[model]
    first = "block0"
    sel = {"backbone_all": lambda k: k.startswith("block"), "backbone_first(block0)": lambda k: k.startswith(first + "."), "backbone_joint": lambda k: k.startswith(joint + "."),
           "backbone_mid": lambda k: k.startswith("block") and not k.startswith(first + ".") and not k.startswith(joint + "."), "head": lambda k: k.startswith("head")}
    for g, f in sel.items():
        keys = [k for k in d["layers"] if f(k) and "relL2" in d["layers"][k]]
        summ[g] = {"n_layers": len(keys)}
        for kind in ("fwd", "dgrad", "wgrad"):
            names = d["layers"][keys[0]]["relL2"][kind].keys()
            summ[g][kind] = {}
            for s in names:
                vals = [d["layers"][k]["relL2"][kind][s] for k in keys if isinstance(d["layers"][k]["relL2"][kind][s], float)]
                if vals:
                    summ[g][kind][s] = {"median": statistics.median(vals), "max": max(vals), "min": min(vals)}
    out["models"][model]["summary"] = summ
(P / "fp4_error_table.json").write_text(json.dumps(out, indent=1))
for model in out["models"]:
    print("=====", model, out["models"][model]["capture"]["n_captured"], "tensors")
    for g, v in out["models"][model]["summary"].items():
        print(" --", g, "n=", v["n_layers"])
        for kind in ("fwd", "dgrad", "wgrad"):
            print("    ", kind, " | ".join(f"{s}:{m['median']:.3f}({m['max']:.3f})" for s, m in v[kind].items()))
