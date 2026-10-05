"""Stage 1: per-layer / per-type / per-granularity FP8 sensitivity on the bf16-autocast model.

  python sens.py --model vitl16 --stage all
Writes sens_<model>.jsonl (one line per config) and merges summaries into results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "impl"))
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "round7_head_postprocess"))

import model_factory as mf  # noqa: E402
import pipeline as pl  # noqa: E402
import fp8sim  # noqa: E402

TYPES = ["qkv", "proj", "fc1", "fc2"]


class Evaluator:
    def __init__(self, name):
        self.name = name
        self.model = mf.load_model(name)
        pl.init(self.model)
        self.frames = mf.load_frames()
        self.batches, self.metas = [], []
        with torch.inference_mode():
            for f in self.frames:
                b, m = pl.fast_prep(f)
                self.batches.append(b.clone())
                self.metas.append(m)
        self.refs = [torch.load(mf.ref_path(name, i)) for i in range(len(self.frames))]
        self.ref_logits = [r["logits"].cuda() for r in self.refs]
        self.L = len(self.model.backbone.blocks)

    def run(self):
        acc, mae = {}, []
        with torch.inference_mode():
            for i, (b, meta) in enumerate(zip(self.batches, self.metas)):
                ml, cl = pl.model_lowres(self.model, b, "bf16-mixed")
                mask, conf, logits = pl.fast_post(ml, cl, meta, return_logits=True)
                mae.append((logits - self.ref_logits[i]).abs().mean().item())
                rm = self.refs[i]["mask"].numpy() == 1
                pm = mask == 1
                un, it = int((rm | pm).sum()), int((rm & pm).sum())
                acc[f"frame{i}"] = {"head_inter": it, "head_union": un, "head_iou": it / un if un else 1.0,
                                    "confidence_rel_diff": abs(conf - self.refs[i]["confidence"]) / abs(self.refs[i]["confidence"])}
        g = mf.gate(acc)
        return {"pooled_iou": g["pooled_head_iou"], "min_iou": g["min_frame_head_iou"],
                "conf_rel_max": g["max_conf_rel_diff"], "logit_mae": sum(mae) / len(mae), "pass": bool(g["pass"])}

    def eval_cfg(self, units, gran, act_gran=None):
        t = time.time()
        if units:
            with fp8sim.patched(self.model, units, gran, act_gran):
                r = self.run()
        else:
            r = self.run()
        r["sec"] = round(time.time() - t, 2)
        return r


def units_of(layers, types):
    return [(i, t) for i in layers for t in types]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    ev = Evaluator(args.model)
    L = ev.L
    log = open(HERE / f"sens_{args.model}.jsonl", "a")

    def rec(tag, units, gran, act_gran=None, **extra):
        r = ev.eval_cfg(units, gran, act_gran)
        r.update({"tag": tag, "gran": gran, "n_units": len(units), **extra})
        log.write(json.dumps(r) + "\n")
        log.flush()
        print(f"{tag:40s} {gran:6s} n={len(units):3d} iou {r['pooled_iou']:.4f} minIoU {r['min_iou']:.4f} "
              f"conf {r['conf_rel_max']*100:5.2f}% mae {r['logit_mae']:.5f} {'PASS' if r['pass'] else 'FAIL'}", flush=True)
        return r

    base = rec("baseline-bf16", [], "row")
    st = args.stage
    all_layers = list(range(L))
    grans = ["tensor", "row", "b32", "b128"]
    if st in ("A", "all"):
        for gran in grans:
            for name, types in [("qkv", ["qkv"]), ("proj", ["proj"]), ("fc1", ["fc1"]), ("fc2", ["fc2"]),
                                ("attn", ["qkv", "proj"]), ("mlp", ["fc1", "fc2"]), ("all4", TYPES)]:
                rec(f"A:{name}/all-layers", units_of(all_layers, types), gran)
    if st in ("B", "all"):
        k = max(L // 4, 1)
        ranges = {"first-half": all_layers[: L // 2], "last-half": all_layers[L // 2:],
                  f"first-{k}": all_layers[:k], f"last-{k}": all_layers[-k:],
                  "joint-last3": all_layers[-3:], "all-but-joint": all_layers[:-3]}
        for gran in ("row", "b32"):
            for rname, layers in ranges.items():
                for tname, types in [("all4", TYPES), ("mlp", ["fc1", "fc2"])]:
                    rec(f"B:{tname}/{rname}", units_of(layers, types), gran)
    if st in ("C", "all"):
        for gran in ("row", "b32"):
            for i in all_layers:
                for t in TYPES:
                    rec(f"C:L{i}.{t}", [(i, t)], gran, layer=i, type=t)
                rec(f"C:L{i}.all4", units_of([i], TYPES), gran, layer=i, type="all4")


if __name__ == "__main__":
    main()
