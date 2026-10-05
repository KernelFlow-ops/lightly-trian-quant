"""Greedy FP8 coverage search + held-out validation on all 126 video frames.

  python greedy.py --model vitl16 --gran row
Unit = (layer, type). Units are ranked by single-unit logit MAE increase (least sensitive first),
prefixes of the ranking are evaluated on the 8 gate frames; the chosen prefix is then validated on
all 126 frames against an fp32 eager reference computed on the fly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sens  # noqa: E402
import fp8sim  # noqa: E402
import model_factory as mf  # noqa: E402
import pipeline as pl  # noqa: E402


def frames_all():
    cap = cv2.VideoCapture(mf.VIDEO_PATH)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def masks_conf(model, precision, frames, metas_cache={}):
    res = []
    with torch.inference_mode():
        for f in frames:
            b, meta = pl.fast_prep(f)
            ml, cl = pl.model_lowres(model, b, precision)
            mask, conf = pl.fast_post(ml, cl, meta)
            res.append((mask.copy(), conf))
    return res


def heldout(model_name, ev, units, gran, frames, ref):
    with fp8sim.patched(ev.model, units, gran):
        cand = masks_conf(ev.model, "bf16-mixed", frames)
    inter = union = 0
    cr = []
    ious = []
    for (rm, rc), (m, c) in zip(ref, cand):
        a, b = rm == 1, m == 1
        i, u = int((a & b).sum()), int((a | b).sum())
        inter += i
        union += u
        ious.append(i / u if u else 1.0)
        cr.append(abs(c - rc) / abs(rc))
    return {"pooled_iou": inter / union, "min_iou": min(ious), "conf_rel_max": max(cr),
            "conf_rel_mean": float(np.mean(cr)), "conf_rel_p90": float(np.percentile(cr, 90)), "frames": len(ref)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gran", default="row")
    ap.add_argument("--types", default="qkv,proj,fc1,fc2")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--conf-max", type=float, default=0.03)
    ap.add_argument("--iou-min", type=float, default=0.975)
    args = ap.parse_args()
    ev = sens.Evaluator(args.model)
    L = ev.L
    base = ev.eval_cfg([], args.gran)
    units = [(i, t) for i in range(L) for t in args.types.split(',')]
    single = {}
    for u in units:
        r = ev.eval_cfg([u], args.gran)
        single[u] = r
    order = sorted(units, key=lambda u: (single[u]["logit_mae"] - base["logit_mae"], single[u]["conf_rel_max"]))
    print("least sensitive 12:", [(u, round((single[u]["logit_mae"] - base["logit_mae"]) * 1e5, 2)) for u in order[:12]])
    print("most sensitive 12:", [(u, round((single[u]["logit_mae"] - base["logit_mae"]) * 1e5, 2)) for u in order[-12:]])
    step = 2
    frontier = []
    best = None
    for n in list(range(step, len(order) + 1, step)):
        r = ev.eval_cfg(order[:n], args.gran)
        r["n_units"] = n
        r["frac_flops"] = None
        ok = r["pass"] and r["conf_rel_max"] <= args.conf_max and r["pooled_iou"] >= args.iou_min
        r["ok"] = ok
        frontier.append(r)
        if ok:
            best = n
        print(f"prefix {n:3d}/{len(order)} iou {r['pooled_iou']:.4f} min {r['min_iou']:.4f} conf {r['conf_rel_max']*100:5.2f}% "
              f"mae {r['logit_mae']*1e5:.1f}e-5 {'OK' if ok else '--'}", flush=True)
    out = {"gran": args.gran, "baseline": base, "order": [list(u) for u in order],
           "single_mae_delta_e5": {f"{u[0]}.{u[1]}": (single[u]["logit_mae"] - base["logit_mae"]) * 1e5 for u in units},
           "single_conf": {f"{u[0]}.{u[1]}": single[u]["conf_rel_max"] for u in units},
           "frontier": frontier, "chosen_n": best}
    if best:
        chosen = order[:best]
        out["chosen_units"] = [list(u) for u in chosen]
        counts = {t: sum(1 for u in chosen if u[1] == t) for t in sens.TYPES}
        out["chosen_counts"] = counts
        print("chosen", best, counts, flush=True)
        frames = frames_all()
        # fp32 reference on the fly
        m32 = mf.load_model(args.model)
        pl.init(m32)
        ref = masks_conf(m32, "32-true", frames)
        ev.model, saved = ev.model, None
        pl.init(ev.model)
        base_ref = heldout(args.model, ev, [], args.gran, frames, ref)
        out["heldout_baseline_bf16_vs_fp32"] = base_ref
        out["heldout_fp8_vs_fp32"] = heldout(args.model, ev, chosen, args.gran, frames, ref)
        print("heldout baseline", base_ref, "\nheldout fp8", out["heldout_fp8_vs_fp32"], flush=True)
    (HERE / f"greedy_{args.model}_{args.gran}{args.suffix}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
