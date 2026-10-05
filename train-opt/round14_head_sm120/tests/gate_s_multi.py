"""Gate S (multi-stack): 50-step trajectory on 2 real batches for R0, the bf16 reference and N candidates, for the
real mask schedule ('schedule') and fixed phases (mid, late); gate_c(interval=10, max_rel=0.05) of every candidate vs R0 and
vs the reference.  Curves are cached per (model, stack, phase) so an interrupted run resumes.

  python tests/gate_s_multi.py --model vitl16 --cands TB1_R13 MX8_fc --out profile/gateS_x.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND))
sys.path.insert(0, str(ROUND.parent))

import torch

import run as r14run  # noqa: E402
from common import bench
from common.check import curve, gate_c
from common.harness import build, preload_batches

CACHE = Path("/root/autodl-tmp/train-opt-tmp/round14_head_sm120/curves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--ref", default="TB1")
    ap.add_argument("--cands", nargs="+", default=["TB1_R13"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--phases", nargs="+", default=["schedule", "mid", "late"])
    ap.add_argument("--tag", default="v1", help="cache tag; use a NEW tag for every run (the batches of a run are only shared within it)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    names = ["R0", args.ref, *args.cands]
    curves = {n: {} for n in names}
    batches = None
    t0 = time.time()
    for phase in args.phases:
        for name in names:
            tag = args.tag  # ALL stacks carry the tag: curves are only comparable when computed on the same batches (one run)
            cache = CACHE / f"{args.model}_{r14run.safe_name(name)}_{phase}_{args.steps}_{tag}.json"
            if cache.exists():
                curves[name][phase] = json.loads(cache.read_text())
                print(f"{phase} {name}: cached", flush=True)
                continue
            cfg = r14run.make_cfg(args.model, name, seed=0)
            with bench.gpu_lock():
                r14run.wait_exclusive()
                if batches is None:
                    lc = build(r14run.make_cfg(args.model, "R0", seed=0))
                    batches = preload_batches(lc, k=2)
                    del lc
                    r14run.free_gpu()
                if phase == "schedule":
                    cv = curve(cfg, batches, steps=args.steps)
                else:
                    cv = r14run.curve_phase(cfg, batches, args.steps, phase)
            r14run.free_gpu()
            curves[name][phase] = cv
            cache.write_text(json.dumps(cv))
            print(f"{phase} {name}: finite={cv['finite']} loss[0]={cv['loss'][0]:.4f} loss[-1]={cv['loss'][-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    report = {"command": [sys.executable, *sys.argv], "model": args.model, "gates": {}, "curves": curves}
    for cand in names[1:]:
        for refn in ("R0", args.ref):
            if cand == refn:
                continue
            for phase in args.phases:
                g = gate_c(curves[refn][phase], curves[cand][phase], interval=10, max_rel=0.05)
                key = f"{cand}_vs_{refn}_{phase}"
                report["gates"][key] = g
                gn = [v for v in g["grad_norm_ratios"].values() if v is not None]
                print(f"{key:44s} max_rel {g['curve300_max_rel']:.4f} finite={g['finite']} gn_ratio[{min(gn):.3f},{max(gn):.3f}] PASS={g['pass']}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, allow_nan=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
