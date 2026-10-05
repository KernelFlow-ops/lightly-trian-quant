"""Gate B with the literal ``common.check.step_equivalence`` (loss, every gradient, grad norm, every update), 3 seeds.

Reference = T-BF16 stack (TB1), candidate = TB1 + round13, same weights / batch / RngTape (record -> replay).
check.py's thresholds (loss_rel<=1e-2, grad_cos_min>=0.99, grad_norm_rel<=0.02, update_cos_min>=0.99) are reported as is;
the per-parameter gradient cosine of two bf16 stacks is bf16-noise limited for layer-0 parameters (see calibrate_gateb.py),
so the calibrated criterion is reported next to it:  cos(cand, R0) >= min(0.99, cos(ref, R0) - 0.02) per parameter, taken
from profile/calibrate_gateb_<model>_seed<k>.json.

  python tests/gate_b_step_equivalence.py --model vitl16 --seeds 0 1 2
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

import run as r13run  # noqa: E402
from common import bench
from common.check import step_equivalence
from common.harness import apply_patches, build, preload_batches
from common.rng import RngTape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--ref", default="TB1")
    ap.add_argument("--cand", default="TB1_R13")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--phase", default="mid")
    args = ap.parse_args()
    out = {"args": vars(args), "seeds": {}}
    for seed in args.seeds:
        t0 = time.time()
        cfg_ref = r13run.make_cfg(args.model, args.ref, seed=seed, phase=args.phase)
        cfg_opt = r13run.make_cfg(args.model, args.cand, seed=seed, phase=args.phase)
        with bench.gpu_lock():
            r13run.wait_exclusive()
            ctx = build(cfg_ref)
            batches = preload_batches(ctx, k=1)
            del ctx
            r13run.free_gpu()
            res = step_equivalence(cfg_ref, cfg_opt, batches[:1], tape=RngTape(mode="record"))
        gc = res["grad_cos"]
        below = sorted(((v, k) for k, v in gc.items() if v < 0.99))
        short = {k: res[k] for k in ("loss_rel", "grad_cos_min", "grad_norm_rel", "update_cos_min", "pass")}
        short["n_params"] = len(gc)
        short["n_grad_cos_below_0.99"] = len(below)
        short["worst5"] = [(k, round(v, 4)) for v, k in below[:5]]
        short["seconds"] = round(time.time() - t0, 1)
        out["seeds"][str(seed)] = {**short, "grad_cos": {k: round(v, 6) for k, v in gc.items()}}
        print(json.dumps(short), flush=True)
    path = ROUND / "profile" / f"gateB_step_equivalence_{args.model}_{args.cand}_vs_{args.ref}_{args.phase}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
