"""Gate-B calibration: how far is each bf16 stack from the fp32 reference, per parameter?

Runs ONE training step with identical weights / batch / RNG tape for three stacks and compares all gradients
pairwise:  R0 (fp32 source semantics), AC (torch autocast = T-BF16 reference), and the candidate.
If AC-vs-R0 per-parameter cosines already fall below the 0.99 gate for some parameters (layer-0 LayerNorm
weights with the massive activations), the per-parameter threshold of check.py is bf16-noise limited and the
candidate must be judged by  cos(cand, R0) >= cos(AC, R0) - margin  for those parameters.

  python tests/calibrate_gateb.py --model vitl16 --cand "r11:autocast,r13:full" --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND))
sys.path.insert(0, str(ROUND.parent))

import torch

import run as r13run  # noqa: E402  (round13/run.py: stack parsing helpers)
from common import bench
from common.check import _capture_step, _clone_tape, _cosine, _rel
from common.harness import apply_patches, build, preload_batches
from common.rng import RngTape


def one(cfg, batches, initial, tape, cpu_rng, cuda_rng):
    ctx = build(cfg)
    apply_patches(ctx)
    if initial is None:
        initial = {k: v.detach().cpu().clone() for k, v in ctx.model.state_dict().items()}
        if batches is None:
            batches = preload_batches(ctx, k=1)
    else:
        ctx.model.load_state_dict(initial, strict=True)
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    out = _capture_step(ctx, batches[0], tape, 0)
    res = {"loss": float(out["out"].loss), "grads": out["grads"]}
    del ctx
    r13run.free_gpu()
    return res, initial, batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--cand", default="r11:autocast,r13:full")
    ap.add_argument("--ref", default="AC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", default="mid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfgs = {
        "R0": r13run.make_cfg(args.model, "R0", seed=args.seed, phase=args.phase),
        "REF": r13run.make_cfg(args.model, args.ref, seed=args.seed, phase=args.phase),
        "CAND": r13run.make_cfg(args.model, args.cand, seed=args.seed, phase=args.phase),
    }
    with bench.gpu_lock():
        r13run.wait_exclusive()
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        tape = RngTape(mode="record")
        results, initial, batches = {}, None, None
        for i, (name, cfg) in enumerate(cfgs.items()):
            if i == 0:
                t_use = tape
            else:
                t_use = RngTape(mode="replay", values={k: v.clone() for k, v in tape.values.items()})
            res, initial, batches = one(cfg, batches, initial, t_use, cpu_rng, cuda_rng)
            results[name] = res
            print(name, "loss", res["loss"], flush=True)
    names = list(results["R0"]["grads"])
    rows = {}
    for n in names:
        g0, gr, gc = (results[k]["grads"][n] for k in ("R0", "REF", "CAND"))
        rows[n] = {"ref_vs_r0": _cosine(gr, g0), "cand_vs_r0": _cosine(gc, g0), "cand_vs_ref": _cosine(gc, gr)}

    def norm(g):
        return math.sqrt(sum(float(torch.dot(x.double().reshape(-1), x.double().reshape(-1))) for x in g.values() if x is not None))

    summary = {
        "loss": {k: results[k]["loss"] for k in results},
        "grad_norm": {k: norm(results[k]["grads"]) for k in results},
        "min_cos": {k: min(r[k] for r in rows.values()) for k in ("ref_vs_r0", "cand_vs_r0", "cand_vs_ref")},
        "n_params": len(rows),
        "n_below_0.99": {k: sum(1 for r in rows.values() if r[k] < 0.99) for k in ("ref_vs_r0", "cand_vs_r0", "cand_vs_ref")},
    }
    # calibrated per-parameter criterion: the candidate must be as close to fp32 as the bf16 reference is (minus a margin)
    margin = 0.02
    offenders = {n: r for n, r in rows.items() if r["cand_vs_r0"] < min(0.99, r["ref_vs_r0"] - margin)}
    summary["calibrated_pass"] = len(offenders) == 0
    summary["calibration_margin"] = margin
    summary["calibrated_offenders"] = {n: [round(r["ref_vs_r0"], 4), round(r["cand_vs_r0"], 4)] for n, r in offenders.items()}
    gcos = {}
    worst = sorted(rows.items(), key=lambda kv: kv[1]["cand_vs_ref"])[:12]
    print(json.dumps(summary, indent=1))
    print("worst by cand_vs_ref:")
    for n, r in worst:
        print(f"  {n:48s} ref/R0={r['ref_vs_r0']:.5f} cand/R0={r['cand_vs_r0']:.5f} cand/ref={r['cand_vs_ref']:.5f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "per_param": rows, "args": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
