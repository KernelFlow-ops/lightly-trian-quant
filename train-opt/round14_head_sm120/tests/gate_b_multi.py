"""Gate B (multi-stack, one process): one training step with identical weights / batch / RNG tape for R0 (fp32),
TB1 (T-BF16) and N candidates; per-parameter and global gradient / update cosines against both references.

Criteria (CONTRACTS Amendment 1, 9.4)
  same-precision candidate (kind=same; e.g. TB1+R13), reference TB1:
      loss_rel <= 1e-2, global grad cos >= 0.99, grad_norm_rel <= 0.02, global update cos >= 0.99,
      per-parameter (grad) cos(cand,R0) >= min(0.99, cos(TB1,R0) - 0.02)          [calibrated]
  precision-change candidate (kind=prec; e.g. MXFP8), vs R0 and vs TB1(+R13):
      loss_rel <= 2e-2 and global grad cos >= 0.99; per-parameter cos(cand,R0) >= cos(TB1,R0) - 0.05 is recorded
      (offenders listed, not a blocking criterion).

  python tests/gate_b_multi.py --model vitl16 --cands TB1_R13 MX8_fc --seeds 0 1 2 --out profile/gateB_x.json
Candidates whose name starts with MX8_ are kind=prec, everything else kind=same.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND))
sys.path.insert(0, str(ROUND.parent))

import torch

import run as r14run  # noqa: E402
from common import bench
from common.check import _capture_step, _rel
from common.harness import apply_patches, build, preload_batches
from common.rng import RngTape


def capture(cfg, initial, batches, tape, rng):
    ctx = build(cfg)
    apply_patches(ctx)
    if initial is None:
        initial = {k: v.detach().cpu().clone() for k, v in ctx.model.state_dict().items()}
        batches = preload_batches(ctx, k=1)
        rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state())
    else:
        ctx.model.load_state_dict(initial, strict=True)
    torch.random.set_rng_state(rng[0])
    torch.cuda.set_rng_state(rng[1])
    torch.cuda.reset_peak_memory_stats()
    out = _capture_step(ctx, batches[0], tape, 0)
    res = {"loss": float(out["out"].loss), "grads": out["grads"], "updates": out["updates"], "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
    complete = True
    if tape.mode == "replay":
        try:
            tape.assert_complete()
        except AssertionError as exc:
            complete = str(exc)[:200]
    res["tape_complete"] = complete
    del ctx
    r14run.free_gpu()
    return res, initial, batches, rng


def cmp(a, b):
    g = r14run._accumulate(a["grads"], b["grads"])
    u = r14run._accumulate(a["updates"], b["updates"])
    per = g["per_param_cos"]
    worst = min(per.items(), key=lambda kv: kv[1])
    return {
        "loss_a": a["loss"], "loss_b": b["loss"], "loss_rel": _rel(a["loss"], b["loss"]),
        "grad_cos_global": g["global_cos"], "grad_cos_min": worst[1], "grad_cos_min_name": worst[0],
        "grad_norm_a": g["norm_a"], "grad_norm_b": g["norm_b"], "grad_norm_rel": _rel(g["norm_a"], g["norm_b"]),
        "update_cos_global": u["global_cos"], "update_cos_min": min(u["per_param_cos"].values()),
        "group_cos": g["group_cos"], "per_param": per, "per_param_update": u["per_param_cos"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--ref", default="TB1", help="bf16 reference stack (default TB1)")
    ap.add_argument("--cands", nargs="+", default=["TB1_R13"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--phase", default="mid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {"command": [sys.executable, *sys.argv], "model": args.model, "phase": args.phase, "seeds": {}}
    outp = Path(args.out)
    if outp.exists():
        old = json.loads(outp.read_text())
        report["seeds"] = old.get("seeds", {})
    names = ["R0", args.ref, *args.cands]
    for seed in args.seeds:
        if str(seed) in report["seeds"]:
            print(f"seed {seed} cached", flush=True)
            continue
        t0 = time.time()
        results, initial, batches, rng = {}, None, None, None
        tape = RngTape(mode="record")
        with bench.gpu_lock():
            r14run.wait_exclusive()
            for i, name in enumerate(names):
                cfg = r14run.make_cfg(args.model, name, seed=seed, phase=args.phase)
                use = tape if i == 0 else RngTape(mode="replay", values={k: v.clone() for k, v in tape.values.items()})
                results[name], initial, batches, rng = capture(cfg, initial, batches, use, rng)
                print(f"  seed {seed} {name}: loss {results[name]['loss']:.6f} tape_complete={results[name]['tape_complete']} peak {results[name]['peak_gb']:.1f}GB ({time.time() - t0:.0f}s)", flush=True)
        ref_vs_r0 = cmp(results["R0"], results[args.ref])
        seed_rep = {"loss": {k: v["loss"] for k, v in results.items()}, "peak_gb": {k: v["peak_gb"] for k, v in results.items()},
                    "ref_vs_r0": {k: v for k, v in ref_vs_r0.items() if k not in ("per_param", "per_param_update")}, "cands": {}}
        for name in args.cands:
            c_r0 = cmp(results["R0"], results[name])
            c_ref = cmp(results[args.ref], results[name])
            prec = name.startswith("MX8_") or name.startswith("MXH_")
            margin = 0.05 if prec else 0.02
            # Amendment 2 (10.2): per-parameter criterion  cos(cand,R0) >= min(0.99, cos(ref,R0)-margin)  OR  cos(cand,ref) >= 0.99
            off = {n: [round(ref_vs_r0["per_param"][n], 4), round(c_r0["per_param"][n], 4)] for n in c_r0["per_param"]
                   if c_r0["per_param"][n] < min(0.99, ref_vs_r0["per_param"][n] - margin) and c_ref["per_param"][n] < 0.99}
            off_u = {n: [round(ref_vs_r0["per_param_update"][n], 4), round(c_r0["per_param_update"][n], 4)] for n in c_r0["per_param_update"]
                     if c_r0["per_param_update"][n] < min(0.99, ref_vs_r0["per_param_update"][n] - margin)}
            if prec:
                ok = (c_r0["loss_rel"] <= 2e-2 and c_r0["grad_cos_global"] >= 0.99 and c_ref["loss_rel"] <= 2e-2 and c_ref["grad_cos_global"] >= 0.99)
                crit = "prec: loss_rel<=2e-2 and global grad cos>=0.99 vs R0 and vs ref; per-param (margin 0.05) recorded"
            else:
                ok = (c_ref["loss_rel"] <= 1e-2 and c_ref["grad_cos_global"] >= 0.99 and c_ref["grad_norm_rel"] <= 0.02
                      and c_ref["update_cos_global"] >= 0.99 and not off)
                crit = "same: loss_rel<=1e-2, global grad cos>=0.99, grad_norm_rel<=0.02, update cos>=0.99 (vs ref), calibrated per-param (margin 0.02)"
            seed_rep["cands"][name] = {
                "kind": "prec" if prec else "same", "criterion": crit, "pass": bool(ok), "tape_complete": results[name]["tape_complete"],
                "vs_r0": {k: v for k, v in c_r0.items() if k not in ("per_param", "per_param_update")},
                "vs_ref": {k: v for k, v in c_ref.items() if k not in ("per_param", "per_param_update")},
                "per_param_offenders_vs_r0": off, "update_offenders_vs_r0": off_u,
                "per_param_cos_vs_r0": {k: round(v, 6) for k, v in c_r0["per_param"].items()},
                "per_param_cos_vs_ref": {k: round(v, 6) for k, v in c_ref["per_param"].items()},
            }
            r, f = c_r0, c_ref
            kind = "prec" if prec else "same"
            print(f"seed {seed} {name:24s} [{kind}] vsR0: loss_rel {r['loss_rel']:.2e} gcos {r['grad_cos_global']:.5f} nrel {r['grad_norm_rel']:.3f} ucos {r['update_cos_global']:.4f} "
                  f"| vs{args.ref}: loss_rel {f['loss_rel']:.2e} gcos {f['grad_cos_global']:.5f} nrel {f['grad_norm_rel']:.3f} ucos {f['update_cos_global']:.4f} pmin {f['grad_cos_min']:.4f} "
                  f"| offenders {len(off)} PASS={ok}", flush=True)
        print(f"seed {seed} {args.ref} vs R0: loss_rel {ref_vs_r0['loss_rel']:.2e} gcos {ref_vs_r0['grad_cos_global']:.5f} pmin {ref_vs_r0['grad_cos_min']:.4f} nrel {ref_vs_r0['grad_norm_rel']:.3f}", flush=True)
        report["seeds"][str(seed)] = seed_rep
        outp.write_text(json.dumps(report, indent=1, allow_nan=False))
        del results
        r14run.free_gpu()
    print("wrote", outp)


if __name__ == "__main__":
    main()
