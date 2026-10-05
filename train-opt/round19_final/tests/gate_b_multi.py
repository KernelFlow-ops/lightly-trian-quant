"""Gate B-lite on the FINAL stack, multi-batch rule (CONTRACTS Amendment 2 10.2 + Amendment 4), one process per model.

Per seed (= one real batch; the loader is seeded with the seed): R0 (fp32 original step), TB1/P1 (torch BF16) and the candidate (FINAL, eager, same
weights / batch / RngTape replay) each run one training step; per-parameter and global gradient / update cosines are computed against R0 and against P1.

Pass rule for FINAL (a precision change: MXFP8), separately against R0 and against P1, over >= 8 batches:
    median global grad cos >= 0.995, min >= 0.98, max loss_rel <= 2e-2
Per parameter (recorded, not a blocking criterion, Amendment 4): cos(cand,R0) >= cos(P1,R0) - 0.05, parameters whose cos(P1,R0) is itself < 0.9 are judged
by the global metrics only.

  python tests/gate_b_multi.py --model vitl16 --cands FINAL --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_vitl16_FINAL.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND))
sys.path.insert(0, str(ROUND.parent))

import torch  # noqa: E402

import run as _r19run  # noqa: E402,F401  (registers the presets and redirects round16's outputs)
from round16_full_graph import run as r16  # noqa: E402
from common import bench  # noqa: E402
from common.check import _capture_step, _rel  # noqa: E402
from common.harness import apply_patches, build, preload_batches  # noqa: E402
from common.rng import RngTape  # noqa: E402


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
    res = {"loss": float(out["out"].loss), "grads": out["grads"], "updates": out["updates"], "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
           "batch_sum": float(batches[0]["image"].double().sum())}
    complete = True
    if tape.mode == "replay":
        try:
            tape.assert_complete()
        except AssertionError as exc:
            complete = str(exc)[:200]
    res["tape_complete"] = complete
    del ctx
    r16.free_gpu()
    return res, initial, batches, rng


def cmp(a, b):
    g = r16._accumulate(a["grads"], b["grads"])
    u = r16._accumulate(a["updates"], b["updates"])
    per = g["per_param_cos"]
    worst = min(per.items(), key=lambda kv: kv[1])
    return {
        "loss_a": a["loss"], "loss_b": b["loss"], "loss_rel": _rel(a["loss"], b["loss"]),
        "grad_cos_global": g["global_cos"], "grad_cos_min": worst[1], "grad_cos_min_name": worst[0],
        "grad_norm_a": g["norm_a"], "grad_norm_b": g["norm_b"], "grad_norm_rel": _rel(g["norm_a"], g["norm_b"]),
        "update_cos_global": u["global_cos"], "update_cos_min": min(u["per_param_cos"].values()),
        "group_cos": g["group_cos"], "per_param": per, "per_param_update": u["per_param_cos"],
    }


def summarize(report, cands, ref):
    """Multi-batch rule over the seeds in the report."""
    out = {}
    for name in cands:
        rows = [s["cands"][name] for s in report["seeds"].values() if name in s["cands"]]
        if not rows:
            continue
        o = {"n_batches": len(rows)}
        for tag in ("vs_r0", "vs_ref"):
            gc = [r[tag]["grad_cos_global"] for r in rows]
            lr = [r[tag]["loss_rel"] for r in rows]
            nr = [r[tag]["grad_norm_rel"] for r in rows]
            o[tag] = {"grad_cos_median": statistics.median(gc), "grad_cos_min": min(gc), "grad_cos_max": max(gc), "loss_rel_max": max(lr), "grad_norm_rel_max": max(nr),
                      "grad_cos_each": gc, "pass": bool(len(rows) >= 8 and statistics.median(gc) >= 0.995 and min(gc) >= 0.98 and max(lr) <= 2e-2)}
        o["pass"] = bool(o["vs_r0"]["pass"] and o["vs_ref"]["pass"])
        o["per_param_offenders_union"] = sorted({n for r in rows for n in r["per_param_offenders_vs_r0"]})
        o["per_param_offenders_count_per_batch"] = [len(r["per_param_offenders_vs_r0"]) for r in rows]
        out[name] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--ref", default="TB1", help="bf16 reference stack (default TB1 = P1)")
    ap.add_argument("--cands", nargs="+", default=["FINAL"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--phase", default="mid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {"command": [sys.executable, *sys.argv], "model": args.model, "phase": args.phase, "ref": args.ref, "seeds": {}}
    outp = Path(args.out)
    if outp.exists():
        report["seeds"] = json.loads(outp.read_text()).get("seeds", {})
    names = ["R0", args.ref, *args.cands]
    for seed in args.seeds:
        if str(seed) in report["seeds"]:
            print(f"seed {seed} cached", flush=True)
            continue
        t0 = time.time()
        results, initial, batches, rng = {}, None, None, None
        tape = RngTape(mode="record")
        with bench.gpu_lock():
            r16.wait_exclusive()
            for i, name in enumerate(names):
                cfg = r16.make_cfg(args.model, name, seed=seed, phase=args.phase)
                use = tape if i == 0 else RngTape(mode="replay", values={k: v.clone() for k, v in tape.values.items()})
                results[name], initial, batches, rng = capture(cfg, initial, batches, use, rng)
                print(f"  seed {seed} {name}: loss {results[name]['loss']:.6f} tape_complete={results[name]['tape_complete']} peak {results[name]['peak_gb']:.1f}GB ({time.time() - t0:.0f}s)", flush=True)
        ref_vs_r0 = cmp(results["R0"], results[args.ref])
        seed_rep = {"loss": {k: v["loss"] for k, v in results.items()}, "peak_gb": {k: v["peak_gb"] for k, v in results.items()}, "batch_sum": results["R0"]["batch_sum"],
                    "ref_vs_r0": {k: v for k, v in ref_vs_r0.items() if k not in ("per_param", "per_param_update")}, "cands": {}}
        for name in args.cands:
            c_r0, c_ref = cmp(results["R0"], results[name]), cmp(results[args.ref], results[name])
            # Amendment 4: parameters whose ref-vs-R0 cos is already < 0.9 are judged by the global metrics only; others recorded against cos(ref,R0) - 0.05
            off = {n: [round(ref_vs_r0["per_param"][n], 4), round(c_r0["per_param"][n], 4)] for n in c_r0["per_param"]
                   if ref_vs_r0["per_param"][n] >= 0.9 and c_r0["per_param"][n] < min(0.99, ref_vs_r0["per_param"][n] - 0.05) and c_ref["per_param"][n] < 0.99}
            ok_seed = (c_r0["loss_rel"] <= 2e-2 and c_r0["grad_cos_global"] >= 0.98 and c_ref["loss_rel"] <= 2e-2 and c_ref["grad_cos_global"] >= 0.98)
            seed_rep["cands"][name] = {
                "tape_complete": results[name]["tape_complete"], "seed_pass_loose": bool(ok_seed),
                "vs_r0": {k: v for k, v in c_r0.items() if k not in ("per_param", "per_param_update")},
                "vs_ref": {k: v for k, v in c_ref.items() if k not in ("per_param", "per_param_update")},
                "per_param_offenders_vs_r0": off,
                "per_param_cos_vs_r0": {k: round(v, 6) for k, v in c_r0["per_param"].items()},
                "per_param_cos_vs_ref": {k: round(v, 6) for k, v in c_ref["per_param"].items()},
            }
            r, f = c_r0, c_ref
            print(f"seed {seed} {name:12s} vsR0: loss_rel {r['loss_rel']:.2e} gcos {r['grad_cos_global']:.5f} nrel {r['grad_norm_rel']:.3f} ucos {r['update_cos_global']:.4f} "
                  f"| vs{args.ref}: loss_rel {f['loss_rel']:.2e} gcos {f['grad_cos_global']:.5f} nrel {f['grad_norm_rel']:.3f} pmin {f['grad_cos_min']:.4f} | offenders {len(off)}", flush=True)
        print(f"seed {seed} {args.ref} vs R0: loss_rel {ref_vs_r0['loss_rel']:.2e} gcos {ref_vs_r0['grad_cos_global']:.5f} pmin {ref_vs_r0['grad_cos_min']:.4f} nrel {ref_vs_r0['grad_norm_rel']:.3f}", flush=True)
        report["seeds"][str(seed)] = seed_rep
        report["summary"] = summarize(report, args.cands, args.ref)
        outp.write_text(json.dumps(report, indent=1, allow_nan=False))
        del results
        r16.free_gpu()
    report["summary"] = summarize(report, args.cands, args.ref)
    outp.write_text(json.dumps(report, indent=1, allow_nan=False))
    sums = {n: {t: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in s[t].items() if k != "grad_cos_each"} for t in ("vs_r0", "vs_ref")} | {"n": s["n_batches"], "pass": s["pass"]} for n, s in report["summary"].items()}
    print("SUMMARY", json.dumps(sums), flush=True)
    print("wrote", outp)


if __name__ == "__main__":
    main()
