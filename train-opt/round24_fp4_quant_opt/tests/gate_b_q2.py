"""R24 gate B (copy of round22 tests/gate_b_real.py): R0 / P1 / FINAL / round22 real FP4 block with round21 ``fp4q`` (impl "real") / the SAME round22 block source with the quantizer
module swapped for round24 ``fp4q2`` (impl "real2", "real2p7" = Philox4x32-7 SR noise), all in ONE process on the same weights / batch / RngTape.  ``vs_real_r21a`` = gradient cosine of a
"real2" candidate against the "real" candidate of the same policy (fresh FP4State each -> same seed -> identical SR noise at 10 rounds).

Original round22 docstring.  Adapted from round20_fp4_probe/tests/gate_b_multi.py.

One process per model.  Per seed (= one real batch): R0 (fp32 original step), P1/TB1 (torch bf16) and ONE context with FINAL + round20 (simulated FP4 block) + round22
(real FP4 block) run one forward + loss + backward with the same initial weights / batch / RngTape replay.  Candidates are (impl, policy) pairs; ``sim`` = round20 block,
``real`` = round22 block, FINAL = round20 block with the empty policy (bitwise FINAL).  Gradients are captured right before the optimizer (no step: weights / shadows identical for
every candidate); cosines on the GPU in fp64.  Real candidates run a fresh FP4State per candidate (first call exact init, then headroom x prev amax = what a graph-runner step sees).

Relaxed rule (separately vs R0 and vs P1): global grad cosine median >= 0.98, min >= 0.97, loss_rel <= 2e-2 (n >= 8).  Strict rule + vs FINAL + per-param min recorded only.

  python tests/gate_b_real.py --model vitl16 --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_r22_vitl16.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["QUACK_CACHE_DIR"] = "/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/quack_cache"
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))

import torch  # noqa: E402

from common import bench  # noqa: E402
from common.check import _params, _rel  # noqa: E402
from common.config import BEST_STACK_V2, BEST_STACK_V2_KWARGS  # noqa: E402
from common.harness import RunCfg, apply_patches, build, preload_batches, set_phase, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round16_full_graph import run as r16  # noqa: E402
from round19_final.impl import presets  # noqa: E402

presets.register(ROUND)
R20, R22 = "round20_fp4_probe", "round22_fp4_backbone"


def load_block_q2():
    """round22's UNMODIFIED fp4_block22.py source executed as a second module in which ``fp4q`` is round24's fp4q2 (temporary sys.modules / package alias)."""
    import importlib
    import importlib.util

    from round24_fp4_quant_opt.impl import fp4q2

    pkg = importlib.import_module("round21_fp4_quant_head.impl")
    old_mod = sys.modules.get("round21_fp4_quant_head.impl.fp4q")
    old_attr = getattr(pkg, "fp4q", None)
    sys.modules["round21_fp4_quant_head.impl.fp4q"] = fp4q2
    pkg.fp4q = fp4q2
    try:
        spec = importlib.util.spec_from_file_location("round22_fp4_backbone.impl.fp4_block22_q2", str(ROUND.parent / "round22_fp4_backbone" / "impl" / "fp4_block22.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        assert mod.fp4q is fp4q2
    finally:
        if old_mod is None:
            sys.modules.pop("round21_fp4_quant_head.impl.fp4q", None)
        else:
            sys.modules["round21_fp4_quant_head.impl.fp4q"] = old_mod
        if old_attr is not None:
            pkg.fp4q = old_attr
    return mod, fp4q2
B3 = "all.d=nv4+sr+all.w=nv4+sr"
B4 = B3 + "+nofirst"
CANDS = {   # name -> (impl, policy)
    "FINAL": ("sim", ""),
    "SIM_B3": ("sim", B3), "REAL_B3": ("real", B3),
    "SIM_B4": ("sim", B4), "REAL_B4": ("real", B4),
    "REAL_B3rn": ("real", "all.d=nv4+rn+all.w=nv4+rn"), "REAL_B3x": ("real", B3 + "+amax=exact"), "REAL_B4x": ("real", B4 + "+amax=exact"),
    "SIM_B3rht": ("sim", "all.d=nv4+sr+all.w=nv4+sr+rht"), "SIM_B4rht": ("sim", B4.replace("all.w=nv4+sr", "all.w=nv4+sr+rht")),
    "REAL_B3nohr": ("real", B3 + "+dyhr=2"),
    "Q2_B3": ("real2", B3), "Q2_B4": ("real2", B4), "Q2p7_B3": ("real2p7", B3), "Q2p7_B4": ("real2p7", B4),
    "FINAL_again": ("sim", ""),
}


def r22_cfg(model, seed, phase):
    return RunCfg(model=model, batch_size=None, phase=phase, patches=BEST_STACK_V2 + (R20, R22),
                  patch_kwargs={**BEST_STACK_V2_KWARGS, R20: {"variant": ""}, R22: {"variant": ""}}, seed=seed, num_workers=0)


def lean_capture(ctx, batch, tape, step=0):
    """fwd + loss + bwd of ``train_step``; the gradients are cloned (GPU) right before the optimizer, which is replaced by a no-op, and then released."""
    ctx.rng = tape
    params = _params(ctx)
    grads: dict = {}
    original = ctx.ops.opt_step

    def capture(*a, **k):
        for n, p in params.items():
            grads[n] = None if p.grad is None else p.grad.detach().clone()
            p.grad = None
        return torch.zeros((), device="cuda")

    ctx.ops.opt_step = capture
    try:
        out = train_step(ctx, batch, step)
    finally:
        ctx.ops.opt_step = original
    loss = float(out.loss)
    complete = True
    if tape.mode == "replay":
        try:
            tape.assert_complete()
        except AssertionError as exc:
            complete = str(exc)[:200]
    return {"loss": loss, "grads": grads, "tape_complete": complete}


def cmp_gpu(a: dict, b: dict) -> dict:
    """global / per-parameter gradient cosine of two {name: tensor|None} dicts (GPU, fp64)."""
    per, rows = {}, []
    for n, ga in a["grads"].items():
        gb = b["grads"].get(n)
        if ga is None or gb is None:
            per[n] = 1.0 if (ga is None and gb is None) else 0.0
            continue
        x, y = ga.reshape(-1).double(), gb.reshape(-1).double()
        rows.append((n, torch.stack([(x * y).sum(), (x * x).sum(), (y * y).sum()])))
    vals = torch.stack([v for _, v in rows]).cpu().tolist()
    tot = [0.0, 0.0, 0.0]
    for (n, _), (d, xx, yy) in zip(rows, vals):
        per[n] = (d / math.sqrt(xx * yy)) if xx > 0 and yy > 0 else (1.0 if xx == 0 and yy == 0 else 0.0)
        tot[0] += d
        tot[1] += xx
        tot[2] += yy
    gcos = tot[0] / math.sqrt(tot[1] * tot[2]) if tot[1] > 0 and tot[2] > 0 else 1.0
    worst = min(per.items(), key=lambda kv: kv[1])
    return {"loss_a": a["loss"], "loss_b": b["loss"], "loss_rel": _rel(a["loss"], b["loss"]), "grad_cos_global": gcos, "grad_cos_min": worst[1], "grad_cos_min_name": worst[0],
            "grad_norm_a": math.sqrt(tot[1]), "grad_norm_b": math.sqrt(tot[2]), "grad_norm_rel": _rel(math.sqrt(tot[1]), math.sqrt(tot[2])), "per_param": per}


def run_ref(cfg, initial, batches, tape, rng):
    """R0 / P1: build, capture, free.  Returns (result, initial, batches, rng)."""
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
    res = lean_capture(ctx, batches[0], tape, 0)
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res["batch_sum"] = float(batches[0]["image"].double().sum())
    del ctx
    r16.free_gpu()
    return res, initial, batches, rng


def short(c):
    return {k: v for k, v in c.items() if k != "per_param"}


def summarize(report, names):
    out = {}
    for name in names:
        rows = [s["cands"][name] for s in report["seeds"].values() if name in s["cands"]]
        if not rows:
            continue
        o = {"n_batches": len(rows), "policy": report["cands"].get(name)}
        for tag in ("vs_r0", "vs_p1", "vs_final"):
            gc = [r[tag]["grad_cos_global"] for r in rows]
            mn = [r[tag]["grad_cos_min"] for r in rows]
            lr = [r[tag]["loss_rel"] for r in rows]
            o[tag] = {"gcos_median": statistics.median(gc), "gcos_min": min(gc), "gcos_max": max(gc), "pmin_min": min(mn), "loss_rel_max": max(lr),
                      "grad_norm_rel_max": max(r[tag]["grad_norm_rel"] for r in rows), "gcos_each": gc,
                      "relaxed_ok": bool(statistics.median(gc) >= 0.98 and min(gc) >= 0.97 and max(lr) <= 2e-2),
                      "strict_ok": bool(statistics.median(gc) >= 0.99 and min(gc) >= 0.98 and max(lr) <= 2e-2)}
        vr = [r["vs_real_r21a"] for r in rows if "vs_real_r21a" in r]
        if vr:
            o["vs_real_r21a"] = {"gcos_median": statistics.median([v["grad_cos_global"] for v in vr]), "gcos_min": min(v["grad_cos_global"] for v in vr), "pmin_min": min(v["grad_cos_min"] for v in vr),
                                 "loss_rel_max": max(v["loss_rel"] for v in vr), "grad_norm_rel_max": max(v["grad_norm_rel"] for v in vr), "n": len(vr)}
        o["relaxed_pass_vs_r0_and_p1"] = bool(o["vs_r0"]["relaxed_ok"] and o["vs_p1"]["relaxed_ok"])
        o["strict_pass_vs_r0_and_p1"] = bool(o["vs_r0"]["strict_ok"] and o["vs_p1"]["strict_ok"])
        o["relaxed_verdict_needs_n>=8"] = len(rows) < 8
        o["step_seconds_mean"] = statistics.fmean(r["seconds"] for r in rows)
        o["peak_gb_max"] = max(r["peak_gb"] for r in rows)
        out[name] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--cands", nargs="+", default=None, help="names from CANDS or NAME=impl:policy (impl sim|real)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--phase", default="mid")
    ap.add_argument("--per-param", action="store_true", help="store the per-parameter cosines vs R0 / P1 of every candidate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cands = {}
    for c in (args.cands or [n for n in CANDS if n not in ("FINAL", "FINAL_again")]):
        if c in CANDS:
            cands[c] = CANDS[c]
        else:
            n, _, p = c.partition("=")
            impl, _, pol = p.partition(":")
            assert impl in ("sim", "real", "real2", "real2p7"), c
            cands[n] = (impl, pol)
    run_names = ["FINAL", *cands, "FINAL_again"]
    outp = Path(args.out)
    report = {"command": [sys.executable, *sys.argv], "model": args.model, "phase": args.phase, "cands": {"FINAL": ["sim", ""], **{k: list(v) for k, v in cands.items()}, "FINAL_again": ["sim", ""]}, "seeds": {}}
    if outp.exists():
        old = json.loads(outp.read_text())
        report["seeds"] = old.get("seeds", {})
        report["cands"].update(old.get("cands", {}))
    lock_cm = bench.gpu_lock()   # whole-process lock: no CUDA context is ever alive outside the lock (CONTRACTS 6a 12.1)
    lock_cm.__enter__()
    r16.wait_exclusive()
    for seed in args.seeds:
        have = report["seeds"].get(str(seed), {}).get("cands", {})
        todo = [n for n in run_names if n not in have]
        if todo and "FINAL" in have:
            have = {k: v for k, v in have.items() if k != "FINAL"}  # FINAL grads are needed as the reference of vs_final: always rerun
        if not todo:
            print(f"seed {seed} cached", flush=True)
            continue
        t0 = time.time()
        results, initial, batches, rng = {}, None, None, None
        tape = RngTape(mode="record")
        if True:  # the dual GPU lock is held by the whole process (taken before the seed loop)
            for i, name in enumerate(["R0", "TB1"]):
                cfg = r16.make_cfg(args.model, name, seed=seed, phase=args.phase)
                use = tape if i == 0 else RngTape(mode="replay", values={k: v.clone() for k, v in tape.values.items()})
                results[name], initial, batches, rng = run_ref(cfg, initial, batches, use, rng)
                print(f"  seed {seed} {name}: loss {results[name]['loss']:.6f} tape_complete={results[name]['tape_complete']} peak {results[name]['peak_gb']:.1f}GB ({time.time() - t0:.0f}s)", flush=True)
            # round20 context (FINAL + simulated FP4), all candidates in it
            ctx = build(r22_cfg(args.model, seed, args.phase))
            apply_patches(ctx)
            ctx.model.load_state_dict(initial, strict=True)
            h20, h22 = ctx.meta[R20], ctx.meta[R22]
            blk22 = ctx.ops.block
            blk20 = blk22.previous
            h2 = blk2 = fp4q2 = None
            if any(report["cands"][n][0].startswith("real2") for n in run_names):
                mod2, fp4q2 = load_block_q2()
                fp4q2.WEIGHT_ROW = False  # as in the integrated run (round22 drops the weight row copy)
                h2 = mod2.Handle(ctx, "")
                blk2 = mod2.make_block(ctx, blk20, h2)
            real_ref = {}
            print(f"  seed {seed} r20 + r22 ctx built ({time.time() - t0:.0f}s)", flush=True)
            seed_rep = report["seeds"].setdefault(str(seed), {"cands": {}})
            seed_rep.update(loss_r0=results["R0"]["loss"], loss_p1=results["TB1"]["loss"], batch_sum=results["R0"]["batch_sum"], peak_gb_ref={k: results[k]["peak_gb"] for k in ("R0", "TB1")})
            seed_rep["ref_p1_vs_r0"] = short(cmp_gpu(results["R0"], results["TB1"]))
            final = None
            for name in run_names:
                if name in have:
                    continue
                tc = time.time()
                try:
                    impl, pol = report["cands"][name]
                    h20.set_policy(pol if impl == "sim" else "")
                    h22.set_policy(pol if impl == "real" else "")
                    if h2 is not None:
                        h2.set_policy(pol if impl.startswith("real2") else "")
                        fp4q2.PHILOX_ROUNDS = 7 if impl == "real2p7" else 10
                    ctx.ops.block = blk22 if impl == "real" else (blk2 if impl.startswith("real2") else blk20)
                    set_phase(ctx, args.phase)
                    torch.random.set_rng_state(rng[0])
                    torch.cuda.set_rng_state(rng[1])
                    torch.cuda.reset_peak_memory_stats()
                    res = lean_capture(ctx, batches[0], RngTape(mode="replay", values={k: v.clone() for k, v in tape.values.items()}), 0)
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    traceback.print_exc()
                    seed_rep["cands"][name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                    torch.cuda.synchronize()
                    r16.free_gpu()
                    continue
                torch.cuda.synchronize()
                sec = time.time() - tc
                res["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                if name == "FINAL":
                    final = res
                c_r0, c_p1 = cmp_gpu(results["R0"], res), cmp_gpu(results["TB1"], res)
                c_fn = cmp_gpu(final, res) if final is not None else c_r0
                rec = {"seconds": sec, "peak_gb": res["peak_gb"], "tape_complete": res["tape_complete"], "loss": res["loss"],
                       "vs_r0": short(c_r0), "vs_p1": short(c_p1), "vs_final": short(c_fn),
                       "seed_pass_relaxed": bool(c_r0["loss_rel"] <= 2e-2 and c_r0["grad_cos_global"] >= 0.98 and c_p1["loss_rel"] <= 2e-2 and c_p1["grad_cos_global"] >= 0.98)}
                if args.per_param:
                    rec["per_param_cos_vs_r0"] = {k: round(v, 6) for k, v in c_r0["per_param"].items()}
                    rec["per_param_cos_vs_p1"] = {k: round(v, 6) for k, v in c_p1["per_param"].items()}
                if impl == "real":
                    real_ref[pol] = res  # kept (grads) for the vs_real_r21a comparison of the real2 candidates of the same policy
                if impl.startswith("real2") and pol in real_ref:
                    rec["vs_real_r21a"] = short(cmp_gpu(real_ref[pol], res))
                seed_rep["cands"][name] = rec
                print(f"seed {seed} {name:14s} vsR0 gcos {c_r0['grad_cos_global']:.5f} lrel {c_r0['loss_rel']:.1e} pmin {c_r0['grad_cos_min']:.3f} | vsP1 gcos {c_p1['grad_cos_global']:.5f} "
                      f"lrel {c_p1['loss_rel']:.1e} | vsFINAL {c_fn['grad_cos_global']:.5f} | {sec:.1f}s {res['peak_gb']:.1f}GB", flush=True)
                del res, c_r0, c_p1, c_fn
                r16.free_gpu()
            h20.set_policy("")
            h22.set_policy("")
            if h2 is not None:
                h2.set_policy("")
                h2.ctx = None
                fp4q2.PHILOX_ROUNDS = 10
            real_ref.clear()
            h20.ctx = None  # Handle.ctx -> ctx -> model/optimizer: drop the cycle (the local names outlive this seed and would pin ~6 GB of ViT-L state)
            h22.ctx = None
            del ctx, results, final, h20, h22, blk20, blk22, h2, blk2
            r16.free_gpu()
        report["summary"] = summarize(report, list(report["cands"]))
        outp.write_text(json.dumps(report, indent=1, allow_nan=False))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    lock_cm.__exit__(None, None, None)
    report["summary"] = summarize(report, list(report["cands"]))
    outp.write_text(json.dumps(report, indent=1, allow_nan=False))
    for n, s in report["summary"].items():
        print(f"SUMMARY {n:14s} n={s['n_batches']} vsR0 med {s['vs_r0']['gcos_median']:.4f} min {s['vs_r0']['gcos_min']:.4f} | vsP1 med {s['vs_p1']['gcos_median']:.4f} min {s['vs_p1']['gcos_min']:.4f} "
              f"| lrel max {s['vs_r0']['loss_rel_max']:.1e} | relaxed {s['relaxed_pass_vs_r0_and_p1']} strict {s['strict_pass_vs_r0_and_p1']}", flush=True)
    print("wrote", outp)


if __name__ == "__main__":
    main()
