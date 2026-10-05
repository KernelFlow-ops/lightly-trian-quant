"""R21b gate B (FP4 relaxed rule, CONTRACTS 11.5 / 6b 13.4) for the REAL-kernel head (round21 patch).  Adapted from round20_fp4_probe/tests/gate_b_multi.py.

One process per model.  Per seed (= one real batch; loader and init seeded with the seed): R0 (fp32 original step), P1/TB1 (torch bf16) and the round21 stack (FINAL + round21 patch,
built ONCE per seed) run one forward + loss + backward with the same initial weights / batch / RngTape replay; the round21 context then runs every candidate policy
(``handle.set_policy`` + head-room overrides) and the empty policy (= FINAL's head) before and after them.  Gradients are captured right before the optimizer (no optimizer step is taken);
cosines are computed on the GPU in fp64.  The delayed amax starts from the exact amax of the batch (first call), so a single step uses ts = headroom x exact amax (x2 for X / W, x8 for dZ).

Relaxed rule (separately vs R0 and vs P1): global grad cosine median >= 0.98, min >= 0.97, loss_rel <= 2e-2 (n >= 8 for a verdict).  Recorded only: strict rule (median >= 0.99, min >= 0.98), cosine vs FINAL.

  python tests/gate_b_real.py --model vitl16 --seeds 0 1 2 3 4 5 6 7 --cands H3 H1 H2 --out profile/gateB_r21b_vitl16.json
  candidates: H3 H1 H2 H3rn (RN) H3x (amax=exact) H3hr2 (dY headroom 2) | NAME=policy[@dy_headroom]
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

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND))

import torch  # noqa: E402

import run as r21run  # noqa: E402,F401  (registers the presets, redirects round16 / round19 outputs to round21)
from common import bench  # noqa: E402
from common.check import _params, _rel  # noqa: E402
from common.config import BEST_STACK_V2, BEST_STACK_V2_KWARGS  # noqa: E402
from common.harness import RunCfg, apply_patches, build, preload_batches, set_phase, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round16_full_graph import run as r16  # noqa: E402

R21 = "round21_fp4_quant_head"
H3 = "head.f=mx8,head.dw=nv4+sr"
CANDS = {
    "H3": (H3, {}), "H1": ("head.dw=nv4+sr", {}), "H2": ("head.dw=mx8", {}),
    "H3rn": ("head.f=mx8,head.dw=nv4+rn", {}), "H3x": (H3 + ",amax=exact", {}), "H3hr2": (H3, {"dy_headroom": 2.0}),
    "H1rn": ("head.dw=nv4+rn", {}), "H3f": ("head.f=nv4,head.dw=nv4+sr", {}),
}


def r21_cfg(model, seed, phase):
    return RunCfg(model=model, batch_size=None, phase=phase, patches=BEST_STACK_V2 + (R21,), patch_kwargs={**BEST_STACK_V2_KWARGS, R21: {"variant": ""}}, seed=seed, num_workers=0)


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
    head = {k: v for k, v in per.items() if "upscale" in k}
    return {"loss_a": a["loss"], "loss_b": b["loss"], "loss_rel": _rel(a["loss"], b["loss"]), "grad_cos_global": gcos, "grad_cos_min": worst[1], "grad_cos_min_name": worst[0],
            "head_grad_cos_min": min(head.values()) if head else None, "grad_norm_a": math.sqrt(tot[1]), "grad_norm_b": math.sqrt(tot[2]),
            "grad_norm_rel": _rel(math.sqrt(tot[1]), math.sqrt(tot[2])), "per_param": per}


def run_ref(cfg, initial, batches, tape, rng):
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
        rows = [s["cands"][name] for s in report["seeds"].values() if name in s["cands"] and "vs_r0" in s["cands"][name]]
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
        o["relaxed_pass_vs_r0_and_p1"] = bool(o["vs_r0"]["relaxed_ok"] and o["vs_p1"]["relaxed_ok"])
        o["strict_pass_vs_r0_and_p1"] = bool(o["vs_r0"]["strict_ok"] and o["vs_p1"]["strict_ok"])
        o["relaxed_verdict_needs_n>=8"] = len(rows) < 8
        hm = [r["vs_r0"]["head_grad_cos_min"] for r in rows if r["vs_r0"].get("head_grad_cos_min") is not None]
        o["head_pmin_min"] = min(hm) if hm else None
        o["step_seconds_mean"] = statistics.fmean(r["seconds"] for r in rows)
        o["peak_gb_max"] = max(r["peak_gb"] for r in rows)
        out[name] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--phase", default="mid")
    ap.add_argument("--per-param", action="store_true")
    ap.add_argument("--redo", type=int, nargs="*", default=[], help="drop these seeds from the cached report and recompute them")
    ap.add_argument("--no-warmup", action="store_true", help="skip the throw-away warm-up pass (JIT / QuACK autotune of every policy) before the seed loop")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cands = {}
    for c in args.cands:
        if c in CANDS:
            cands[c] = CANDS[c]
        else:
            n, _, p = c.partition("=")
            pol, _, hr = p.partition("@")
            assert p, f"unknown candidate {c!r}"
            cands[n] = (pol, {"dy_headroom": float(hr)} if hr else {})
    run_names = ["FINAL", *cands, "FINAL_again"]
    spec = {"FINAL": ("", {}), **cands, "FINAL_again": ("", {})}
    outp = Path(args.out)
    report = {"command": [sys.executable, *sys.argv], "model": args.model, "phase": args.phase,
              "cands": {k: v[0] + (f"@dy_headroom={v[1]['dy_headroom']}" if v[1] else "") for k, v in spec.items()}, "seeds": {}}
    if outp.exists():
        old = json.loads(outp.read_text())
        report["seeds"] = old.get("seeds", {})
        report["cands"].update(old.get("cands", {}))
    for s in args.redo:
        report["seeds"].pop(str(s), None)
    lock_cm = bench.gpu_lock()
    lock_cm.__enter__()
    r16.wait_exclusive()
    if not args.no_warmup and any(str(s) not in report["seeds"] for s in args.seeds):
        # the first calls in a fresh process run QuACK's autotuner / JIT, which can leave the first gradients of a policy different from its later ones (seen on the FINAL run of
        # the first seed): run every policy once on a throw-away context and discard the results
        t0 = time.time()
        wctx = build(r21_cfg(args.model, args.seeds[0], args.phase))
        apply_patches(wctx)
        wh = wctx.meta[R21]
        wb = preload_batches(wctx, k=1)
        for name in run_names:
            pol, kw = spec[name]
            wh.set_policy(pol)
            wh.dy_headroom = float(kw.get("dy_headroom", 8.0))
            set_phase(wctx, args.phase)
            lean_capture(wctx, wb[0], RngTape(mode="off"), 0)
            torch.cuda.synchronize()
            r16.free_gpu()
        del wctx, wh, wb
        r16.free_gpu()
        print(f"warm-up pass done ({time.time() - t0:.0f}s)", flush=True)
    for seed in args.seeds:
        have = report["seeds"].get(str(seed), {}).get("cands", {})
        todo = [n for n in run_names if n not in have]
        if todo and "FINAL" in have:
            have = {k: v for k, v in have.items() if k != "FINAL"}
        if not todo:
            print(f"seed {seed} cached", flush=True)
            continue
        t0 = time.time()
        results, initial, batches, rng = {}, None, None, None
        tape = RngTape(mode="record")
        for i, name in enumerate(["R0", "TB1"]):
            cfg = r16.make_cfg(args.model, name, seed=seed, phase=args.phase)
            use = tape if i == 0 else RngTape(mode="replay", values={k: v.clone() for k, v in tape.values.items()})
            results[name], initial, batches, rng = run_ref(cfg, initial, batches, use, rng)
            print(f"  seed {seed} {name}: loss {results[name]['loss']:.6f} tape_complete={results[name]['tape_complete']} peak {results[name]['peak_gb']:.1f}GB ({time.time() - t0:.0f}s)", flush=True)
        ctx = build(r21_cfg(args.model, seed, args.phase))
        apply_patches(ctx)
        ctx.model.load_state_dict(initial, strict=True)
        handle = ctx.meta[R21]
        print(f"  seed {seed} r21 ctx built ({time.time() - t0:.0f}s)", flush=True)
        seed_rep = report["seeds"].setdefault(str(seed), {"cands": {}})
        seed_rep.update(loss_r0=results["R0"]["loss"], loss_p1=results["TB1"]["loss"], batch_sum=results["R0"]["batch_sum"], peak_gb_ref={k: results[k]["peak_gb"] for k in ("R0", "TB1")})
        seed_rep["ref_p1_vs_r0"] = short(cmp_gpu(results["R0"], results["TB1"]))
        final = None
        for name in run_names:
            if name in have:
                continue
            tc = time.time()
            try:
                pol, kw = spec[name]
                handle.set_policy(pol)
                handle.dy_headroom = float(kw.get("dy_headroom", 8.0))
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
            seed_rep["cands"][name] = rec
            print(f"seed {seed} {name:12s} vsR0 gcos {c_r0['grad_cos_global']:.5f} lrel {c_r0['loss_rel']:.1e} pmin {c_r0['grad_cos_min']:.3f} hmin {c_r0['head_grad_cos_min']:.3f} | vsP1 gcos {c_p1['grad_cos_global']:.5f} "
                  f"lrel {c_p1['loss_rel']:.1e} | vsFINAL {c_fn['grad_cos_global']:.5f} | {sec:.1f}s {res['peak_gb']:.1f}GB", flush=True)
            del res, c_r0, c_p1, c_fn
            r16.free_gpu()
        handle.set_policy("")
        del ctx, results, final, handle
        r16.free_gpu()
        report["summary"] = summarize(report, list(report["cands"]))
        outp.write_text(json.dumps(report, indent=1, allow_nan=False))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    lock_cm.__exit__(None, None, None)
    report["summary"] = summarize(report, list(report["cands"]))
    outp.write_text(json.dumps(report, indent=1, allow_nan=False))
    for n, s in report["summary"].items():
        print(f"SUMMARY {n:12s} n={s['n_batches']} vsR0 med {s['vs_r0']['gcos_median']:.4f} min {s['vs_r0']['gcos_min']:.4f} | vsP1 med {s['vs_p1']['gcos_median']:.4f} min {s['vs_p1']['gcos_min']:.4f} "
              f"| lrel max {s['vs_r0']['loss_rel_max']:.1e} | relaxed {s['relaxed_pass_vs_r0_and_p1']} strict {s['strict_pass_vs_r0_and_p1']}", flush=True)
    print("wrote", outp)


if __name__ == "__main__":
    main()
