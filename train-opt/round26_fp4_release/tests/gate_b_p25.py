"""[round25 copy of round23_fp4_final/tests/gate_b_p3.py with more candidates: P3_hr16 / P3_hr32 / P3_hr64 (backbone dY headroom), P3_H2 / P3_H2f (head MXFP8 backward)]
R23 gate B (FP4 relaxed rule, CONTRACTS 11.5 / 13.4) for the COMBINED stack P3 (round21 FP4 head + round22 FP4 backbone on FINAL).  Adapted from round22_fp4_backbone/tests/gate_b_real.py.

One process per model and per draw.  Per seed (= one real batch): R0 (fp32 original step), P1/TB1 (torch bf16) and ONE context built from P3_STACK (FINAL + round21 + round22, both
patches with the empty policy = FINAL's ops) run one forward + loss + backward with the same initial weights / batch / RngTape replay.  The candidates are (head policy, backbone
policy) pairs applied to the two handles with ``set_policy`` (fresh FP4State each; first call exact init, then headroom x prev amax = what a graph-runner step sees):
    FINAL       ("", "")                     P3H     (head, "")        P3B  ("", backbone)        P3  (head, backbone)        P3_RN  (RN head, RN backbone)
Gradients are captured right before the optimizer (no step); cosines on the GPU in fp64.

Relaxed rule (separately vs R0 and vs P1): global grad cosine median >= 0.98, min >= 0.97, loss_rel <= 2e-2 (n >= 8).  Strict rule (median >= 0.99, min >= 0.98), vs FINAL, per-param
min recorded only.  Extra: group cosines of P3 vs P3H (head ConvT params: both FP4 head => should be ~1) and of P3 vs P3B (backbone blocks), a combined-stack consistency check.

  python tests/gate_b_p25.py --model vitl16 --seeds 0 1 2 3 4 5 6 7 --out profile/gateB_hr_vitl16_draw1.json  --cands P3 P3_hr16 P3_hr32 P3_exact P3_H2 [--per-param]
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

os.environ["QUACK_CACHE_DIR"] = "/root/autodl-tmp/train-opt-tmp/round25_fp4_harden/quack_cache"
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "tests"))

from round25_fp4_harden.impl import presets25 as p23  # noqa: E402


import torch  # noqa: E402

from common import bench  # noqa: E402
from common.check import _params, _rel  # noqa: E402
from common.config import BEST_STACK_V2_KWARGS, P3_PATCH_KWARGS_VITB, P3_PATCH_KWARGS_VITL, P3_STACK  # noqa: E402
from common.harness import RunCfg, apply_patches, build, preload_batches, set_phase, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round19_final import run as r19run  # noqa: E402,F401  (registers presets, redirects round16 outputs to r19: undone by register below)
from round16_full_graph import run as r16  # noqa: E402

R21, R22 = p23.R21, p23.R22
H3 = "head.f=mx8,head.dw=nv4+sr"
H3RN = "head.f=mx8,head.dw=nv4+rn"


def policies(model: str) -> dict:
    kw = P3_PATCH_KWARGS_VITL if model == "vitl16" else P3_PATCH_KWARGS_VITB
    assert kw[R21]["variant"] == H3
    bb = kw[R22]["variant"]
    return {"FINAL": ("", ""), "P3": (H3, bb), "P3H": (H3, ""), "P3B": ("", bb), "P3_RN": (H3RN, bb.replace("+sr", "+rn")), "FINAL_again": ("", ""),
            "P3_exact": (H3 + ",amax=exact", bb + "+amax=exact"), "P3_q8": (H3, bb + "+qkv.f=mx8"),
            "P3_hr16": (H3, bb + "+dyhr=16"), "P3_hr32": (H3, bb + "+dyhr=32"), "P3_hr64": (H3, bb + "+dyhr=64"),
            "P3_H2": ("head.dw=mx8", bb), "P3_H2f": ("head.f=mx8,head.dw=mx8", bb)}


def p3_cfg(model, seed, phase):
    return RunCfg(model=model, batch_size=None, phase=phase, patches=P3_STACK, patch_kwargs={**BEST_STACK_V2_KWARGS, R21: {"variant": ""}, R22: {"variant": ""}}, seed=seed, num_workers=0)


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
    """global / per-parameter / per-group gradient cosine of two {name: tensor|None} dicts (GPU, fp64)."""
    per, rows = {}, []
    for n, ga in a["grads"].items():
        gb = b["grads"].get(n)
        if ga is None or gb is None:
            per[n] = 1.0 if (ga is None and gb is None) else 0.0
            continue
        x, y = ga.reshape(-1).double(), gb.reshape(-1).double()
        rows.append((n, torch.stack([(x * y).sum(), (x * x).sum(), (y * y).sum()])))
    vals = torch.stack([v for _, v in rows]).cpu().tolist()
    tot, groups = [0.0, 0.0, 0.0], {}
    for (n, _), (d, xx, yy) in zip(rows, vals):
        per[n] = (d / math.sqrt(xx * yy)) if xx > 0 and yy > 0 else (1.0 if xx == 0 and yy == 0 else 0.0)
        g = groups.setdefault("blocks" if ".blocks." in n else ("head_convt" if "upscale" in n else "other"), [0.0, 0.0, 0.0])
        for i, v in enumerate((d, xx, yy)):
            tot[i] += v
            g[i] += v
    gcos = tot[0] / math.sqrt(tot[1] * tot[2]) if tot[1] > 0 and tot[2] > 0 else 1.0
    worst = min(per.items(), key=lambda kv: kv[1])
    gc = {k: (v[0] / math.sqrt(v[1] * v[2]) if v[1] > 0 and v[2] > 0 else 1.0) for k, v in groups.items()}
    return {"loss_a": a["loss"], "loss_b": b["loss"], "loss_rel": _rel(a["loss"], b["loss"]), "grad_cos_global": gcos, "grad_cos_min": worst[1], "grad_cos_min_name": worst[0],
            "grad_norm_a": math.sqrt(tot[1]), "grad_norm_b": math.sqrt(tot[2]), "grad_norm_rel": _rel(math.sqrt(tot[1]), math.sqrt(tot[2])), "group_cos": gc, "per_param": per}


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


from gate_b_p3_summary import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--cands", nargs="+", default=["P3", "P3H", "P3B"], help="names from policies(model) (FINAL and FINAL_again are always run)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--phase", default="mid")
    ap.add_argument("--per-param", action="store_true", help="store the per-parameter cosines vs R0 / P1 of every candidate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p23.register(ROUND, args.model)
    pol = policies(args.model)
    cands = {c: pol[c] for c in args.cands}
    mid = [c for c in cands if c not in ("FINAL", "FINAL_again")]
    mid = [c for c in ("P3H", "P3B") if c in mid] + [c for c in mid if c not in ("P3H", "P3B")]   # P3H / P3B first: P3 is compared against both (combined-stack consistency)
    run_names = ["FINAL", *mid, "FINAL_again"]
    outp = Path(args.out)
    report = {"command": [sys.executable, *sys.argv], "model": args.model, "phase": args.phase, "cands": {n: list(pol[n]) for n in run_names}, "seeds": {}}
    if outp.exists():
        old = json.loads(outp.read_text())
        report["seeds"] = old.get("seeds", {})
    lock_cm = bench.gpu_lock()   # whole-process lock: no CUDA context is ever alive outside the lock (CONTRACTS 6a 12.1)
    lock_cm.__enter__()
    r16.wait_exclusive()
    # discard-warm-up: the first FINAL / QuACK autotune of the process perturbs the first batch (R21b notes); build the P3 ctx once and run FINAL + P3 on seed 0's batch, drop the results
    for seed in args.seeds:
        have = report["seeds"].get(str(seed), {}).get("cands", {})
        todo = [n for n in run_names if n not in have]
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
        ctx = build(p3_cfg(args.model, seed, args.phase))
        apply_patches(ctx)
        ctx.model.load_state_dict(initial, strict=True)
        h21, h22 = ctx.meta[R21], ctx.meta[R22]
        print(f"  seed {seed} P3 ctx built ({time.time() - t0:.0f}s)", flush=True)
        seed_rep = report["seeds"].setdefault(str(seed), {"cands": {}})
        seed_rep.update(loss_r0=results["R0"]["loss"], loss_p1=results["TB1"]["loss"], batch_sum=results["R0"]["batch_sum"], peak_gb_ref={k: results[k]["peak_gb"] for k in ("R0", "TB1")})
        seed_rep["ref_p1_vs_r0"] = short(cmp_gpu(results["R0"], results["TB1"]))
        final, by_name = None, {}
        for name in run_names:
            tc = time.time()
            try:
                hp, bp = report["cands"][name]
                h21.set_policy(hp)
                h22.set_policy(bp)
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
            if name == "P3":                        # combined-stack consistency: head ConvT grads vs the head-only candidate, backbone block grads vs the backbone-only candidate
                for ref in ("P3H", "P3B"):
                    if ref in by_name:
                        rec[f"vs_{ref}"] = short(cmp_gpu(by_name[ref], res))
            if args.per_param:
                rec["per_param_cos_vs_r0"] = {k: round(v, 6) for k, v in c_r0["per_param"].items()}
                rec["per_param_cos_vs_p1"] = {k: round(v, 6) for k, v in c_p1["per_param"].items()}
            seed_rep["cands"][name] = rec
            print(f"seed {seed} {name:12s} vsR0 gcos {c_r0['grad_cos_global']:.5f} lrel {c_r0['loss_rel']:.1e} pmin {c_r0['grad_cos_min']:.3f} | vsP1 gcos {c_p1['grad_cos_global']:.5f} "
                  f"lrel {c_p1['loss_rel']:.1e} | vsFINAL {c_fn['grad_cos_global']:.5f} | {sec:.1f}s {res['peak_gb']:.1f}GB", flush=True)
            if name in ("P3H", "P3B") and "P3" in run_names:
                by_name[name] = res               # kept (GPU) until the P3 candidate has been compared; P3 runs after both (order of ``run_names``)
            else:
                del res
            del c_r0, c_p1, c_fn
            r16.free_gpu()
        h21.set_policy("")
        h22.set_policy("")
        h22.ctx = None  # Handle.ctx -> ctx -> model / optimizer: drop the cycle (the local names outlive this seed and would pin ~6 GB of ViT-L state)
        h21.wq = []
        del ctx, results, final, by_name, h21, h22
        r16.free_gpu()
        report["summary"] = summarize(report, list(report["cands"]))
        outp.write_text(json.dumps(report, indent=1, allow_nan=False))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    lock_cm.__exit__(None, None, None)
    report["summary"] = summarize(report, list(report["cands"]))
    outp.write_text(json.dumps(report, indent=1, allow_nan=False))
    for n, s in report["summary"].items():
        print(f"SUMMARY {n:12s} n={s['n_batches']} vsR0 med {s['vs_r0']['gcos_median']:.4f} min {s['vs_r0']['gcos_min']:.4f} (seed {s['vs_r0']['gcos_min_seed']}) | vsP1 med {s['vs_p1']['gcos_median']:.4f} "
              f"min {s['vs_p1']['gcos_min']:.4f} | lrel max {s['vs_r0']['loss_rel_max']:.1e} | relaxed {s['relaxed_pass_vs_r0_and_p1']} strict {s['strict_pass_vs_r0_and_p1']}", flush=True)
    print("wrote", outp)


if __name__ == "__main__":
    main()
