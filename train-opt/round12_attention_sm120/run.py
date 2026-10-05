"""round12_attention_sm120 entrypoint: step timing, gates B / S, nsys capture for the fused attention stack.

It reuses the helpers of round11_bf16_mixed/run.py (stack parsing, single-step B-lite comparison, phase-frozen curves) and writes only
into this round's directory.  Every GPU action holds the dual lock (common.bench.gpu_lock; re-entrant under gpu_run.sh / profile_nsys.sh).

Presets (comma separated patch specs, see round11_bf16_mixed/run.py):
  TB1   Agent A's T-BF16 stack: r1,r3,r5,r7 + r11:autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16   (the bar)
  R12   same without r11's attention variants, plus round12_attention_sm120:fused
  R12D  R12 with the device-probability mask builder (phase-agnostic graph mode)

    python round12_attention_sm120/run.py --model vitl16 --stack R12 --action bench
    python round12_attention_sm120/run.py --model vitl16 --stack R12 --ref TB1 --action gate_b
    python round12_attention_sm120/run.py --model vitl16 --stack R12 --ref TB1 --action gate_s
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT.parent / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import round11_bf16_mixed.run as r11run  # noqa: E402
from common import bench  # noqa: E402
from common.check import curve, gate_c  # noqa: E402
from common.config import B_AUTO  # noqa: E402
from common.harness import apply_patches, build, preload_batches, set_phase, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402

ROUND = Path(__file__).resolve().parent
BASE = "r1,r3,r5,r7"
PRESETS = {
    "TB1": f"{BASE},r11:autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16",
    "TB1_noln": f"{BASE},r11:autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn",
    "R12": f"{BASE},r11:autocast+head_ln_bf16,round12_attention_sm120:fused",
    "R12D": f"{BASE},r11:autocast+head_ln_bf16,round12_attention_sm120:fused+devprob",
    "R12_maskdense": f"{BASE},r11:autocast+head_ln_bf16,round12_attention_sm120:fused+maskdense",
}
r11run.PRESETS.update(PRESETS)


def unique_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cand, i = path, 1
    while cand.exists():
        i += 1
        cand = path.with_name(f"{path.stem}_{i}{path.suffix}")
    return cand


def action_bench(args) -> None:
    cfg = r11run.make_cfg(args.model, args.stack, seed=args.seed)
    phases = ("early", "mid", "late") if args.phases == "all" else (args.phases,)
    with bench.gpu_lock():
        r11run.wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        batches = preload_batches(ctx, k=args.preload)
        measured = bench.time_steps(ctx, batches, phases=phases, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        env = bench.env_info()
    res = bench.empty_result(round_name="round12_attention_sm120", variant=args.name or args.stack, model=args.model, batch=int(cfg.batch_size),
                             precision=cfg.precision, patches=cfg.patches, desc=f"stack={args.stack}; patch_kwargs={cfg.patch_kwargs}")
    for ph in phases:
        res["gpu_step"][ph] = {"step_ms": measured[ph]["step_ms"], "img_s": measured[ph]["img_s"]}
        res.setdefault("breakdown_by_phase", {})[ph] = measured[ph]["breakdown_ms"]
    res["gpu_step"]["weighted_img_s"] = measured["weighted_img_s"]
    if "mid" in phases:
        res["breakdown_ms"].update(measured["mid"]["breakdown_ms"])
    res["peak_mem_gb"] = max(measured[p]["peak_allocated_gb"] for p in phases)
    res["env"] = env
    res["command"] = [sys.executable, *sys.argv]
    out = unique_path(ROUND / "profile" / f"step_{args.model}_{args.name or r11run.safe_name(args.stack)}.json")
    out.write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    bench.write_results(ROUND / "results.json", {args.model: {args.name or args.stack: res}})
    line = {p: round(measured[p]["step_ms"]["mean"], 2) for p in phases}
    print(json.dumps({"model": args.model, "name": args.name or args.stack, "step_ms_mean": line, "weighted_img_s": measured["weighted_img_s"], "peak_gb": res["peak_mem_gb"]}), flush=True)


def action_profile(args) -> None:
    cfg = r11run.make_cfg(args.model, args.stack, seed=args.seed, phase=args.phases)
    with bench.gpu_lock():
        r11run.wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        batches = preload_batches(ctx, k=args.preload)
        ctx.nvtx = True
        ctx.meta["phase_override"] = True
        set_phase(ctx, args.phases)
        bench.profile_steps(lambda i: train_step(ctx, batches[i % len(batches)], i), warmup=args.warmup, steps=5)


def action_gate_b(args) -> None:
    results = []
    for seed in args.seeds:
        t0 = time.time()
        cfg_ref = r11run.make_cfg(args.model, args.ref, seed=seed, phase=args.phases)
        cfg_opt = r11run.make_cfg(args.model, args.stack, seed=seed, phase=args.phases)
        with bench.gpu_lock():
            r11run.wait_exclusive()
            r = r11run.b_lite_once(cfg_ref, cfg_opt, RngTape(mode="record"))
        r.update(seed=seed, phase=args.phases, ref=args.ref, stack=args.stack, seconds=time.time() - t0)
        r["pass"] = r["loss_rel"] <= 1e-2 and r["grad_cos_min"] >= 0.99 and r["grad_norm_rel"] <= 0.02 and r["update_cos_min"] >= 0.99
        r["criterion"] = "check.py: loss_rel<=1e-2, grad_cos_min>=0.99, grad_norm_rel<=0.02, update_cos_min>=0.99"
        results.append(r)
        short = {k: r[k] for k in ("seed", "loss_rel", "grad_cos_global", "grad_cos_min", "grad_norm_rel", "update_cos_global", "update_cos_min", "pass")}
        print(json.dumps(short), flush=True)
        print("  worst per-param grad cos:", r["grad_cos_worst5"], flush=True)
    out = unique_path(ROUND / "profile" / f"gateB_{args.model}_{r11run.safe_name(args.stack)}_vs_{r11run.safe_name(args.ref)}_{args.phases}.json")
    out.write_text(json.dumps({"command": [sys.executable, *sys.argv], "results": results}, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")


def action_gate_s(args) -> None:
    cfg_ref = r11run.make_cfg(args.model, args.ref, seed=0)
    cfg_opt = r11run.make_cfg(args.model, args.stack, seed=0)
    report = {"command": [sys.executable, *sys.argv], "runs": {}}
    with bench.gpu_lock():
        r11run.wait_exclusive()
        loader_ctx = build(cfg_ref)
        batches = preload_batches(loader_ctx, k=2)
        del loader_ctx
        r11run.free_gpu()
        runs = {"schedule": lambda cfg: curve(cfg, batches, steps=args.steps)}
        for phase in args.s_phases:
            runs[f"fixed_{phase}"] = lambda cfg, ph=phase: r11run.curve_phase(cfg, batches, args.steps, ph)
        for key, runner in runs.items():
            ref_curve = runner(cfg_ref)
            r11run.free_gpu()
            opt_curve = runner(cfg_opt)
            r11run.free_gpu()
            gate = gate_c(ref_curve, opt_curve, interval=10, max_rel=0.05)
            report["runs"][key] = {"gate": gate, "ref_loss_ema": ref_curve["loss_ema"], "opt_loss_ema": opt_curve["loss_ema"], "ref_finite": ref_curve["finite"], "opt_finite": opt_curve["finite"]}
            print(key, json.dumps({"max_rel": gate["curve300_max_rel"], "pass": gate["pass"], "finite": gate["finite"], "checkpoints": gate["checkpoints"], "grad_norm_ratios": gate["grad_norm_ratios"]}), flush=True)
    out = unique_path(ROUND / "profile" / f"gateS_{args.model}_{r11run.safe_name(args.stack)}_vs_{r11run.safe_name(args.ref)}.json")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=tuple(B_AUTO), required=True)
    p.add_argument("--stack", default="R12")
    p.add_argument("--name")
    p.add_argument("--action", choices=("bench", "profile", "gate_b", "gate_s"), required=True)
    p.add_argument("--phases", default="all", help="bench: all|early|mid|late; profile/gate_b: one phase (default mid)")
    p.add_argument("--ref", default="TB1")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--s-phases", nargs="*", default=["mid"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--preload", type=int, default=1)
    args = p.parse_args()
    if args.action in ("profile", "gate_b") and args.phases == "all":
        args.phases = "mid"
    {"bench": action_bench, "profile": action_profile, "gate_b": action_gate_b, "gate_s": action_gate_s}[args.action](args)


if __name__ == "__main__":
    main()
