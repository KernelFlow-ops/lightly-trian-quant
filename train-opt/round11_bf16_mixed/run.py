"""round11_bf16_mixed entrypoint: step timing, gates, max-batch probe and nsys capture for any composition.

Every GPU action holds the dual lock (common.bench.gpu_lock; re-entrant under profile_nsys.sh/gpu_run.sh).

A composition is a comma separated list of patches, each optionally ``name:variant``:
    python run.py --model vitl16 --stack "r1,r3,r5,r7,r11:autocast+rope_triton+plain_flash" --action bench
Short names: r1 r3 r4 r5 r7 r11.  Named presets (see PRESETS) can be used instead of a spec.

Actions:
    bench     0.2/0.6/0.2 weighted early/mid/late step timing (warmup 20, 50 x 3 iterations), time_steps
    profile   steady-state capture window for nsys (use common/profile_nsys.sh)
    gate_b    B-lite single-step comparison against --ref (default R0 = fp32) for several seeds
    gate_s    50-step trajectory on 2 batches (check.curve + gate_c(interval=10)), plus fixed mid/late phases
    maxbatch  largest batch that fits (peak memory per batch size)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import bench  # noqa: E402
from common.check import _capture_step, _clone_tape, _cosine, _params, _rel, curve, gate_c  # noqa: E402
from common.config import B_AUTO, DEFAULT_PATCH_KWARGS  # noqa: E402
from common.harness import RunCfg, apply_patches, build, preload_batches, set_phase, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402

ROUND = Path(__file__).resolve().parent
TMP = Path("/root/autodl-tmp/train-opt-tmp/round11_bf16_mixed")
SHORT = {
    "r1": "round1_torch_cleanup", "r3": "round3_loss_matcher", "r4": "round4_attention",
    "r5": "round5_block_fusion", "r7": "round7_optimizer", "r11": "round11_bf16_mixed",
}
DEFAULT_KW = {**DEFAULT_PATCH_KWARGS, "round11_bf16_mixed": {"variant": "autocast"}}

# T-BF16 building blocks; BEST* are filled from measurements (see NOTES.md).
ATTN_BEST = "plain_flash+joint_qsplit+joint_cudnn"
PRESETS = {
    "R0": "",
    "P0": "r1,r3,r4,r5,r7",
    "AC": "r11:autocast",
    "AC_SDPA": f"r11:autocast+{ATTN_BEST}",
    "AC_SDPA_ROPE": f"r11:autocast+rope_triton+{ATTN_BEST}",
    "P0_AC": "r1,r3,r4,r5,r7,r11:autocast",
}


def parse_stack(spec: str):
    spec = PRESETS.get(spec, spec)
    patches: list[str] = []
    kwargs: dict[str, dict] = {}
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        name, _, variant = item.partition(":")
        full = SHORT.get(name, name)
        patches.append(full)
        kw = dict(DEFAULT_KW.get(full, {}))
        if variant:
            kw["variant"] = variant
        if kw:
            kwargs[full] = kw
    return tuple(patches), kwargs


def make_cfg(model: str, spec: str, *, seed: int = 0, phase: str = "mid", batch: int | None = None) -> RunCfg:
    patches, kwargs = parse_stack(spec)
    return RunCfg(model=model, batch_size=batch or B_AUTO[model], phase=phase, patches=patches, patch_kwargs=kwargs, seed=seed, num_workers=0)


def unique_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate, i = path, 1
    while candidate.exists():
        i += 1
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
    return candidate


def safe_name(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec)[:100]


def wait_exclusive(timeout: float = 400.0) -> None:
    """assert_gpu_exclusive with a grace period: the previous lock holder's CUDA context may still be tearing down."""
    deadline = time.time() + timeout
    while True:
        try:
            bench.assert_gpu_exclusive()
            return
        except RuntimeError:
            if time.time() > deadline:
                raise
            time.sleep(2.0)


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ bench
def action_bench(args) -> None:
    cfg = make_cfg(args.model, args.stack, seed=args.seed)
    phases = ("early", "mid", "late") if args.phases == "all" else (args.phases,)
    with bench.gpu_lock():
        wait_exclusive()
        t0 = time.time()
        ctx = build(cfg)
        apply_patches(ctx)
        batches = preload_batches(ctx, k=args.preload)
        build_s = time.time() - t0
        measured = bench.time_steps(ctx, batches, phases=phases, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        env = bench.env_info()
    name = args.name or args.stack
    result = bench.empty_result(
        round_name="round11_bf16_mixed", variant=name, model=args.model, batch=int(cfg.batch_size), precision=cfg.precision,
        patches=cfg.patches, desc=f"stack={args.stack}; patch_kwargs={cfg.patch_kwargs}",
    )
    for phase in phases:
        result["gpu_step"][phase] = {"step_ms": measured[phase]["step_ms"], "img_s": measured[phase]["img_s"]}
    result["gpu_step"]["weighted_img_s"] = measured["weighted_img_s"]
    if "mid" in phases:
        result["breakdown_ms"].update(measured["mid"]["breakdown_ms"])
    result["peak_mem_gb"] = max(measured[p]["peak_allocated_gb"] for p in phases)
    result["env"] = env
    summary = {
        "model": args.model, "name": name, "stack": args.stack, "patch_kwargs": cfg.patch_kwargs, "build_s": build_s,
        "step_ms_mean": {p: measured[p]["step_ms"]["mean"] for p in phases},
        "img_s": {p: measured[p]["img_s"] for p in phases}, "weighted_img_s": measured["weighted_img_s"],
        "breakdown_ms": {p: measured[p]["breakdown_ms"] for p in phases},
        "peak_allocated_gb": {p: measured[p]["peak_allocated_gb"] for p in phases},
    }
    out = unique_path(ROUND / "profile" / f"bench_{args.model}_{safe_name(name)}_{args.phases}.json")
    out.write_text(json.dumps({"summary": summary, "measurement": measured, "command": [sys.executable, *sys.argv]}, indent=2, default=str, allow_nan=False) + "\n")
    if not args.no_results:
        bench.write_results(ROUND / "results.json", {args.model: {name: result}})
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


# ------------------------------------------------------------------ profile
def action_profile(args) -> None:
    cfg = make_cfg(args.model, args.stack, seed=args.seed, phase=args.phases)
    with bench.gpu_lock():
        wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        batches = preload_batches(ctx, k=args.preload)
        ctx.nvtx = True
        ctx.meta["phase_override"] = True
        set_phase(ctx, args.phases)
        bench.profile_steps(lambda i: train_step(ctx, batches[i % len(batches)], i), warmup=args.warmup, steps=5)


# ------------------------------------------------------------------ gate B (single step vs reference)
def _group(name: str) -> str:
    match = re.search(r"backbone\.blocks\.(\d+)\.", name)
    if match:
        return f"block{int(match.group(1)):02d}"
    for key in ("backbone", "queries", "class_head", "mask_head", "upscale.0", "upscale.1"):
        if name.startswith(key) or f".{key}" in name:
            return key
    return "other"


def _accumulate(a: dict, b: dict) -> dict:
    """float64 dot/norm sums per parameter group from two {name: cpu tensor|None} dicts."""
    groups: dict[str, list[float]] = {}
    total = [0.0, 0.0, 0.0]
    per_param: dict[str, float] = {}
    for name, ga in a.items():
        gb = b.get(name)
        per_param[name] = _cosine(ga, gb)
        if ga is None or gb is None:
            continue
        x, y = ga.double().reshape(-1), gb.double().reshape(-1)
        dot, xx, yy = float(torch.dot(x, y)), float(torch.dot(x, x)), float(torch.dot(y, y))
        entry = groups.setdefault(_group(name), [0.0, 0.0, 0.0])
        entry[0] += dot
        entry[1] += xx
        entry[2] += yy
        total[0] += dot
        total[1] += xx
        total[2] += yy

    def cos(v):
        return v[0] / math.sqrt(v[1] * v[2]) if v[1] > 0 and v[2] > 0 else 1.0

    return {"global_cos": cos(total), "group_cos": {k: cos(v) for k, v in sorted(groups.items())}, "per_param_cos": per_param,
            "norm_a": math.sqrt(total[1]), "norm_b": math.sqrt(total[2])}


def b_lite_once(cfg_ref: RunCfg, cfg_opt: RunCfg, tape: RngTape) -> dict:
    """One update with identical weights/batch/RNG tape in both contexts (reference first, then candidate)."""
    ref = build(cfg_ref)
    apply_patches(ref)
    batches = preload_batches(ref, k=1)
    initial = {k: v.detach().cpu().clone() for k, v in ref.model.state_dict().items()}
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    ref_tape, opt_tape = _clone_tape(tape)
    first = _capture_step(ref, batches[0], ref_tape, 0)
    del ref
    free_gpu()
    opt = build(cfg_opt)
    apply_patches(opt)
    opt.model.load_state_dict(initial, strict=True)
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    if tape.mode == "record":
        opt_tape.values = {k: v.clone() for k, v in ref_tape.values.items()}
        opt_tape.mode = "replay"
    second = _capture_step(opt, batches[0], opt_tape, 0)
    opt_tape.assert_complete()
    del opt
    free_gpu()
    grads = _accumulate(first["grads"], second["grads"])
    updates = _accumulate(first["updates"], second["updates"])
    loss_ref, loss_opt = float(first["out"].loss), float(second["out"].loss)
    per = grads["per_param_cos"]
    worst = sorted(per.items(), key=lambda kv: kv[1])[:5]
    return {
        "loss_ref": loss_ref, "loss_opt": loss_opt, "loss_rel": _rel(loss_ref, loss_opt),
        "grad_cos_global": grads["global_cos"], "grad_cos_min": min(per.values()), "grad_cos_worst5": worst,
        "grad_norm_ref": grads["norm_a"], "grad_norm_opt": grads["norm_b"], "grad_norm_rel": _rel(grads["norm_a"], grads["norm_b"]),
        "update_cos_global": updates["global_cos"], "update_cos_min": min(updates["per_param_cos"].values()),
        "per_group_grad_cos": grads["group_cos"], "per_param_grad_cos": {k: round(v, 7) for k, v in per.items()},
    }


def action_gate_b(args) -> None:
    results = []
    for seed in args.seeds:
        t0 = time.time()
        cfg_ref = make_cfg(args.model, args.ref, seed=seed, phase=args.phases)
        cfg_opt = make_cfg(args.model, args.stack, seed=seed, phase=args.phases)
        with bench.gpu_lock():
            wait_exclusive()
            result = b_lite_once(cfg_ref, cfg_opt, RngTape(mode="record"))
        result.update(seed=seed, phase=args.phases, ref=args.ref, stack=args.stack, seconds=time.time() - t0)
        if args.precision_change:
            result["pass"] = result["loss_rel"] <= 2e-2 and result["grad_cos_global"] >= 0.99
            result["criterion"] = "loss_rel<=2e-2 and global grad cos>=0.99"
        else:
            result["pass"] = result["loss_rel"] <= 1e-2 and result["grad_cos_min"] >= 0.99 and result["grad_norm_rel"] <= 0.02 and result["update_cos_min"] >= 0.99
            result["criterion"] = "check.py: loss_rel<=1e-2, grad_cos_min>=0.99, grad_norm_rel<=0.02, update_cos_min>=0.99"
        results.append(result)
        short = {k: result[k] for k in ("seed", "loss_rel", "grad_cos_global", "grad_cos_min", "grad_norm_rel", "update_cos_global", "update_cos_min", "pass")}
        print(json.dumps(short), flush=True)
        print("  worst per-param grad cos:", result["grad_cos_worst5"], flush=True)
    out = unique_path(ROUND / "profile" / f"gateB_{args.model}_{safe_name(args.stack)}_vs_{safe_name(args.ref)}_{args.phases}.json")
    out.write_text(json.dumps({"command": [sys.executable, *sys.argv], "results": results}, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")


# ------------------------------------------------------------------ gate S (50-step trajectory)
def curve_phase(cfg: RunCfg, batches, steps: int, phase: str) -> dict:
    """check.curve with the mask probabilities frozen at ``phase`` (check.curve follows the real schedule)."""
    ctx = build(cfg)
    apply_patches(ctx)
    ctx.rng = RngTape(mode="off")
    ctx.meta["phase_override"] = True
    set_phase(ctx, phase)
    losses, norms, ema, value = [], [], [], None
    for step in range(steps):
        out = train_step(ctx, batches[step % len(batches)], step)
        loss = float(out.loss.float().item())
        losses.append(loss)
        norms.append(None if out.grad_norm is None else float(out.grad_norm.float().item()))
        value = loss if value is None else 0.9 * value + 0.1 * loss
        ema.append(value)
    finite = all(math.isfinite(v) for v in losses) and all(v is None or math.isfinite(v) for v in norms)
    del ctx
    free_gpu()
    return {"steps": steps, "loss": losses, "loss_ema": ema, "grad_norm": norms, "finite": finite}


def action_gate_s(args) -> None:
    cfg_ref = make_cfg(args.model, args.ref, seed=0)
    cfg_opt = make_cfg(args.model, args.stack, seed=0)
    report = {"command": [sys.executable, *sys.argv], "runs": {}}
    with bench.gpu_lock():
        wait_exclusive()
        loader_ctx = build(cfg_ref)
        batches = preload_batches(loader_ctx, k=2)
        del loader_ctx
        free_gpu()
        runs = {"schedule": lambda cfg: curve(cfg, batches, steps=args.steps)}
        for phase in args.s_phases:
            runs[f"fixed_{phase}"] = (lambda cfg, ph=phase: curve_phase(cfg, batches, args.steps, ph))
        for key, runner in runs.items():
            ref_curve = runner(cfg_ref)
            free_gpu()
            opt_curve = runner(cfg_opt)
            free_gpu()
            gate = gate_c(ref_curve, opt_curve, interval=10, max_rel=0.05)
            report["runs"][key] = {"gate": gate, "ref": ref_curve, "opt": opt_curve}
            print(key, json.dumps({"max_rel": gate["curve300_max_rel"], "pass": gate["pass"], "finite": gate["finite"], "checkpoints": gate["checkpoints"], "grad_norm_ratios": gate["grad_norm_ratios"]}), flush=True)
    out = unique_path(ROUND / "profile" / f"gateS_{args.model}_{safe_name(args.stack)}_vs_{safe_name(args.ref)}.json")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")


# ------------------------------------------------------------------ max batch
def action_maxbatch(args) -> None:
    cfg = make_cfg(args.model, args.stack, seed=args.seed)
    rows = []
    with bench.gpu_lock():
        wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        set_phase(ctx, args.phases)
        ctx.meta["phase_override"] = True
        for batch_size in args.batches:
            row = {"batch": batch_size}
            try:
                free_gpu()
                torch.cuda.reset_peak_memory_stats()
                ctx.meta["batch_size"] = batch_size
                batches = preload_batches(ctx, k=1)
                for i in range(3):
                    train_step(ctx, batches[0], i)
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for i in range(5):
                    train_step(ctx, batches[0], 3 + i)
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end) / 5
                row.update(ok=True, step_ms=ms, img_s=batch_size * 1000 / ms, peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                           peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30)
                del batches
            except torch.OutOfMemoryError as exc:
                row.update(ok=False, error=f"OutOfMemoryError: {str(exc).splitlines()[0][:200]}")
                rows.append(row)
                print(json.dumps(row), flush=True)
                break
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = unique_path(ROUND / "profile" / f"maxbatch_{args.model}_{safe_name(args.stack)}.json")
    out.write_text(json.dumps({"command": [sys.executable, *sys.argv], "stack": args.stack, "rows": rows}, indent=2) + "\n")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(B_AUTO), required=True)
    parser.add_argument("--stack", default="AC", help="preset name or comma separated patch spec")
    parser.add_argument("--name", help="variant key for results.json (default: the stack spec)")
    parser.add_argument("--action", choices=("bench", "profile", "gate_b", "gate_s", "maxbatch"), required=True)
    parser.add_argument("--phases", default="all", help="bench: all|early|mid|late; profile/gates/maxbatch: one phase (default mid)")
    parser.add_argument("--ref", default="R0", help="gate reference stack (default fp32 R0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision-change", action="store_true", help="gate B with the precision-change criterion (loss_rel<=2e-2, global grad cos>=0.99)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--s-phases", nargs="*", default=["mid", "late"])
    parser.add_argument("--batches", type=int, nargs="+", default=[7, 8, 10, 12, 14, 16, 20, 24])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preload", type=int, default=1)
    parser.add_argument("--no-results", action="store_true")
    args = parser.parse_args()
    if args.action != "bench" and args.phases == "all":
        args.phases = "mid"
    {"bench": action_bench, "profile": action_profile, "gate_b": action_gate_b, "gate_s": action_gate_s, "maxbatch": action_maxbatch}[args.action](args)


if __name__ == "__main__":
    main()
