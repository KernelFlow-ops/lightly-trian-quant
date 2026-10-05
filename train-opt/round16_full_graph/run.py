"""round15_optimizer_flat / round16_full_graph entrypoint (adapted from round17_mxfp8_sm120/run.py).  The same file lives in both rounds.

A composition is a comma separated list of tokens: a preset (P1, P1R13, P2 = common.config stacks), or ``name[:variant]`` with the
short names r1 r3 r4 r5 r7 r11 r13 r15 r17.  Example: ``--stack P2,r15`` = P2 (BF16 + r13 fused block + r17 MXFP8) + the flat optimizer.

Actions (every GPU action holds the dual lock):
    opt_bench    optimizer-only timing on the real parameter set (eager and CUDA-graph replay), one variant per process
    bench        0.2/0.6/0.2 weighted early/mid/late step timing (warmup 20, 50 x 3), --runner eager|graph
    profile      steady-state window for nsys (use common/profile_nsys.sh), --runner eager|graph
    gate_b       one real step vs --ref (default P1), several seeds (flat optimizer vs P1's optimizer)
    gate_s       50-step trajectories on 2 batches, graph vs eager (schedule / compressed schedule / fixed mid / fixed late)
    graph_equiv  graph vs eager on identical RNG (per-step loss, grad norm, update direction)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import statistics
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
TMP = Path("/root/autodl-tmp/train-opt-tmp") / ROUND.name
SHORT = {
    "r1": "round1_torch_cleanup", "r3": "round3_loss_matcher", "r4": "round4_attention",
    "r5": "round5_block_fusion", "r7": "round7_optimizer", "r11": "round11_bf16_mixed", "r13": "round13_gemm_fusion_sm120", "r17": "round17_mxfp8_sm120",
    "r15": "round15_optimizer_flat",
}
from common.config import P1_PATCH_KWARGS, P1_STACK, P1R13_PATCH_KWARGS, P1R13_STACK, P2_PATCH_KWARGS, P2_STACK  # noqa: E402
PRESET_CFG = {"P1": (P1_STACK, P1_PATCH_KWARGS), "P1R13": (P1R13_STACK, P1R13_PATCH_KWARGS), "P2": (P2_STACK, P2_PATCH_KWARGS), "R0": ((), {})}
OPT_R1 = "r1:nosync+mask_skip+adamw_fused"
OPT_VARIANTS = {   # optimizer-only benchmark compositions (the model/forward patches are irrelevant to the optimizer)
    "torch_fused": OPT_R1,
    "r7_capturable": OPT_R1 + ",r7:capturable_torch",
    "r7_multi": OPT_R1 + ",r7:cuda_multi_tensor",
    "flat": OPT_R1 + ",r15",
}
DEFAULT_KW = {**DEFAULT_PATCH_KWARGS, "round11_bf16_mixed": {"variant": "autocast"}, "round13_gemm_fusion_sm120": {"variant": "full"}, "round17_mxfp8_sm120": {"variant": "fc"}, "round15_optimizer_flat": {"variant": "flat"}}

def parse_stack(spec: str):
    spec = OPT_VARIANTS.get(spec, spec)
    patches: list[str] = []
    kwargs: dict[str, dict] = {}
    for item in [t.strip() for t in spec.split(",") if t.strip()]:
        if item in PRESET_CFG:
            stack, kw = PRESET_CFG[item]
            for name in stack:
                patches.append(name)
                kwargs[name] = dict(kw.get(name, {}))
            continue
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


def wait_exclusive(timeout: float = 90.0) -> None:
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


# ------------------------------------------------------------------ helpers shared by bench / profile / gates
def to_cpu_batch(batch: dict) -> dict:
    out = dict(batch)
    out["image"] = batch["image"].cpu()
    out["binary_masks"] = [{"masks": b["masks"].cpu(), "labels": b["labels"].cpu()} for b in batch["binary_masks"]]
    return out


def make_runner(ctx, batches, args):
    from round16_full_graph.impl.graph_step import StepGraphRunner
    return StepGraphRunner(ctx, batches[0], max_targets=args.max_targets, max_resident=args.max_resident,
                           metrics=args.metrics, side_stream=not args.no_side_stream)


def make_step(ctx, batches, args):
    """(step_fn, runner|None) for --runner eager|graph; the graph runner warms every signature it can visit."""
    if args.runner == "eager":
        return None, None
    runner = make_runner(ctx, batches, args)
    runner.prepare(batches[0])
    return runner.as_step_fn(batches), runner


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
        if args.cpu_batches:
            batches = [to_cpu_batch(b) for b in batches]
        build_s = time.time() - t0
        step_fn, runner = make_step(ctx, batches, args)
        graph_prep_s = time.time() - t0 - build_s
        measured = bench.time_steps(ctx, batches, phases=phases, warmup=args.warmup, iters=args.iters, repeats=args.repeats,
                                    segments=(args.runner == "eager" and not args.no_segments), step_fn=step_fn)
        env = bench.env_info()
        graph_report = runner.report.as_dict() if runner is not None else None
        if runner is not None:
            runner.sync_probs_state()
    name = args.name or f"{args.stack}|{args.runner}"
    result = bench.empty_result(
        round_name=ROUND.name, variant=name, model=args.model, batch=int(cfg.batch_size), precision=cfg.precision,
        patches=cfg.patches, desc=f"stack={args.stack}; runner={args.runner}; patch_kwargs={cfg.patch_kwargs}",
    )
    for phase in phases:
        result["gpu_step"][phase] = {"step_ms": measured[phase]["step_ms"], "img_s": measured[phase]["img_s"]}
    result["gpu_step"]["weighted_img_s"] = measured["weighted_img_s"]
    if "mid" in phases and args.runner == "eager" and not args.no_segments:
        result["breakdown_ms"].update(measured["mid"]["breakdown_ms"])
    result["peak_mem_gb"] = max(measured[p]["peak_allocated_gb"] for p in phases)
    result["env"] = env
    if graph_report is not None:
        result["graph"] = {**graph_report, "graph_prep_s": graph_prep_s, "max_targets": args.max_targets, "cpu_batches": args.cpu_batches,
                           "metrics": args.metrics, "side_stream": not args.no_side_stream}
    summary = {
        "model": args.model, "name": name, "stack": args.stack, "runner": args.runner, "build_s": build_s,
        "step_ms_mean": {p: measured[p]["step_ms"]["mean"] for p in phases},
        "img_s": {p: measured[p]["img_s"] for p in phases}, "weighted_img_s": measured["weighted_img_s"],
        "peak_allocated_gb": {p: measured[p]["peak_allocated_gb"] for p in phases},
        "repeat_ms_mean": {p: [r["step_ms"]["mean"] for r in measured[p]["repeats"]] for p in phases},
        "graph": result.get("graph"),
    }
    out = unique_path(ROUND / "profile" / f"bench_{args.model}_{safe_name(name)}_{args.phases}.json")
    out.write_text(json.dumps({"summary": summary, "measurement": measured, "command": [sys.executable, *sys.argv]}, indent=2, default=str, allow_nan=False) + "\n")
    if not args.no_results:
        bench.write_results(ROUND / "results.json", {args.model: {name: result}})
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)


# ------------------------------------------------------------------ profile
def action_profile(args) -> None:
    cfg = make_cfg(args.model, args.stack, seed=args.seed, phase=args.phases)
    with bench.gpu_lock():
        wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        batches = preload_batches(ctx, k=args.preload)
        if args.cpu_batches:
            batches = [to_cpu_batch(b) for b in batches]
        ctx.meta["phase_override"] = True
        set_phase(ctx, args.phases)
        step_fn, runner = make_step(ctx, batches, args)
        ctx.nvtx = args.runner == "eager"
        if step_fn is None:
            bench.profile_steps(lambda i: train_step(ctx, batches[i % len(batches)], i), warmup=args.warmup, steps=5)
        else:
            bench.profile_steps(lambda i: step_fn(ctx, batches[i % len(batches)], i), warmup=args.warmup, steps=5)
        if runner is not None:
            print("graph report", json.dumps(runner.report.as_dict(), default=str), flush=True)


# ------------------------------------------------------------------ optimizer-only benchmark
def action_opt_bench(args) -> None:
    variant = args.stack if args.stack in OPT_VARIANTS else args.name or args.stack
    cfg = make_cfg(args.model, args.stack, seed=args.seed)
    with bench.gpu_lock():
        wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        opt = getattr(ctx.optimizer, "optimizer", ctx.optimizer)
        params = [p for g in opt.param_groups for p in g["params"] if p.requires_grad]
        torch.manual_seed(0)
        templates = [torch.randn_like(p.detach()) * 0.002 for p in params]

        def refill():
            for p, t in zip(params, templates):
                if p.grad is None:
                    p.grad = t.clone()
                else:
                    p.grad.copy_(t)

        flat = ctx.meta.get("round15_optimizer_flat")
        for _ in range(10):
            refill()
            ctx.ops.opt_step(ctx)
        torch.cuda.synchronize()

        def timed(n):
            ms = []
            for _ in range(n):
                refill()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                ctx.ops.opt_step(ctx)
                e.record()
                ms.append((s, e))
            torch.cuda.synchronize()
            return [a.elapsed_time(b) for a, b in ms]

        eager = [timed(args.iters) for _ in range(args.repeats)]
        eager_flat = sorted(sum(eager, []))
        res = {"variant": variant, "model": args.model, "n_params": len(params), "numel": sum(p.numel() for p in params),
               "eager_ms": {"median": statistics.median(eager_flat), "mean": statistics.fmean(eager_flat), "min": eager_flat[0],
                            "per_repeat_median": [statistics.median(r) for r in eager]}}
        # CUDA-graph replay of the optimizer step alone (pure GPU time)
        graph_ms, graph_err = None, None
        if variant in ("flat", "r7_multi", "r7_capturable"):
            try:
                ctx.meta["round7_graph_capture"] = True
                if flat is not None:
                    flat.graph_mode = True
                refill()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    ctx.ops.opt_step(ctx)
                torch.cuda.synchronize()
                for _ in range(10):
                    g.replay()
                torch.cuda.synchronize()
                ms = []
                for _ in range(args.repeats * args.iters):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    g.replay()
                    e.record()
                    ms.append((s, e))
                torch.cuda.synchronize()
                gm = sorted(a.elapsed_time(b) for a, b in ms)
                graph_ms = {"median": statistics.median(gm), "mean": statistics.fmean(gm), "min": gm[0]}
            except Exception as exc:
                graph_err = f"{type(exc).__name__}: {str(exc)[:240]}"
        res["graph_replay_ms"] = graph_ms
        res["graph_capture_error"] = graph_err
        if flat is not None and graph_ms is not None:
            f = flat.flat

            def ev(fn, n=60):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                pairs = []
                for _ in range(n):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn()
                    e.record()
                    pairs.append((s, e))
                torch.cuda.synchronize()
                return statistics.median(a.elapsed_time(b) for a, b in pairs)

            k1 = ev(lambda: flat.ext.sumsq_prep(f.grad, flat.partial, flat.ticket, flat.ctl, flat.hyp, flat.ext_lr, flat.eff, flat.norm, flat.clip,
                                                flat.grid, flat.max_norm, flat.nw, flat.vw, flat.total, flat.power))
            k2 = ev(lambda: flat.ext.update(f.master, f.grad, f.m, f.v, flat.chunk_gid, flat.hyp, flat.eff, flat.clip))
            n = f.total
            res["kernels_ms"] = {"sumsq_prep": k1, "update": k2, "total_floats": n, "sumsq_GBs": n * 4 / k1 / 1e6, "update_GBs": n * 32 / k2 / 1e6}
    name = f"{args.model}_{variant}"
    out = unique_path(ROUND / "profile" / f"opt_bench_{name}.json")
    out.write_text(json.dumps({"result": res, "command": [sys.executable, *sys.argv]}, indent=2, default=str) + "\n")
    if not args.no_results:
        bench.write_results(ROUND / "results.json", {"opt_bench": {args.model: {variant: res}}})
    print(json.dumps(res, indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)


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
        "update_rel_l2": float(sum(((first["updates"][k].double() - second["updates"][k].double()) ** 2).sum() for k in first["updates"]) ** 0.5
                               / max(float(sum((first["updates"][k].double() ** 2).sum() for k in first["updates"]) ** 0.5), 1e-30)),
        "update_maxabs_diff": max(float((first["updates"][k] - second["updates"][k]).abs().max()) for k in first["updates"]),
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


# ------------------------------------------------------------------ scenarios (mask schedule handling) for gate S / graph_equiv
COMPRESSED = ([3, 9, 15], [9, 15, 21])     # annealing start / end per joint block: all 7 signatures within ~25 steps


def set_scenario_state(ctx, scen: str) -> None:
    """Mask probabilities / schedule mode for a scenario.  schedule + compressed follow the real (or compressed) schedule from step 0."""
    args = ctx.train_model.model_args
    if scen == "compressed":
        if "_orig_anneal" not in ctx.meta:
            ctx.meta["_orig_anneal"] = (list(args.attn_mask_annealing_steps_start), list(args.attn_mask_annealing_steps_end))
        args.attn_mask_annealing_steps_start, args.attn_mask_annealing_steps_end = list(COMPRESSED[0]), list(COMPRESSED[1])
    elif "_orig_anneal" in ctx.meta:
        args.attn_mask_annealing_steps_start, args.attn_mask_annealing_steps_end = ctx.meta.pop("_orig_anneal")
    if scen in ("schedule", "compressed"):
        ctx.meta.pop("phase_override", None)
        set_phase(ctx, "early")
    elif scen.startswith("fixed_"):
        ctx.meta["phase_override"] = True
        set_phase(ctx, scen[len("fixed_"):])
    else:
        raise ValueError(scen)


def curve_run(cfg: RunCfg, batches, steps: int, scen: str, mode: str, args) -> dict:
    ctx = build(cfg)
    apply_patches(ctx)
    ctx.rng = RngTape(mode="off")
    set_scenario_state(ctx, scen)
    runner = None
    if mode == "graph":
        runner = make_runner(ctx, batches, args)
        runner.prepare(batches[0])
        set_scenario_state(ctx, scen)
        runner.sync_probs_state()
    losses, norms, ema, value = [], [], [], None
    for step in range(steps):
        if runner is not None:
            out = runner.step(batches[step % len(batches)], step, batches[(step + 1) % len(batches)])
        else:
            out = train_step(ctx, batches[step % len(batches)], step)
        loss = float(out.loss.float().item())
        losses.append(loss)
        norms.append(None if out.grad_norm is None else float(out.grad_norm.float().item()))
        value = loss if value is None else 0.9 * value + 0.1 * loss
        ema.append(value)
    finite = all(math.isfinite(v) for v in losses) and all(v is None or math.isfinite(v) for v in norms)
    extra = {}
    if runner is not None:
        extra = {"graph": runner.report.as_dict()}
        runner.release()
    del ctx, runner
    free_gpu()
    return {"steps": steps, "loss": losses, "loss_ema": ema, "grad_norm": norms, "finite": finite, **extra}


def action_gate_s(args) -> None:
    cfg = make_cfg(args.model, args.stack, seed=0)
    report = {"command": [sys.executable, *sys.argv], "runs": {}}
    with bench.gpu_lock():
        wait_exclusive()
        loader_ctx = build(cfg)
        batches = preload_batches(loader_ctx, k=2)
        del loader_ctx
        free_gpu()
        for scen in args.scenarios:
            ref = curve_run(cfg, batches, args.steps, scen, "eager", args)
            cand = curve_run(cfg, batches, args.steps, scen, "graph", args)
            gate = gate_c(ref, cand, interval=10, max_rel=0.05)
            report["runs"][scen] = {"gate": gate, "ref": ref, "graph": cand}
            print(scen, json.dumps({"max_rel": gate["curve300_max_rel"], "pass": gate["pass"], "finite": gate["finite"],
                                    "checkpoints": gate["checkpoints"], "grad_norm_ratios": gate["grad_norm_ratios"],
                                    "captures": cand.get("graph", {}).get("captures"), "capture_s": cand.get("graph", {}).get("capture_s")}), flush=True)
    out = unique_path(ROUND / "profile" / f"gateS_{args.model}_{safe_name(args.stack)}_graph_vs_eager.json")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")


# ------------------------------------------------------------------ graph vs eager on identical RNG
def _snapshot(ctx, flat) -> dict:
    f = flat.flat
    return {"master": f.master.clone(), "m": f.m.clone(), "v": f.v.clone(), "ctl": flat.ctl.clone()}


def _restore(flat, snap) -> None:
    f = flat.flat
    f.master.copy_(snap["master"])
    f.m.copy_(snap["m"])
    f.v.copy_(snap["v"])
    flat.ctl.copy_(snap["ctl"])
    f.grad.zero_()


def _delta_stats(a: torch.Tensor, b: torch.Tensor, init: torch.Tensor) -> dict:
    """Update statistics of two parameter snapshots against ``init`` (CPU tensors, float64 accumulation in chunks)."""
    dd = aa = bb = ab = 0.0
    maxabs = 0.0
    for i in range(0, a.numel(), 1 << 25):
        da, db = (a[i:i + (1 << 25)] - init[i:i + (1 << 25)]).double(), (b[i:i + (1 << 25)] - init[i:i + (1 << 25)]).double()
        dd += float(((da - db) ** 2).sum())
        aa += float((da * da).sum())
        bb += float((db * db).sum())
        ab += float((da * db).sum())
        maxabs = max(maxabs, float((a[i:i + (1 << 25)] - b[i:i + (1 << 25)]).abs().max()))
    na, nb = aa ** 0.5, bb ** 0.5
    return {"update_rel_l2": dd ** 0.5 / max(na, 1e-30), "update_cos": ab / max(na * nb, 1e-30), "param_maxabs": maxabs, "bitwise_equal": bool(torch.equal(a, b))}


def action_graph_equiv(args) -> None:
    from round16_full_graph.impl.graph_step import padded_targets
    cfg = make_cfg(args.model, args.stack, seed=args.seed)
    results: dict = {"command": [sys.executable, *sys.argv], "scenarios": {}}
    with bench.gpu_lock():
        wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        ctx.rng = RngTape(mode="off")
        batches = preload_batches(ctx, k=2)
        flat = ctx.meta["round15_optimizer_flat"]
        padded = [{"image": b["image"], "binary_masks": padded_targets(b, args.max_targets)} for b in batches]
        init = _snapshot(ctx, flat)
        for scen in args.scenarios:
            K = args.equiv_steps
            row: dict = {}

            def run_eager():
                _restore(flat, init)
                set_scenario_state(ctx, scen)
                ls, gs = [], []
                for k in range(K):
                    torch.cuda.manual_seed(7000 + k)
                    o = train_step(ctx, padded[k % 2], k)
                    ls.append(o.loss.clone())
                    gs.append(o.grad_norm.clone())
                torch.cuda.synchronize()
                return [float(x) for x in ls], [float(x) for x in gs], flat.flat.master.cpu()

            e1 = run_eager()
            e2 = run_eager()
            free_gpu()
            runner = make_runner(ctx, batches, args)
            _restore(flat, init)
            set_scenario_state(ctx, scen)
            runner.prepare(batches[0])
            _restore(flat, init)
            set_scenario_state(ctx, scen)
            runner.sync_probs_state()
            gl, gg = [], []
            for k in range(K):
                torch.cuda.manual_seed(7000 + k)
                o = runner.step(batches[k % 2], k, batches[(k + 1) % 2])
                gl.append(o.loss.clone())
                gg.append(o.grad_norm.clone())
            torch.cuda.synchronize()
            gl, gg = [float(x) for x in gl], [float(x) for x in gg]
            gmaster = flat.flat.master.cpu()
            row["steps"] = K
            init_cpu = init["master"].cpu()
            row["eager_vs_eager"] = _delta_stats(e1[2], e2[2], init_cpu)
            row["graph_vs_eager"] = _delta_stats(gmaster, e1[2], init_cpu)
            row["loss_eager"] = e1[0]
            row["loss_eager2"] = e2[0]
            row["loss_graph"] = gl
            row["gnorm_eager"] = e1[1]
            row["gnorm_graph"] = gg
            row["loss_rel_max_graph_vs_eager"] = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(gl, e1[0]))
            row["loss_rel_max_eager_vs_eager"] = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(e2[0], e1[0]))
            row["gnorm_rel_max_graph_vs_eager"] = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(gg, e1[1]))
            row["loss_bitwise_first_step"] = bool(gl[0] == e1[0][0])
            row["graph"] = runner.report.as_dict()
            runner.release()
            del runner
            free_gpu()
            results["scenarios"][scen] = row
            print(scen, json.dumps({k: row[k] for k in ("steps", "eager_vs_eager", "graph_vs_eager", "loss_rel_max_graph_vs_eager", "loss_rel_max_eager_vs_eager",
                                                          "gnorm_rel_max_graph_vs_eager", "loss_bitwise_first_step")}), flush=True)
            print("  captures", row["graph"]["captures"], "capture_s", [round(x, 2) for x in row["graph"]["capture_s"]], "keys", row["graph"]["keys"], flush=True)
    out = unique_path(ROUND / "profile" / f"graph_equiv_{args.model}_{safe_name(args.stack)}.json")
    out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(B_AUTO), required=True)
    parser.add_argument("--stack", default="P2,r15", help="preset / opt-variant name or comma separated patch spec")
    parser.add_argument("--name", help="variant key for results.json (default: stack|runner)")
    parser.add_argument("--action", choices=("opt_bench", "bench", "profile", "gate_b", "gate_s", "graph_equiv"), required=True)
    parser.add_argument("--runner", choices=("eager", "graph"), default="eager")
    parser.add_argument("--phases", default="all", help="bench: all|early|mid|late; profile/gate_b: one phase (default mid)")
    parser.add_argument("--ref", default="P1", help="gate_b reference stack")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision-change", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scenarios", nargs="*", default=["schedule", "compressed", "fixed_mid", "fixed_late"])
    parser.add_argument("--equiv-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preload", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=2)
    parser.add_argument("--max-resident", type=int, default=8)
    parser.add_argument("--metrics", choices=("step", "lazy", "off"), default="lazy", help="graph runner: train_metrics update per step (host sync!), lazily on update_metrics(), or never")
    parser.add_argument("--no-metrics", action="store_true", help="alias of --metrics off")
    parser.add_argument("--no-side-stream", action="store_true", help="graph runner: stage the next batch on the compute stream")
    parser.add_argument("--cpu-batches", action="store_true", help="keep the preloaded batches on the host (pinned H2D path)")
    parser.add_argument("--no-segments", action="store_true", help="eager bench without fwd/loss/bwd/opt event marks")
    parser.add_argument("--no-results", action="store_true")
    args = parser.parse_args()
    if args.no_metrics:
        args.metrics = "off"
    if args.action not in ("bench", "opt_bench") and args.phases == "all":
        args.phases = "mid"
    {"opt_bench": action_opt_bench, "bench": action_bench, "profile": action_profile, "gate_b": action_gate_b, "gate_s": action_gate_s,
     "graph_equiv": action_graph_equiv}[args.action](args)


if __name__ == "__main__":
    main()
