"""Run repeatable round0 harness measurements and short nsys captures."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench
from common.config import B_AUTO, MODEL_NAMES, PHASE_WEIGHTS
from common.harness import RunCfg, apply_patches, build, make_loader, preload_batches, set_phase, train_step

RESULTS = Path(__file__).with_name("results.json")


def _write(model: str, variant: str, update: dict, batch: int, precision: str, log: str | None) -> None:
    existing = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if variant not in existing.get(model, {}):
        base = bench.empty_result(
            round_name="round0_baseline", variant=variant, model=model, batch=batch,
            precision=precision, patches=(),
            desc="Original Lightly training semantics; bf16-mixed is configured but single-GPU training_step executes fp32",
        )
        bench.write_results(RESULTS, {model: {variant: base}})
    if log:
        update["logs"] = [log]
    update["env"] = bench.env_info()
    bench.write_results(RESULTS, {model: {variant: update}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), required=True)
    parser.add_argument("--variant", choices=("B0-harness", "B0-fp32"), default="B0-harness")
    parser.add_argument("--phase", choices=("early", "mid", "late"), default="mid")
    parser.add_argument("--action", choices=("bench", "e2e", "profile"), required=True)
    parser.add_argument("--log")
    args = parser.parse_args()
    batch = B_AUTO[args.model]
    if batch is None:
        raise RuntimeError(f"B_AUTO for {args.model} is unset")
    precision = "32-true" if args.variant == "B0-fp32" else "bf16-mixed"
    workers = 25 if args.action == "e2e" else 4
    cfg = RunCfg(model=args.model, batch_size=batch, precision=precision, phase=args.phase, num_workers=workers)
    with bench.gpu_lock():
        bench.assert_gpu_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        if args.action in {"bench", "profile"}:
            batches = preload_batches(ctx, k=8)
        if args.action == "bench":
            measured = bench.time_steps(ctx, batches, phases=(args.phase,), warmup=20, iters=50, repeats=3)
            phase = measured[args.phase]
            phase_result = {"step_ms": phase["step_ms"], "img_s": phase["img_s"]}
            update = {
                "gpu_step": {args.phase: phase_result},
                "breakdown_ms": phase["breakdown_ms"],
                "peak_mem_gb": phase["peak_allocated_gb"],
            }
            previous = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
            previous_phases = previous.get(args.model, {}).get(args.variant, {}).get("gpu_step", {})
            means = {name: previous_phases.get(name, {}).get("step_ms", {}).get("mean") for name in PHASE_WEIGHTS}
            means[args.phase] = phase_result["step_ms"]["mean"]
            if all(value is not None for value in means.values()):
                weighted = sum(PHASE_WEIGHTS[name] * means[name] for name in PHASE_WEIGHTS)
                update["gpu_step"]["weighted_img_s"] = batch * 1000 / weighted
            _write(args.model, args.variant, update, batch, precision, args.log)
            print(json.dumps({"model": args.model, "variant": args.variant, "phase": args.phase, **phase_result}, indent=2))
        elif args.action == "e2e":
            loader = make_loader(ctx, "train")
            measured = bench.time_e2e(ctx, loader, steps=100)
            _write(args.model, args.variant, {"e2e": {"img_s": measured["img_s"], "dataload_wait_ms": measured["dataload_wait_ms"]}}, batch, precision, args.log)
            print(json.dumps(measured, indent=2))
        else:
            ctx.nvtx = True
            set_phase(ctx, args.phase)
            total_steps = ctx.meta["total_steps"]
            poly_power = ctx.train_model.model_args.poly_power
            mid_fraction = 0.4 + 0.2 * (1.0 - 0.5 ** (1.0 / poly_power))
            start_step = {
                "early": 0,
                "mid": round(total_steps * mid_fraction),
                "late": round(total_steps * 0.9),
            }[args.phase]
            bench.profile_steps(
                lambda step: train_step(ctx, batches[step % len(batches)], start_step + step),
                warmup=20,
                steps=5,
            )


if __name__ == "__main__":
    main()
