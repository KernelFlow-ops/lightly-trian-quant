"""Independent round3 microbenchmark and nsys capture entrypoint."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.bench import env_info, gpu_lock, profile_steps, time_steps, write_results
from common.config import B_AUTO, PHASE_WEIGHTS
from common.harness import (
    RunCfg,
    apply_patches,
    build,
    preload_batches,
    set_phase,
    train_step,
)

from lightly_train._task_models.dinov3_eomt_semantic_segmentation.mask_loss import (
    MaskClassificationLoss,
)
from round3_loss_matcher.impl.fast_criterion import FastMaskClassificationLoss


def make_inputs(batch: int, points: int, dtype: torch.dtype):
    torch.manual_seed(451)
    masks = [
        torch.randn(batch, 100, 128, 128, device="cuda", dtype=dtype).requires_grad_()
        for _ in range(4)
    ]
    classes = [
        torch.randn(batch, 100, 3, device="cuda", dtype=dtype).requires_grad_()
        for _ in range(4)
    ]
    targets = [
        {
            "masks": torch.randint(0, 2, (1 + i % 2, 512, 512), device="cuda").bool(),
            "labels": torch.arange(1 + i % 2, device="cuda"),
        }
        for i in range(batch)
    ]
    reference = MaskClassificationLoss(
        num_points=points, oversample_ratio=3.0,
        importance_sample_ratio=0.75, mask_coefficient=5.0,
        dice_coefficient=5.0, class_coefficient=2.0,
        num_labels=2, no_object_coefficient=0.1,
    ).cuda()
    return masks, classes, targets, reference


def loss_forward(variant, masks, classes, targets, reference, fast, num_blocks, dtype):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        if variant != "original":
            return fast.forward_all_layers(masks, classes, targets)[0]
        losses = {}
        for i, (mask, cls) in enumerate(zip(masks, classes)):
            layer = reference(mask, targets, cls)
            suffix = f"_block{num_blocks - 3 + i}" if i < 3 else ""
            losses.update({key + suffix: value for key, value in layer.items()})
        return reference.loss_total(losses)


def one_step(variant, masks, classes, targets, reference, fast, num_blocks, dtype):
    for value in masks + classes:
        value.grad = None
    loss = loss_forward(variant, masks, classes, targets, reference, fast, num_blocks, dtype)
    loss.backward()


def microbench(variant, masks, classes, targets, reference, fast, num_blocks, dtype, warmup, iters):
    for _ in range(warmup):
        one_step(variant, masks, classes, targets, reference, fast, num_blocks, dtype)
    torch.cuda.synchronize()
    forward_ms, backward_ms, wall_ms = [], [], []
    for _ in range(iters):
        for value in masks + classes:
            value.grad = None
        start, mid, end = (
            torch.cuda.Event(enable_timing=True) for _ in range(3)
        )
        wall_start = time.perf_counter()
        start.record()
        loss = loss_forward(variant, masks, classes, targets, reference, fast, num_blocks, dtype)
        mid.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000)
        forward_ms.append(start.elapsed_time(mid))
        backward_ms.append(mid.elapsed_time(end))
    return {
        "forward_gpu_ms": statistics.median(forward_ms),
        "backward_gpu_ms": statistics.median(backward_ms),
        "total_gpu_ms": statistics.median([a + b for a, b in zip(forward_ms, backward_ms)]),
        "wall_ms": statistics.median(wall_ms),
        "iters": iters,
        "warmup": warmup,
    }


def benchmark_steps(
    model: str,
    variant: str,
    batch: int,
    output: Path,
    *,
    warmup: int = 20,
    iters: int = 50,
    repeats: int = 3,
    preload_count: int = 8,
) -> None:
    cfg = RunCfg(
        model=model, batch_size=batch, num_workers=4,
        patches=() if variant == "original" else ("round3_loss_matcher",),
        patch_kwargs={} if variant == "original" else {
            "round3_loss_matcher": {"variant": variant}
        },
    )
    ctx = build(cfg)
    apply_patches(ctx)
    batches = preload_batches(ctx, k=preload_count)
    measured = time_steps(
        ctx, batches, warmup=warmup, iters=iters, repeats=repeats, segments=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(measured, indent=2) + "\n")
    print(json.dumps({
        "model": model, "variant": variant,
        "weighted_img_s": measured["weighted_img_s"],
        "step_ms": {phase: measured[phase]["step_ms"] for phase in PHASE_WEIGHTS},
        "loss_ms": {phase: measured[phase]["breakdown_ms"]["loss"] for phase in PHASE_WEIGHTS},
    }, indent=2))
    if variant == "original":
        return
    result_path = Path(__file__).resolve().parent / "results.json"
    existing = json.loads(result_path.read_text()) if result_path.exists() else {}
    row = existing.get(model, {}).get(variant)
    if row is None or row["accuracy"]["level_b"]["pass"] is not True:
        raise RuntimeError("A passing B-level result is required before publishing step timing")
    gpu_step = {
        phase: {
            "step_ms": measured[phase]["step_ms"],
            "img_s": measured[phase]["img_s"],
        }
        for phase in PHASE_WEIGHTS
    }
    gpu_step["weighted_img_s"] = measured["weighted_img_s"]
    breakdown = {
        segment: sum(
            PHASE_WEIGHTS[phase] * measured[phase]["breakdown_ms"][segment]
            for phase in PHASE_WEIGHTS
        )
        for segment in ("fwd", "loss", "bwd", "opt", "other")
    }
    write_results(result_path, {model: {variant: {
        "gpu_step": gpu_step,
        "breakdown_ms": breakdown,
        "peak_mem_gb": max(measured[phase]["peak_allocated_gb"] for phase in PHASE_WEIGHTS),
        "env": env_info(),
        "logs": list(dict.fromkeys(row["logs"] + [str(output.resolve())])),
    }}})


def profile_real_steps(model: str, variant: str, batch: int, phase: str) -> None:
    cfg = RunCfg(
        model=model, batch_size=batch, phase=phase, num_workers=4,
        patches=() if variant == "original" else ("round3_loss_matcher",),
        patch_kwargs={} if variant == "original" else {
            "round3_loss_matcher": {"variant": variant}
        },
    )
    ctx = build(cfg)
    apply_patches(ctx)
    batches = preload_batches(ctx, k=8)
    set_phase(ctx, phase)
    ctx.meta["phase_override"] = True
    ctx.nvtx = True
    profile_steps(
        lambda step: train_step(ctx, batches[step % len(batches)], step),
        warmup=20, steps=5,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("vitl16", "vitb16"), default="vitl16")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--variant", choices=("original", "batched_torch", "fused", "cuda"), default="fused")
    parser.add_argument("--mode", choices=("microbench", "profile", "step", "profile-step"), default="microbench")
    parser.add_argument("--phase", choices=("early", "mid", "late"), default="mid")
    parser.add_argument("--points", type=int, default=12544)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preload-batches", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batch = args.batch or B_AUTO[args.model]
    if batch is None:
        raise RuntimeError("B_AUTO is unset; provide --batch for independent development")
    num_blocks = 24 if args.model == "vitl16" else 12
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    with gpu_lock():
        compute_apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
            text=True,
        )
        if len(compute_apps.strip().splitlines()) > 1:
            raise RuntimeError(f"Other GPU process active; measurement canceled:\n{compute_apps}")
        if args.mode == "step":
            if args.dtype != "fp32":
                raise ValueError("step mode uses the original Fabric fp32 forward")
            output = args.output or Path(__file__).resolve().parent / "profile" / f"step_{args.model}_{args.variant}.json"
            benchmark_steps(
                args.model, args.variant, batch, output,
                warmup=args.warmup, iters=args.iters, repeats=args.repeats,
                preload_count=args.preload_batches,
            )
            return
        if args.mode == "profile-step":
            if args.dtype != "fp32":
                raise ValueError("profile-step uses the original Fabric fp32 forward")
            profile_real_steps(args.model, args.variant, batch, args.phase)
            return
        masks, classes, targets, reference = make_inputs(batch, args.points, dtype)
        fast = (
            FastMaskClassificationLoss(
                reference, num_blocks=num_blocks,
                variant=args.variant,
            ).cuda()
            if args.variant != "original" else None
        )
        if args.mode == "profile":
            profile_steps(
                lambda _step: one_step(
                    args.variant, masks, classes, targets, reference, fast, num_blocks, dtype
                ),
                warmup=20, steps=5,
            )
            return
        result = microbench(
            args.variant, masks, classes, targets, reference, fast, num_blocks, dtype,
            args.warmup, args.iters,
        )
        result.update(model=args.model, batch=batch, variant=args.variant, dtype=args.dtype,
                      points=args.points, compute_apps=compute_apps.strip())
        text = json.dumps(result, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
