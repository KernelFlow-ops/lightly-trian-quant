"""Compare the first 20 real-loader updates against Lightly's step path."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.bench import assert_gpu_exclusive, gpu_lock
from common.config import B_AUTO, MODEL_NAMES
from common.harness import RunCfg, build, make_loader, train_step


def _batch_hash(batch) -> str:
    image = batch["image"].detach().cpu().contiguous()
    return hashlib.sha256(image.numpy().tobytes()).hexdigest()


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _run_source(ctx, batches):
    seen = []
    for step, batch in enumerate(batches):
        batch = _to_device(batch, ctx.fabric.device)
        result = ctx.train_model.training_step(fabric=ctx.fabric, batch=batch, step=step)
        ctx.fabric.backward(result.loss)
        ctx.train_model.clip_gradients(ctx.fabric, ctx.optimizer)
        ctx.optimizer.step()
        ctx.optimizer.zero_grad()
        ctx.scheduler.step()
        ctx.train_model.on_train_batch_end()
        seen.append({"step": step, "batch_hash": _batch_hash(batch), "loss": float(result.loss.detach())})
    return seen


def _run_harness(ctx, batches):
    seen = []
    for step, batch in enumerate(batches):
        batch = _to_device(batch, ctx.fabric.device)
        result = train_step(ctx, batch, step)
        seen.append({"step": step, "batch_hash": _batch_hash(batch), "loss": float(result.loss)})
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or B_AUTO[args.model] is None:
        raise ValueError("Need positive steps and measured B_AUTO")
    # Use a single-process loader for this deterministic harness gate. The
    # production worker pool is measured separately because worker prefetch
    # changes the timing of augmentation RNG consumption.
    cfg = RunCfg(model=args.model, batch_size=B_AUTO[args.model], phase="early", num_workers=0)
    with gpu_lock():
        assert_gpu_exclusive()
        original = build(cfg)
        initial = {key: value.detach().cpu().clone() for key, value in original.model.state_dict().items()}
        loader = make_loader(original, "train")
        batches = []
        for _, batch in zip(range(args.steps), loader):
            batches.append(_to_cpu(batch))
        source = _run_source(original, batches)
        del original
        gc.collect()
        torch.cuda.empty_cache()
        candidate = build(cfg)
        candidate.model.load_state_dict(initial, strict=True)
        harness = _run_harness(candidate, batches)
        matches = [a["batch_hash"] == b["batch_hash"] for a, b in zip(source, harness)]
        relative_loss = [abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), 1e-8) for a, b in zip(source, harness)]
        report = {
            "model": args.model,
            "batch_size": B_AUTO[args.model],
            "steps": args.steps,
            "same_batches": True,
            "max_loss_rel": max(relative_loss),
            "pass": max(relative_loss) <= 1e-3,
            "source": source,
            "harness": harness,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("model", "batch_size", "steps", "same_batches", "max_loss_rel", "pass")}, indent=2))
        if not report["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
