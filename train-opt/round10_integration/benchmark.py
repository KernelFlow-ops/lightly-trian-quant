"""Locked M0 benchmark for the published train-opt compositions.

Each invocation measures one model/composition so the previous model and its
optimizer state can be released before the next run.  Results are JSON files
that can be compared directly with ``round0_baseline/results.json``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import assert_gpu_exclusive, env_info, gpu_lock, profile_steps, time_e2e, time_steps
from common.config import BACKBONE_WEIGHTS, B_AUTO, P0_STACK, DEFAULT_PATCH_KWARGS
from common.harness import RunCfg, apply_patches, build, make_loader, preload_batches, set_phase, train_step
from round10_integration.impl.full_graph import FullGraphRunner


def _composition(name: str) -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
    if name == "baseline":
        return (), {}
    selections = {
        "r1": ("round1_torch_cleanup",),
        "r3": ("round1_torch_cleanup", "round3_loss_matcher"),
        "r4": ("round1_torch_cleanup", "round4_attention"),
        "r5": ("round1_torch_cleanup", "round5_block_fusion"),
        "r7": ("round1_torch_cleanup", "round7_optimizer"),
        "p0": P0_STACK, "opt_eager": P0_STACK, "stack": P0_STACK,
    }
    if name in selections:
        selected = tuple(selections[name])
        kwargs = {key: dict(DEFAULT_PATCH_KWARGS.get(key, {})) for key in selected}
        return selected, kwargs
    raise ValueError(f"unknown composition {name!r}")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(B_AUTO), required=True)
    parser.add_argument("--composition", choices=("baseline", "r1", "r3", "r4", "r5", "r7", "p0", "opt_eager", "stack"), default="baseline")
    parser.add_argument("--action", choices=("bench", "e2e", "profile", "graph", "check", "curve"), default="bench")
    parser.add_argument("--patch-kwargs", default="{}")
    parser.add_argument("--phase", choices=("early", "mid", "late", "all"), default="all")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preload-count", type=int, default=1,
                        help="number of GPU-resident batches; one avoids starving ViT-B backward on 32 GiB")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _args()
    batch = B_AUTO[args.model]
    assert batch is not None
    patches, patch_kwargs = _composition(args.composition)
    for name, values in json.loads(args.patch_kwargs).items():
        if name not in patches or not isinstance(values, dict):
            raise ValueError("patch kwargs must refer to selected patches")
        patch_kwargs[name].update(values)
    cfg = RunCfg(
        model=args.model,
        batch_size=batch,
        phase="mid",
        patches=patches,
        patch_kwargs=patch_kwargs,
        num_workers=args.workers,
        backbone_weights=BACKBONE_WEIGHTS[args.model],
    )
    result: dict = {
        "model": args.model,
        "batch": batch,
        "composition": args.composition,
        "patches": list(patches),
        "patch_kwargs": patch_kwargs,
        "action": args.action,
        "command": [sys.executable, *sys.argv],
    }
    with gpu_lock():
        assert_gpu_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        if args.action in {"bench", "profile", "graph", "check"}:
            batches = preload_batches(ctx, k=max(1, args.preload_count))
        if args.action == "bench":
            phases = ("early", "mid", "late") if args.phase == "all" else (args.phase,)
            measurement = time_steps(
                ctx, batches, phases=phases, warmup=args.warmup,
                iters=args.iters, repeats=args.repeats,
            )
            result["gpu_step"] = measurement
        elif args.action == "e2e":
            result["e2e"] = time_e2e(ctx, make_loader(ctx, "train"), steps=args.steps)
        elif args.action == "profile":
            if args.phase == "all":
                raise ValueError("profile requires one phase")
            set_phase(ctx, args.phase)
            ctx.meta["phase_override"] = True
            ctx.nvtx = True
            profile_steps(lambda i: train_step(ctx, batches[i % len(batches)], i), warmup=args.warmup, steps=5)
            result["profile"] = {"phase": args.phase, "warmup": args.warmup, "steps": 5}
        elif args.action == "check":
            from common.check import step_equivalence
            from common.rng import RngTape
            del ctx
            gc.collect()
            torch.cuda.empty_cache()
            reference = RunCfg(model=args.model, batch_size=batch, num_workers=args.workers)
            result["checks"] = {}
            for phase in (("early", "mid", "late") if args.phase == "all" else (args.phase,)):
                cfg.phase = reference.phase = phase
                result["checks"][phase] = step_equivalence(reference, cfg, batches, tape=RngTape(mode="record"))
        elif args.action == "curve":
            from common.check import curve
            del ctx
            gc.collect()
            torch.cuda.empty_cache()
            result["curve"] = curve(cfg, steps=args.steps)
        else:
            set_phase(ctx, "mid" if args.phase == "all" else args.phase)
            runner = FullGraphRunner(ctx, warmup=min(3, args.warmup))
            report = runner.capture(batches[0], step=0)
            result["graph"] = report.as_dict()
            if report.captured:
                torch.cuda.synchronize()
                start = time.perf_counter()
                for i in range(args.iters):
                    runner.replay(batches[i % len(batches)], step=i)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                result["graph"].update({
                    "replay_step_ms": elapsed * 1000.0 / args.iters,
                    "replay_img_s": batch * args.iters / elapsed,
                })
        result["env"] = env_info()
    result["output"] = str(args.output) if args.output else None
    output = args.output or Path(__file__).resolve().parent / "profile" / (
        f"m0_{args.model}_{args.composition}_{args.action}_{args.phase}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.action not in {"check", "curve"}:
        del ctx
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"output": str(output), "model": args.model, "composition": args.composition}, indent=2))


if __name__ == "__main__":
    main()
