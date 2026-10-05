"""Round1 check, benchmark, and profiler entrypoint."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import empty_result, env_info, gpu_lock, profile_steps, time_e2e, time_steps, write_results  # noqa: E402
from common.check import step_equivalence  # noqa: E402
from common.config import B_AUTO  # noqa: E402
from common.harness import RunCfg, apply_patches, build, make_loader, preload_batches, set_phase, train_step  # noqa: E402

ROUND = Path(__file__).resolve().parent
PATCH = "round1_torch_cleanup"


def _cfg(args: argparse.Namespace, *, optimized: bool) -> RunCfg:
    patch_kwargs = {PATCH: {"variant": args.variant}}
    if args.workers is not None:
        patch_kwargs[PATCH]["workers"] = args.workers
    return RunCfg(
        model=args.model,
        batch_size=args.batch_size or B_AUTO[args.model],
        precision=args.precision,
        phase=args.phase if args.phase != "all" else "mid",
        patches=(PATCH,) if optimized else (),
        patch_kwargs=patch_kwargs if optimized else {},
        seed=args.seed,
        num_workers=args.workers if args.workers is not None else 4,
    )


def _result(cfg: RunCfg, variant: str) -> dict:
    return empty_result(
        round_name=PATCH, variant=variant, model=cfg.model,
        batch=int(cfg.batch_size or B_AUTO[cfg.model]), precision=cfg.precision,
        patches=cfg.patches, desc=f"PyTorch cleanup: {variant}",
    )


def _log_path(args: argparse.Namespace) -> Path:
    if args.out:
        path = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_variant = args.variant.replace("+", "_").replace(",", "_")
        path = ROUND / "profile" / f"{args.model}_{safe_variant}_{args.action}_{args.phase}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(B_AUTO), default="vitl16")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--action", choices=("check", "bench", "e2e", "profile"), required=True)
    parser.add_argument("--phase", choices=("early", "mid", "late", "all"), default="mid")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.batch_size is None and B_AUTO[args.model] is None:
        raise RuntimeError("B_AUTO is not fixed")
    if args.action == "profile" and args.phase == "all":
        parser.error("profile needs one phase")
    output = _log_path(args)
    cfg = _cfg(args, optimized=True)
    result = _result(cfg, args.variant)
    command = [sys.executable, *sys.argv]

    with gpu_lock():
        if args.action == "check":
            reference = build(_cfg(args, optimized=False))
            batch = preload_batches(reference, k=1)
            del reference
            gc.collect()
            torch.cuda.empty_cache()
            measurement = step_equivalence(_cfg(args, optimized=False), cfg, batch)
            result["accuracy"]["level_b"].update(measurement)
        else:
            ctx = build(cfg)
            apply_patches(ctx)
            if args.action == "bench":
                batches = preload_batches(ctx, k=8)
                phases = ("early", "mid", "late") if args.phase == "all" else (args.phase,)
                measurement = time_steps(ctx, batches, phases=phases)
                result["gpu_step"].update(measurement)
                result["breakdown_ms"].update(measurement.get("mid", {}).get("breakdown_ms", {}))
                result["peak_mem_gb"] = max(measurement[phase]["peak_allocated_gb"] for phase in phases)
            elif args.action == "e2e":
                loader = make_loader(ctx, "train")
                measurement = time_e2e(ctx, loader, steps=100)
                result["e2e"].update(measurement)
            else:
                batches = preload_batches(ctx, k=8)
                ctx.nvtx = True
                ctx.meta["phase_override"] = True
                set_phase(ctx, args.phase)

                def step(index: int) -> None:
                    train_step(ctx, batches[index % len(batches)], index)

                profile_steps(step, warmup=20, steps=5)
                measurement = {"profile_steps": 5, "warmup_steps": 20, "phase": args.phase}
        result["env"] = env_info()
    record = {"command": command, "action": args.action, "measurement": measurement, "result": result}
    output.write_text(json.dumps(record, indent=2, default=str) + "\n")
    result["logs"].append(str(output))
    write_results(ROUND / "results.json", {cfg.model: {args.variant: result}})
    print(json.dumps({"pass": result["accuracy"]["level_b"]["pass"], "weighted_img_s": result["gpu_step"]["weighted_img_s"], "log": str(output)}, indent=2))


if __name__ == "__main__":
    main()
