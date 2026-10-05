"""One locked data-pipeline sweep point on the fixed training batch."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from common.bench import assert_gpu_exclusive, empty_result, env_info, gpu_lock, time_e2e, write_results  # noqa: E402
from common.config import B_AUTO  # noqa: E402
from common.harness import RunCfg, apply_patches, build  # noqa: E402
from round8_data_pipeline.impl.loader import LoaderOptions, make_loader  # noqa: E402

ROUND = Path(__file__).resolve().parent


def _loader_only(loader, steps: int) -> dict:
    iterator = iter(loader)
    waits = []
    images = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        waits.append((time.perf_counter() - t0) * 1000)
        images += len(batch["image"])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"img_s": images / elapsed, "dataload_wait_ms": statistics.fmean(waits), "wall_s": elapsed, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(B_AUTO), required=True)
    parser.add_argument("--action", choices=("e2e", "loader_only"), default="e2e")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or B_AUTO[args.model] is None:
        raise ValueError("steps must be positive and B_AUTO fixed")
    options = LoaderOptions(args.workers, args.prefetch_factor, args.pin_memory, args.persistent_workers)
    options.args()
    variant = f"loader_w{args.workers}_pf{args.prefetch_factor}_pin{int(args.pin_memory)}_persist{int(args.persistent_workers)}_{args.action}"
    output = args.out or ROUND / "profile" / f"{args.model}_{variant}_{args.action}_s{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    cfg = RunCfg(
        model=args.model, batch_size=B_AUTO[args.model], precision="bf16-mixed",
        phase="early", seed=args.seed, num_workers=args.workers,
        patches=("round8_data_pipeline",),
        patch_kwargs={"round8_data_pipeline": {"variant": "loader", "workers": args.workers, "prefetch_factor": args.prefetch_factor, "pin_memory": args.pin_memory, "persistent_workers": args.persistent_workers}},
    )
    with gpu_lock():
        before = assert_gpu_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        loader, effective = make_loader(ctx, "train", options)
        if args.action == "e2e":
            measurement = time_e2e(ctx, loader, steps=args.steps)
        else:
            measurement = _loader_only(loader, args.steps)
        environment = env_info()
        after = assert_gpu_exclusive()
    result = empty_result(
        round_name="round8_data_pipeline", variant=variant, model=args.model,
        batch=B_AUTO[args.model], precision="bf16-mixed", patches=cfg.patches,
        desc=f"{args.action} loader sweep: {effective}",
    )
    result["status"] = "measured_accuracy_unverified"
    result["loader_options"] = effective
    if args.action == "e2e":
        result["e2e"].update({"img_s": measurement["img_s"], "dataload_wait_ms": measurement["dataload_wait_ms"]})
    else:
        result["loader_only"] = measurement
    result["env"] = environment
    result["logs"] = [str(output)]
    output.write_text(json.dumps({"command": [sys.executable, *sys.argv], "model": args.model, "batch": B_AUTO[args.model], "variant": variant, "action": args.action, "requested": options.__dict__, "effective": effective, "measurement": measurement, "gpu_snapshot_before": before, "gpu_snapshot_after": after}, indent=2) + "\n")
    write_results(ROUND / "results.json", {args.model: {variant: result}})
    print(json.dumps({"variant": variant, "measurement": measurement, "log": str(output)}, indent=2))


if __name__ == "__main__":
    main()
