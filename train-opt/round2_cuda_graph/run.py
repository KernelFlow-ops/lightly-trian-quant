"""Round2 correctness, fixed-phase throughput, e2e, and profile entrypoint."""
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
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/A/inductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/A/triton")
os.environ.setdefault("MAX_JOBS", "6")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import empty_result, env_info, gpu_lock, profile_steps, time_e2e, time_steps, write_results  # noqa: E402
from common.check import step_equivalence  # noqa: E402
from common.config import B_AUTO  # noqa: E402
from common.harness import RunCfg, apply_patches, build, make_loader, preload_batches, set_phase, train_step  # noqa: E402
from round2_cuda_graph.impl.segmented import GraphState  # noqa: E402

ROUND = Path(__file__).resolve().parent
R1 = "round1_torch_cleanup"
R2 = "round2_cuda_graph"


def _cfg(args: argparse.Namespace, *, optimized: bool) -> RunCfg:
    patches = (R1, R2) if optimized else (R1,)
    kwargs = {R1: {"variant": args.r1_variant}}
    if optimized:
        kwargs[R2] = {"variant": args.variant, "max_joint_signatures": args.max_joint_signatures}
    return RunCfg(
        model=args.model, batch_size=args.batch_size or B_AUTO[args.model],
        precision=args.precision, phase=args.phase if args.phase != "all" else "mid",
        patches=patches, patch_kwargs=kwargs, seed=args.seed, num_workers=4,
    )


def _path(args: argparse.Namespace) -> Path:
    if args.out:
        path = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = ROUND / "profile" / f"{args.model}_{args.variant}_{args.action}_{args.phase}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    return path


def _capture_report(ctx) -> dict:
    state = ctx.meta.get("round2_graph_state")
    return state.report() if isinstance(state, GraphState) else {"error": "no round2 state"}


def _graph_worked(report: dict, phases: tuple[str, ...]) -> bool:
    if not report.get("plain_captured"):
        return False
    available = {tuple(item) for item in report.get("joint_signatures", [])}
    signatures = {
        "early": ("one", "one", "one"),
        "mid": ("zero", "middle", "one"),
        "late": ("zero", "zero", "zero"),
    }
    return all(signatures[phase] in available for phase in phases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(B_AUTO), default="vitl16")
    parser.add_argument("--variant", choices=("segmented", "compile_default", "compile_reduce_overhead", "compile_max_autotune_no_cudagraphs"), default="segmented")
    parser.add_argument("--r1-variant", default="nosync")
    parser.add_argument("--action", choices=("check", "bench", "e2e", "profile"), required=True)
    parser.add_argument("--phase", choices=("early", "mid", "late", "all"), default="mid")
    parser.add_argument("--max-joint-signatures", type=int, default=7)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.batch_size is None and B_AUTO[args.model] is None:
        raise RuntimeError("B_AUTO is not fixed")
    if args.action in {"profile", "check"} and args.phase == "all":
        parser.error("profile/check need one phase")
    output = _path(args)
    cfg = _cfg(args, optimized=True)
    result = empty_result(
        round_name=R2, variant=args.variant, model=cfg.model,
        batch=int(cfg.batch_size or B_AUTO[cfg.model]), precision=cfg.precision,
        patches=cfg.patches, desc=f"{args.variant} after {args.r1_variant}",
    )
    phases = ("early", "mid", "late") if args.phase == "all" else (args.phase,)
    with gpu_lock():
        if args.action == "check":
            reference = build(_cfg(args, optimized=False))
            apply_patches(reference)
            batch = preload_batches(reference, k=1)
            del reference
            gc.collect()
            torch.cuda.empty_cache()
            measurement = step_equivalence(_cfg(args, optimized=False), cfg, batch)
            report = measurement.get("capture_report") or {}
            result["accuracy"]["level_b"].update(measurement)
            if args.variant == "segmented" and not _graph_worked(report, phases):
                result["accuracy"]["level_b"]["pass"] = False
                result["status"] = "capture_unverified"
            elif args.variant != "segmented" and (report.get("compile_error") or report.get("compile_s") is None):
                result["accuracy"]["level_b"]["pass"] = False
                result["status"] = "compile_unverified"
        else:
            ctx = build(cfg)
            apply_patches(ctx)
            if args.action == "bench":
                batches = preload_batches(ctx, k=8)
                measurement = time_steps(ctx, batches, phases=phases)
                result["gpu_step"].update(measurement)
                result["breakdown_ms"].update(measurement.get("mid", {}).get("breakdown_ms", {}))
                result["peak_mem_gb"] = max(measurement[phase]["peak_allocated_gb"] for phase in phases)
            elif args.action == "e2e":
                measurement = time_e2e(ctx, make_loader(ctx, "train"), steps=100)
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
            report = _capture_report(ctx)
            if args.variant == "segmented" and not _graph_worked(report, phases):
                result["status"] = "capture_fallback"
            elif args.variant == "segmented":
                result["status"] = "captured_unverified"
            else:
                result["status"] = "compile_fallback" if report.get("compile_error") else "compiled_unverified" if report.get("compile_s") is not None else "compile_not_executed"
        result["env"] = env_info()
    result["capture_report"] = report
    record = {"command": [sys.executable, *sys.argv], "action": args.action, "measurement": measurement, "capture_report": report, "result": result}
    output.write_text(json.dumps(record, indent=2, default=str) + "\n")
    result["logs"].append(str(output))
    write_results(ROUND / "results.json", {cfg.model: {args.variant: result}})
    print(json.dumps({"status": result.get("status"), "B_pass": result["accuracy"]["level_b"]["pass"], "weighted_img_s": result["gpu_step"]["weighted_img_s"], "log": str(output)}, indent=2))


if __name__ == "__main__":
    main()
