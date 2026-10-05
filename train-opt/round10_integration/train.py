"""Train through Lightly's loop or the patchable, export-compatible fast loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/A/inductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/A/triton")
os.environ.setdefault("MAX_JOBS", "6")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import assert_gpu_exclusive, empty_result, env_info, gpu_lock, write_results  # noqa: E402
from common.check import read_lightly_metrics  # noqa: E402
from common.config import BACKBONE_WEIGHTS, B_AUTO, BEST_STACK, P0_STACK, DEFAULT_PATCH_KWARGS, DATASET_ROOT, MODEL_NAMES, RUNS_ROOT, TMP_ROOT  # noqa: E402
from common.data import _source_script, data_config_path  # noqa: E402
from common.harness import RunCfg, apply_patches, build  # noqa: E402
from round10_integration.impl.fast_loop import run_fast  # noqa: E402
from round10_integration.impl.full_graph import FullGraphRunner  # noqa: E402
from round10_integration.impl.lightly_patch import patch_lightly_loop  # noqa: E402

PATCH_ORDER = (
    "round1_torch_cleanup", "round3_loss_matcher", "round4_attention",
    "round5_block_fusion", "round6_head_fusion", "round7_optimizer",
    "round9_low_precision", "round8_data_pipeline", "round2_cuda_graph",
)


def _model_key(name: str) -> str:
    for key, value in MODEL_NAMES.items():
        if name in (key, value):
            return key
    raise ValueError(f"Unknown model {name}; expected {list(MODEL_NAMES)} or {list(MODEL_NAMES.values())}")


def _patches(args: argparse.Namespace) -> tuple[tuple[str, ...], dict]:
    explicit = tuple(value.strip() for value in args.patches.split(",") if value.strip())
    if explicit and not args.opt:
        raise ValueError("--patches requires --opt")
    preset = getattr(args, "preset", None)
    if explicit and preset:
        raise ValueError("Choose --preset or --patches")
    selected = (explicit or (P0_STACK if preset == "p0" else tuple(BEST_STACK))) if args.opt else ()
    if preset and not args.opt:
        raise ValueError("--preset requires --opt")
    if args.opt and not selected:
        raise ValueError("No BEST_STACK is published; pass an explicit --patches list")
    if len(selected) != len(set(selected)) or any(name not in PATCH_ORDER for name in selected):
        raise ValueError(f"Unknown or duplicate patch in {selected}")
    if selected != tuple(sorted(selected, key=PATCH_ORDER.index)):
        raise ValueError("Patches must follow SCHEDULE's standard combination order")
    kwargs = json.loads(args.patch_kwargs)
    if not isinstance(kwargs, dict) or any(not isinstance(value, dict) for value in kwargs.values()):
        raise ValueError("--patch-kwargs must be a JSON object keyed by patch name")
    if set(kwargs) - set(selected):
        raise ValueError(f"Patch kwargs have no matching selected patch: {sorted(set(kwargs) - set(selected))}")
    # The published stack has explicit, tested variants.  An explicitly
    # supplied object wins per patch, which keeps exploratory runs reproducible
    # without making ``--opt`` silently select an unconfigured implementation.
    defaults = {name: dict(DEFAULT_PATCH_KWARGS.get(name, {})) for name in selected}
    for name, values in kwargs.items():
        defaults[name].update(values)
    kwargs = defaults
    return selected, kwargs


def _args() -> argparse.Namespace:
    source = _source_script()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--model", default=source.MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=source.DEFAULT_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=source.DEFAULT_NUM_WORKERS)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--batch-candidates", default=",".join(str(value) for value in source.DEFAULT_BATCH_CANDIDATES))
    parser.add_argument("--precision", default=source.DEFAULT_PRECISION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone-freeze", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--data-config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-batch-probe", action="store_true")
    parser.add_argument("--loop", choices=("lightly", "fast"), default="fast")
    parser.add_argument("--opt", action="store_true")
    parser.add_argument("--preset", choices=("p0",), help="explicit P0 candidate; C/D validation is required")
    parser.add_argument("--patches", default="")
    parser.add_argument("--patch-kwargs", default="{}")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--full-graph", action="store_true", help="experimental full-step CUDA graph (requires r3+r7)")
    return parser.parse_args()


def main() -> None:
    args = _args()
    model = _model_key(args.model)
    source = _source_script()
    config_path = args.data_config or data_config_path()
    weights = args.backbone_weights or BACKBONE_WEIGHTS[model]
    if args.data_root != DATASET_ROOT and args.data_config is None:
        raise ValueError("A custom --data-root needs --data-config; source images and masks stay read-only")
    if not config_path.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Missing data config or backbone weights: {config_path}, {weights}")
    if args.epochs < 1 or args.num_workers < 0:
        raise ValueError("epochs must be positive and num-workers nonnegative")
    batch = B_AUTO[model] if args.batch_size == "auto" else int(args.batch_size)
    if batch is None or batch < 1:
        raise ValueError("batch size is unavailable")
    patches, patch_kwargs = _patches(args)
    if args.full_graph and args.loop != "fast":
        raise ValueError("--full-graph requires --loop fast")
    if args.loop == "lightly" and (set(patches) - {"round1_torch_cleanup"}):
        raise ValueError("Lightly runtime injection currently supports round1 only; choose --loop fast for this patch stack")
    if args.loop == "lightly" and patches and patch_kwargs.get("round1_torch_cleanup", {}).get("variant", "default") == "default":
        raise ValueError("--opt with the Lightly loop needs a non-default round1 variant")
    steps, steps_per_epoch = source.steps_for_epochs(3672, batch, args.epochs)
    data_config = json.loads(config_path.read_text())
    out = args.out or args.runs_root / f"trainopt_{model}_{args.loop}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not out.resolve().is_relative_to(TMP_ROOT.resolve()):
        raise ValueError(f"Training output must stay under {TMP_ROOT}: {out}")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite run: {out}")
    if shutil.disk_usage("/root/autodl-tmp").free < 15 * 1024**3:
        raise RuntimeError("At least 15 GB free space is required before a training run")
    with gpu_lock():
        assert_gpu_exclusive()
        if args.probe_only:
            probe_args = argparse.Namespace(**vars(args))
            probe_args.model = MODEL_NAMES[model]
            probe_args.backbone_weights = weights
            probe_args.data_config = config_path
            probe_args.out = out
            probe_args.batch_size = batch
            source.run_probe(probe_args)
            return
        out.mkdir(parents=True)
        if args.loop == "lightly":
            kwargs = source.common_train_kwargs(
                data_config=data_config, backbone_weights=weights, model=MODEL_NAMES[model],
                batch_size=batch, num_workers=args.num_workers, precision=args.precision,
                steps=steps, steps_per_epoch=steps_per_epoch, seed=args.seed,
                backbone_freeze=args.backbone_freeze,
            )
            kwargs["metric_args"]["classwise"] = True
            if args.no_save_checkpoints:
                kwargs["save_checkpoint_args"].update(save_last=False, save_best=False)
            r1_variant = patch_kwargs.get("round1_torch_cleanup", {}).get("variant", "default") if patches else "default"
            with patch_lightly_loop(r1_variant=r1_variant, precision=args.precision) if patches else nullcontext():
                source.run_lightly_train(out, kwargs)
            history = None
        else:
            cfg = RunCfg(
                model=model, batch_size=batch, precision=args.precision, phase="early",
                patches=patches, patch_kwargs=patch_kwargs, seed=args.seed,
                num_workers=args.num_workers, total_steps=steps,
                data_config=config_path, backbone_weights=weights,
                backbone_freeze=args.backbone_freeze,
            )
            ctx = build(cfg)
            apply_patches(ctx)
            if args.full_graph:
                ctx.meta["full_graph_runner"] = FullGraphRunner(ctx)
            history = run_fast(ctx, out, epochs=args.epochs, save_checkpoints=not args.no_save_checkpoints)
        if not args.no_save_checkpoints:
            source.copy_exported_pths(out)
        args.model = MODEL_NAMES[model]
        args.backbone_weights = weights
        source.write_run_config(
            out_dir=out, data_config=data_config, batch_size=batch,
            steps=steps, steps_per_epoch=steps_per_epoch, args=args,
        )
        summary = {
            "command": [sys.executable, *sys.argv], "model": model, "batch": batch,
            "epochs": args.epochs, "steps": steps, "seed": args.seed, "loop": args.loop,
            "patches": list(patches), "patch_kwargs": patch_kwargs,
            "full_graph": ctx.meta["full_graph_runner"].report.as_dict() if args.full_graph else None,
            "history_path": str(out / "metrics.jsonl") if history is not None else None,
            "exported_best": str(out / "exported_models" / "exported_best.pt") if not args.no_save_checkpoints else None,
            "env": env_info(),
        }
        (out / "train_opt_run.json").write_text(json.dumps(summary, indent=2) + "\n")
        final_metrics = None
        if history is not None and history:
            last = history[-1]
            final_metrics = {"miou": last["val_miou"], "head_iou": last["val_head_iou"]}
        elif args.loop == "lightly":
            try:
                final_metrics = read_lightly_metrics(out)
            except ValueError as exc:
                summary["metrics_error"] = str(exc)
                (out / "train_opt_run.json").write_text(json.dumps(summary, indent=2) + "\n")
        variant = "final" if args.opt else args.loop
        run_id = hashlib.sha1(str(out.resolve()).encode()).hexdigest()[:8]
        run_variant = f"{variant}_e{args.epochs}_s{args.seed}_{run_id}"
        update = empty_result(
            round_name="round10_integration", variant=run_variant, model=model,
            batch=batch, precision=args.precision, patches=patches,
            desc=f"{args.loop} loop; completed run at seed {args.seed}",
        )
        update["status"] = "run_complete_unverified"
        update["run"] = {"seed": args.seed, "out": str(out), "loop": args.loop}
        update["accuracy"]["level_d"].update({
            "budget_epochs": args.epochs,
            "val_miou": final_metrics.get("miou") if final_metrics else None,
            "val_head_iou": final_metrics.get("head_iou") if final_metrics else None,
        })
        update["env"] = summary["env"]
        update["logs"] = [str(out / "train_opt_run.json")]
        if history:
            update["epoch_s"] = history[-1]["epoch_s"]
        write_results(ROOT / "round10_integration" / "results.json", {model: {run_variant: update}})
    print(json.dumps({"out": str(out), "loop": args.loop, "patches": patches, "epochs": args.epochs, "batch": batch}, indent=2), flush=True)


if __name__ == "__main__":
    main()
