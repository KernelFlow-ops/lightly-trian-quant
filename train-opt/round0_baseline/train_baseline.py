"""Run the original Lightly loop with isolated data and backbone weights."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import BACKBONE_WEIGHTS, DATA_ROOT, MODEL_NAMES, RUNS_ROOT
from common.data import _source_script, load_data_config


def train(model: str, batch_size: int, epochs: int, seed: int, out: Path | None = None, num_workers: int = 25, save_checkpoints: bool = True) -> Path:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")
    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    source = _source_script()
    steps, steps_per_epoch = source.steps_for_epochs(3672, batch_size, epochs)
    kwargs = source.common_train_kwargs(
        data_config=load_data_config(),
        backbone_weights=BACKBONE_WEIGHTS[model],
        model=MODEL_NAMES[model],
        batch_size=batch_size,
        num_workers=num_workers,
        precision="bf16-mixed",
        steps=steps,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        backbone_freeze=False,
    )
    # Classwise logging supplies the head IoU required by the accuracy gate.
    kwargs["metric_args"]["classwise"] = True
    if not save_checkpoints:
        kwargs["save_checkpoint_args"].update(save_last=False, save_best=False)
    if out is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RUNS_ROOT / f"baseline_{model}_e{epochs}_s{seed}_{run_id}"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite run: {out}")
    source.run_lightly_train(out, kwargs)
    (out / "train_opt_config.json").write_text(
        json.dumps({"model": model, "batch_size": batch_size, "epochs": epochs, "seed": seed, "kwargs": kwargs}, indent=2, default=str),
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--num-workers", type=int, default=25)
    parser.add_argument("--no-save-checkpoints", action="store_true")
    args = parser.parse_args()
    print(train(args.model, args.batch_size, args.epochs, args.seed, args.out, args.num_workers, not args.no_save_checkpoints), flush=True)
