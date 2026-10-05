#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


DATA_ROOT = Path("/root/autodl-tmp/dinov3_head_2026_7_19_17h28")
BACKBONE_WEIGHTS = Path(
    "/root/autodl-tmp/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
RUNS_ROOT = Path("/root/autodl-tmp/runs")
MODEL_NAME = "dinov3/vitb16-eomt"
IMAGE_SIZE = (512, 512)
DEFAULT_EPOCHS = 20
DEFAULT_NUM_WORKERS = 25
DEFAULT_PRECISION = "bf16-mixed"
IGNORE_LABEL = 255
BACKGROUND_CLASS_ID = 0
DEFAULT_BATCH_CANDIDATES = (
    64,
    56,
    48,
    40,
    32,
    28,
    24,
    20,
    18,
    16,
    15,
    14,
    13,
    12,
    11,
    10,
    9,
    8,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--backbone-weights", type=Path, default=BACKBONE_WEIGHTS)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument(
        "--batch-candidates",
        default=",".join(str(x) for x in DEFAULT_BATCH_CANDIDATES),
    )
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone-freeze", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--data-config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-batch-probe", action="store_true")
    return parser.parse_args()


def load_coco(split_dir: Path) -> COCO:
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")
    return COCO(str(annotation_path))


def extract_foreground_classes(coco: COCO) -> tuple[dict[int, str], dict[int, int]]:
    categories = []
    for category in coco.loadCats(coco.getCatIds()):
        category_id = int(category["id"])
        if category_id == 0:
            continue
        categories.append((category_id, str(category["name"])))
    categories.sort()
    classes = {BACKGROUND_CLASS_ID: "background"}
    classes.update(
        {
            class_id: name
            for class_id, (_, name) in enumerate(categories, start=BACKGROUND_CLASS_ID + 1)
        }
    )
    category_id_to_class_id = {
        category_id: class_id
        for class_id, (category_id, _) in enumerate(
            categories, start=BACKGROUND_CLASS_ID + 1
        )
    }
    return classes, category_id_to_class_id


def write_semantic_masks(
    coco: COCO,
    split_dir: Path,
    mask_dir: Path,
    category_id_to_class_id: dict[int, int],
) -> int:
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_ids = coco.getImgIds()
    for image_info in coco.loadImgs(image_ids):
        height = int(image_info["height"])
        width = int(image_info["width"])
        semantic_mask = np.full((height, width), BACKGROUND_CLASS_ID, dtype=np.uint8)
        annotation_ids = coco.getAnnIds(imgIds=[image_info["id"]])
        for annotation in coco.loadAnns(annotation_ids):
            category_id = int(annotation["category_id"])
            class_id = category_id_to_class_id.get(category_id)
            if class_id is None:
                continue
            instance_mask = coco.annToMask(annotation).astype(bool)
            semantic_mask[instance_mask] = class_id

        image_path = split_dir / str(image_info["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed in COCO file is missing: {image_path}")
        mask_path = mask_dir / Path(image_info["file_name"]).with_suffix(".png").name
        Image.fromarray(semantic_mask).save(mask_path)
    return len(image_ids)


def prepare_data(data_root: Path) -> tuple[dict[str, Any], int, int]:
    train_coco = load_coco(data_root / "train")
    val_coco = load_coco(data_root / "valid")
    train_classes, train_category_id_to_class_id = extract_foreground_classes(
        train_coco
    )
    val_classes, val_category_id_to_class_id = extract_foreground_classes(val_coco)
    if train_classes != val_classes:
        raise ValueError(f"Train/val classes differ: {train_classes} != {val_classes}")
    if train_category_id_to_class_id != val_category_id_to_class_id:
        raise ValueError(
            "Train/val category mappings differ: "
            f"{train_category_id_to_class_id} != {val_category_id_to_class_id}"
        )

    train_masks = data_root / "train_masks"
    val_masks = data_root / "valid_masks"
    num_train = write_semantic_masks(
        train_coco,
        data_root / "train",
        train_masks,
        train_category_id_to_class_id,
    )
    num_val = write_semantic_masks(
        val_coco,
        data_root / "valid",
        val_masks,
        train_category_id_to_class_id,
    )

    classes_path = data_root / "semantic_classes.json"
    with classes_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in train_classes.items()}, f, indent=2)

    mapping_path = data_root / "semantic_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "background_class_id": BACKGROUND_CLASS_ID,
                "mask_ignore_label": None,
                "coco_category_id_to_class_id": {
                    str(k): v for k, v in train_category_id_to_class_id.items()
                },
            },
            f,
            indent=2,
        )

    data_config = {
        "train": {"images": str(data_root / "train"), "masks": str(train_masks)},
        "val": {"images": str(data_root / "valid"), "masks": str(val_masks)},
        "classes": train_classes,
    }
    config_path = data_root / "semantic_segmentation_data.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(data_config, f, indent=2)
    return data_config, num_train, num_val


def run_name(model: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_part = model.replace("/", "_").replace("-", "_")
    return f"{timestamp}_{model_part}"


def parse_batch_candidates(value: str) -> list[int]:
    candidates = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not candidates:
        raise ValueError("No batch candidates were provided.")
    return sorted(set(candidates), reverse=True)


def steps_for_epochs(num_train: int, batch_size: int, epochs: int) -> tuple[int, int]:
    steps_per_epoch = max(1, num_train // batch_size)
    return steps_per_epoch * epochs, steps_per_epoch


def common_train_kwargs(
    *,
    data_config: dict[str, Any],
    backbone_weights: Path,
    model: str,
    batch_size: int,
    num_workers: int,
    precision: str,
    steps: int,
    steps_per_epoch: int,
    seed: int,
    backbone_freeze: bool,
) -> dict[str, Any]:
    loader_args: dict[str, Any] = {"pin_memory": True}
    if num_workers > 0:
        loader_args["prefetch_factor"] = 2

    return {
        "model": model,
        "data": data_config,
        "steps": steps,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "devices": 1,
        "accelerator": "gpu",
        "precision": precision,
        "float32_matmul_precision": "high",
        "seed": seed,
        "model_args": {
            "backbone_weights": str(backbone_weights),
            "backbone_freeze": backbone_freeze,
            "num_joint_blocks": 3,
        },
        "transform_args": {"image_size": IMAGE_SIZE},
        "metric_args": {"watch_metric": "val_metric/miou"},
        "loader_args": loader_args,
        "logger_args": {
            "log_every_num_steps": max(1, min(100, steps_per_epoch // 5)),
            "val_every_num_steps": steps_per_epoch,
            "val_log_every_num_steps": 20,
        },
        "save_checkpoint_args": {
            "save_every_num_steps": steps_per_epoch,
            "save_last": True,
            "save_best": True,
        },
        "gradient_accumulation_steps": 1,
        "overwrite": False,
    }


def run_lightly_train(out_dir: Path, kwargs: dict[str, Any]) -> None:
    import lightly_train

    lightly_train.train_semantic_segmentation(out=out_dir, **kwargs)


def run_probe(args: argparse.Namespace) -> None:
    if args.data_config is None or args.out is None:
        raise ValueError("--probe-only requires --data-config and --out")
    data_config = json.loads(args.data_config.read_text(encoding="utf-8"))
    batch_size = int(args.batch_size)
    kwargs = common_train_kwargs(
        data_config=data_config,
        backbone_weights=args.backbone_weights,
        model=args.model,
        batch_size=batch_size,
        num_workers=min(args.num_workers, 4),
        precision=args.precision,
        steps=2,
        steps_per_epoch=2,
        seed=args.seed,
        backbone_freeze=args.backbone_freeze,
    )
    kwargs["logger_args"]["tensorboard"] = None
    kwargs["save_checkpoint_args"] = {
        "save_every_num_steps": 2,
        "save_last": False,
        "save_best": False,
    }
    run_lightly_train(args.out, kwargs)


def is_probable_oom(output: str) -> bool:
    lowered = output.lower()
    return (
        "out of memory" in lowered
        or "cuda error: out of memory" in lowered
        or "cublas_status_alloc_failed" in lowered
    )


def choose_batch_size(
    *,
    args: argparse.Namespace,
    data_config_path: Path,
    candidates: list[int],
) -> int:
    args.runs_root.mkdir(parents=True, exist_ok=True)
    for batch_size in candidates:
        probe_dir = args.runs_root / f"_batch_probe_{batch_size}"
        shutil.rmtree(probe_dir, ignore_errors=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--probe-only",
            "--data-config",
            str(data_config_path),
            "--out",
            str(probe_dir),
            "--data-root",
            str(args.data_root),
            "--backbone-weights",
            str(args.backbone_weights),
            "--runs-root",
            str(args.runs_root),
            "--model",
            args.model,
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(args.num_workers),
            "--precision",
            args.precision,
            "--seed",
            str(args.seed),
        ]
        if args.backbone_freeze:
            command.append("--backbone-freeze")
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        combined_output = f"{result.stdout}\n{result.stderr}"
        shutil.rmtree(probe_dir, ignore_errors=True)
        if result.returncode == 0:
            return batch_size
        if not is_probable_oom(combined_output):
            tail = combined_output[-4000:]
            raise RuntimeError(f"Batch-size probe failed for {batch_size}:\n{tail}")
    raise RuntimeError("All batch-size candidates failed.")


def copy_exported_pths(out_dir: Path) -> None:
    exported_dir = out_dir / "exported_models"
    last_src = exported_dir / "exported_last.pt"
    best_src = exported_dir / "exported_best.pt"
    if not last_src.is_file():
        raise FileNotFoundError(f"Missing exported last model: {last_src}")
    if not best_src.is_file():
        best_src = last_src
    shutil.copy2(last_src, out_dir / "last.pth")
    shutil.copy2(best_src, out_dir / "best.pth")


def write_run_config(
    *,
    out_dir: Path,
    data_config: dict[str, Any],
    batch_size: int,
    steps: int,
    steps_per_epoch: int,
    args: argparse.Namespace,
) -> None:
    config = {
        "data_root": str(args.data_root),
        "backbone_weights": str(args.backbone_weights),
        "model": args.model,
        "image_size": list(IMAGE_SIZE),
        "epochs": args.epochs,
        "steps": steps,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "precision": args.precision,
        "backbone_freeze": args.backbone_freeze,
        "background_class_id": BACKGROUND_CLASS_ID,
        "mask_ignore_label": None,
        "data": data_config,
    }
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.probe_only:
        run_probe(args)
        return

    if not args.backbone_weights.is_file():
        raise FileNotFoundError(f"Backbone weights not found: {args.backbone_weights}")

    data_config, num_train, num_val = prepare_data(args.data_root)
    data_config_path = args.data_root / "semantic_segmentation_data.json"
    args.runs_root.mkdir(parents=True, exist_ok=True)

    if args.batch_size == "auto":
        if args.skip_batch_probe:
            batch_size = 16
        else:
            candidates = [
                candidate
                for candidate in parse_batch_candidates(args.batch_candidates)
                if candidate <= num_train
            ]
            batch_size = choose_batch_size(
                args=args,
                data_config_path=data_config_path,
                candidates=candidates,
            )
    else:
        batch_size = int(args.batch_size)

    steps, steps_per_epoch = steps_for_epochs(num_train, batch_size, args.epochs)
    out_dir = args.out if args.out is not None else args.runs_root / run_name(args.model)
    print(
        json.dumps(
            {
                "out": str(out_dir),
                "model": args.model,
                "train_images": num_train,
                "val_images": num_val,
                "epochs": args.epochs,
                "batch_size": batch_size,
                "steps": steps,
                "num_workers": args.num_workers,
                "background_class_id": BACKGROUND_CLASS_ID,
                "mask_ignore_label": None,
            },
            indent=2,
        )
    )

    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = common_train_kwargs(
        data_config=data_config,
        backbone_weights=args.backbone_weights,
        model=args.model,
        batch_size=batch_size,
        num_workers=args.num_workers,
        precision=args.precision,
        steps=steps,
        steps_per_epoch=steps_per_epoch,
        seed=args.seed,
        backbone_freeze=args.backbone_freeze,
    )
    run_lightly_train(out_dir, train_kwargs)
    copy_exported_pths(out_dir)
    write_run_config(
        out_dir=out_dir,
        data_config=data_config,
        batch_size=batch_size,
        steps=steps,
        steps_per_epoch=steps_per_epoch,
        args=args,
    )
    print(f"Saved {out_dir / 'last.pth'}")
    print(f"Saved {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
