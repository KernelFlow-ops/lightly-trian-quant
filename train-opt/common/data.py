"""Prepare semantic masks outside the source dataset and build Lightly loaders."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from lightly_train._commands import train_task_helpers as helpers
from lightly_train._data.mask_semantic_segmentation_dataset import MaskSemanticSegmentationDataArgs

from .config import DATA_ROOT, DATASET_ROOT, PROJECT_ROOT


def _source_script() -> Any:
    path = PROJECT_ROOT / "scripts" / "train_dinov3_eomt_head.py"
    spec = importlib.util.spec_from_file_location("train_opt_source_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_config_path() -> Path:
    return DATA_ROOT / "semantic_segmentation_data.json"


def load_data_config(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or data_config_path()).read_text(encoding="utf-8"))


def prepare_semantic_data() -> tuple[dict[str, Any], int, int]:
    """Use the source script's COCO conversion, changing only output paths."""
    source = _source_script()
    train_coco = source.load_coco(DATASET_ROOT / "train")
    val_coco = source.load_coco(DATASET_ROOT / "valid")
    train_classes, train_mapping = source.extract_foreground_classes(train_coco)
    val_classes, val_mapping = source.extract_foreground_classes(val_coco)
    if train_classes != {0: "background", 1: "head"}:
        raise ValueError(f"Unexpected class mapping: {train_classes}")
    if train_classes != val_classes or train_mapping != val_mapping:
        raise ValueError("Train and validation category mappings differ")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    num_train = source.write_semantic_masks(
        train_coco, DATASET_ROOT / "train", DATA_ROOT / "train_masks", train_mapping
    )
    num_val = source.write_semantic_masks(
        val_coco, DATASET_ROOT / "valid", DATA_ROOT / "valid_masks", train_mapping
    )
    (DATA_ROOT / "semantic_classes.json").write_text(
        json.dumps({str(k): v for k, v in train_classes.items()}, indent=2), encoding="utf-8"
    )
    (DATA_ROOT / "semantic_mapping.json").write_text(
        json.dumps(
            {
                "background_class_id": source.BACKGROUND_CLASS_ID,
                "mask_ignore_label": None,
                "coco_category_id_to_class_id": {str(k): v for k, v in train_mapping.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    data_config = {
        "train": {"images": str(DATASET_ROOT / "train"), "masks": str(DATA_ROOT / "train_masks")},
        "val": {"images": str(DATASET_ROOT / "valid"), "masks": str(DATA_ROOT / "valid_masks")},
        "classes": train_classes,
    }
    data_config_path().write_text(json.dumps(data_config, indent=2), encoding="utf-8")
    if (num_train, num_val) != (3672, 477):
        raise AssertionError(f"Unexpected image counts: {(num_train, num_val)}")
    return data_config, num_train, num_val


def make_data_args(path: Path | None = None) -> MaskSemanticSegmentationDataArgs:
    path = path or data_config_path()
    args = MaskSemanticSegmentationDataArgs.model_validate(load_data_config(path))
    args.resolve_data_paths(base_dir=path.parent)
    return args


def make_loader(
    fabric: Any,
    data_args: MaskSemanticSegmentationDataArgs,
    transform_args: Any,
    train_model_cls: type,
    model_name: str,
    split: str,
    batch: int,
    workers: int,
    seed: int = 0,
) -> DataLoader:
    """Build the same dataset, transform, collate and loader as Lightly."""
    if split not in {"train", "val"}:
        raise ValueError(f"Unknown split: {split}")
    dataset_args = data_args.get_train_args() if split == "train" else data_args.get_val_args()
    image_info = list(dataset_args.list_image_info())
    dataset = dataset_args.get_dataset_cls()(dataset_args=dataset_args, image_info=image_info)
    transform = (
        helpers.get_train_transform(train_model_cls, model_name, transform_args)
        if split == "train"
        else helpers.get_val_transform(train_model_cls, model_name, transform_args)
    )
    dataset.set_transform(transform)
    loader_args = {"pin_memory": True}
    if workers > 0:
        loader_args["prefetch_factor"] = 2
    build = helpers.get_train_dataloader if split == "train" else helpers.get_val_dataloader
    loader = build(fabric, dataset, batch_size=batch, num_workers=workers, loader_args=loader_args)
    collate = dataset.get_batch_collate_fn_cls()(split=split, transform_args=transform_args)
    loader.collate_fn = collate
    helpers.disable_persistent_workers_for_step_aware_transforms(loader, transform, collate)
    return fabric.setup_dataloaders(loader)
