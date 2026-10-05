"""Extract local backbone checkpoints and construct the original training model."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from lightly_train._commands import train_task_helpers as helpers
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.train_model import (
    DINOv3EoMTSemanticSegmentationTrain,
)

from .config import BACKBONE_WEIGHTS, EXPORT_WEIGHTS, MODEL_NAMES


def _export(model: str) -> dict[str, Any]:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")
    return torch.load(EXPORT_WEIGHTS[model], map_location="cpu", weights_only=False, mmap=True)


def extract_backbone(model: str) -> Path:
    exported = _export(model)
    prefix = "model.backbone."
    original = {
        key[len(prefix):]: value
        for key, value in exported["train_model"].items()
        if key.startswith(prefix)
    }
    if not original:
        raise RuntimeError(f"No backbone tensors in {EXPORT_WEIGHTS[model]}")
    path = BACKBONE_WEIGHTS[model]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(original, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if original.keys() != reloaded.keys():
        raise AssertionError("Extracted backbone keys changed")
    for key in original:
        if not torch.equal(original[key], reloaded[key]):
            raise AssertionError(f"Extracted backbone tensor changed: {key}")
    return path


def build_train_model(
    model: str,
    data_args: Any,
    batch_size: int,
    total_steps: int,
    backbone_weights: Path | None = None,
    backbone_freeze: bool = False,
) -> tuple[DINOv3EoMTSemanticSegmentationTrain, Any, Any]:
    if batch_size < 1 or total_steps < 1:
        raise ValueError("batch_size and total_steps must be positive")
    weights = backbone_weights or BACKBONE_WEIGHTS[model]
    if not weights.is_file():
        raise FileNotFoundError(f"Run extract_backbone({model!r}) first: {weights}")
    model_name = MODEL_NAMES[model]
    model_init_args = _export(model)["model_init_args"]
    cls = DINOv3EoMTSemanticSegmentationTrain
    train_batches = math.floor(3672 / batch_size)
    model_args = helpers.get_train_model_args(
        model_args={"backbone_weights": str(weights), "backbone_freeze": backbone_freeze, "num_joint_blocks": 3},
        model_args_cls=cls.get_train_model_args_cls(model_name),
        total_steps=total_steps,
        gradient_accumulation_steps=1,
        train_num_batches=train_batches,
        model_name=model_name,
        model_init_args=model_init_args,
        data_args=data_args,
    )
    train_transform_args, val_transform_args = helpers.get_transform_args(
        train_model_cls=cls,
        model_name=model_name,
        transform_args={"image_size": (512, 512)},
        ignore_index=getattr(data_args, "ignore_index", None),
        model_init_args=model_init_args,
        total_steps=total_steps,
        train_num_batches=train_batches,
        gradient_accumulation_steps=1,
    )
    metric_args = helpers.get_metric_args(
        train_model_cls=cls,
        metric_args={"watch_metric": "val_metric/miou", "classwise": True},
        data_args=data_args,
    )
    train_model = cls(
        model_name=model_name,
        model_args=model_args,
        data_args=data_args,
        train_transform_args=train_transform_args,
        val_transform_args=val_transform_args,
        load_weights=True,
        metric_args=metric_args,
        gradient_accumulation_steps=1,
    )
    train_model.set_train_mode()
    return train_model, train_transform_args, val_transform_args
