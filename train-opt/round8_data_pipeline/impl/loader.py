"""Lightly-equivalent data loader with controlled scheduling parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lightly_train._commands import train_task_helpers as helpers
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.train_model import DINOv3EoMTSemanticSegmentationTrain

from common.config import MODEL_NAMES


def loader_args(options: Any, transform: Any, collate: Any) -> dict[str, Any]:
    """Resolve worker persistence before constructing PyTorch's DataLoader."""
    args = options.args()
    aware = (
        transform.uses_step_dependent_worker_state()
        or collate.uses_step_dependent_worker_state()
    )
    if aware and options.workers > 0:
        args["persistent_workers"] = False
    return args


@dataclass(frozen=True)
class LoaderOptions:
    workers: int
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False

    def args(self) -> dict[str, Any]:
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")
        result = {"pin_memory": self.pin_memory}
        if self.workers > 0:
            result.update(prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)
        elif self.persistent_workers:
            raise ValueError("persistent_workers requires workers > 0")
        return result


def make_loader(ctx: Any, split: str, options: LoaderOptions):
    if split not in {"train", "val"}:
        raise ValueError(f"Unknown split: {split}")
    data_args = ctx.meta["data_args"]
    transform_args = ctx.meta["train_transform_args"] if split == "train" else ctx.meta["val_transform_args"]
    dataset_args = data_args.get_train_args() if split == "train" else data_args.get_val_args()
    image_info = list(dataset_args.list_image_info())
    dataset = dataset_args.get_dataset_cls()(dataset_args=dataset_args, image_info=image_info)
    model_name = MODEL_NAMES[ctx.cfg.model]
    if split == "train":
        transform = helpers.get_train_transform(DINOv3EoMTSemanticSegmentationTrain, model_name, transform_args)
        factory = helpers.get_train_dataloader
    else:
        transform = helpers.get_val_transform(DINOv3EoMTSemanticSegmentationTrain, model_name, transform_args)
        factory = helpers.get_val_dataloader
    dataset.set_transform(transform)
    collate = dataset.get_batch_collate_fn_cls()(split=split, transform_args=transform_args)
    loader = factory(ctx.fabric, dataset, batch_size=ctx.meta["batch_size"], num_workers=options.workers, loader_args=loader_args(options, transform, collate))
    loader.collate_fn = collate
    effective = {"workers": loader.num_workers, "prefetch_factor": getattr(loader, "prefetch_factor", None), "pin_memory": loader.pin_memory, "persistent_workers": loader.persistent_workers, "non_blocking_cuda_transfer": ctx.fabric.device.type == "cuda"}
    return ctx.fabric.setup_dataloaders(loader), effective
