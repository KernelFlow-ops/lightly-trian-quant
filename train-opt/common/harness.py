"""A small, patchable equivalent of Lightly's semantic segmentation training step."""
from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from lightning_fabric import Fabric

from .config import MODEL_NAMES, PHASES, B_AUTO, TRAIN_OPT_ROOT
from .data import make_data_args, make_loader as build_loader
from .fast_forward import forward_train_v2
from .model_factory import build_train_model
from .ops import Ops, default_ops
from .rng import RngTape


@dataclass
class RunCfg:
    model: str = "vitl16"
    batch_size: int | None = None
    precision: str = "bf16-mixed"
    phase: str = "mid"
    patches: tuple[str, ...] = ()
    patch_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    seed: int = 0
    num_workers: int = 25
    total_steps: int | None = None
    data_config: Path | None = None
    backbone_weights: Path | None = None
    backbone_freeze: bool = False


@dataclass
class Ctx:
    cfg: RunCfg
    fabric: Fabric
    train_model: Any
    model: Any
    optimizer: Any
    scheduler: Any
    ops: Ops
    rng: RngTape
    host_probs: list[float]
    shadow: dict[str, torch.Tensor] = field(default_factory=dict)
    post_opt_hooks: list[Callable[["Ctx"], None]] = field(default_factory=list)
    nvtx: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepOut:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    grad_norm: torch.Tensor | None


def build(cfg: RunCfg) -> Ctx:
    if cfg.model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {cfg.model!r}")
    data_args = make_data_args(cfg.data_config)
    batch = cfg.batch_size or B_AUTO.get(cfg.model)
    if batch is None:
        raise RuntimeError(f"B_AUTO[{cfg.model!r}] is unset; run the batch probe or set batch_size explicitly")
    total_steps = cfg.total_steps or max(1, (3672 // batch) * 20)
    fabric = Fabric(accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1, precision=cfg.precision)
    fabric.launch()
    fabric.seed_everything(seed=cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    train_model, train_transform_args, val_transform_args = build_train_model(
        cfg.model, data_args, batch, total_steps,
        backbone_weights=cfg.backbone_weights,
        backbone_freeze=cfg.backbone_freeze,
    )
    optimizer, scheduler = train_model.get_optimizer(total_steps=total_steps, global_batch_size=batch)
    train_model, optimizer = fabric.setup(train_model, optimizer)
    ops = default_ops()
    ctx = Ctx(
        cfg=cfg, fabric=fabric, train_model=train_model, model=train_model.model,
        optimizer=optimizer, scheduler=scheduler, ops=ops, rng=RngTape(),
        host_probs=list(PHASES.get(cfg.phase, PHASES["mid"])),
    )
    ctx.meta.update(data_args=data_args, train_transform_args=train_transform_args, val_transform_args=val_transform_args, batch_size=batch, total_steps=total_steps)
    set_phase(ctx, cfg.phase)
    return ctx


def _state_shapes(module: Any) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def apply_patches(ctx: Ctx) -> None:
    before = _state_shapes(ctx.train_model)
    loaded: list[str] = []
    for patch_name in ctx.cfg.patches:
        patch_path = Path(__file__).resolve().parents[1] / patch_name / "impl" / "patch.py"
        if not patch_path.is_file():
            raise FileNotFoundError(f"Patch not found: {patch_path}")
        if str(TRAIN_OPT_ROOT) not in sys.path:
            sys.path.insert(0, str(TRAIN_OPT_ROOT))
        module = importlib.import_module(f"{patch_name}.impl.patch")
        requires = tuple(getattr(module, "REQUIRES", ()))
        conflicts = tuple(getattr(module, "CONFLICTS", ()))
        if any(req not in loaded for req in requires):
            raise RuntimeError(f"Patch {patch_name} requires {requires}; loaded {loaded}")
        if any(conflict in loaded for conflict in conflicts):
            raise RuntimeError(f"Patch {patch_name} conflicts with {conflicts}")
        result = module.apply(ctx, **ctx.cfg.patch_kwargs.get(patch_name, {}))
        if result:
            ctx.meta.setdefault("patches", []).append(result)
        loaded.append(patch_name)
    after = _state_shapes(ctx.train_model)
    if before != after:
        raise AssertionError("Patch changed state_dict keys or tensor shapes")


def set_phase(ctx: Ctx, phase: str) -> None:
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    ctx.host_probs[:] = list(PHASES[phase])
    # Keep the model's device values synchronized without a host-side branch in the step.
    for i, value in enumerate(ctx.host_probs):
        ctx.model.attn_mask_probs[i].fill_(value)


def update_mask_schedule(ctx: Ctx, step: int) -> tuple[float, ...]:
    """Update the annealed mask probabilities before a graph/eager step.

    The eager path historically performed this update at the end of the
    forward pass.  A CUDA graph cannot execute that Python callback, so the
    graph runner calls this helper immediately before replay.  Keeping the
    device tensors at stable addresses makes the values visible to a captured
    graph while preserving the source schedule and the round1 host branch.
    """
    values: list[float] = []
    for i in range(len(ctx.model.attn_mask_probs)):
        value = ctx.train_model.mask_annealing(
            start_iter=ctx.train_model.model_args.attn_mask_annealing_steps_start[i],
            current_iter=step,
            final_iter=ctx.train_model.model_args.attn_mask_annealing_steps_end[i],
        )
        if isinstance(value, torch.Tensor):
            ctx.model.attn_mask_probs[i].copy_(value.reshape_as(ctx.model.attn_mask_probs[i]))
        else:
            values.append(float(value))
            ctx.model.attn_mask_probs[i].fill_(value)
    return tuple(values)


def scheduled_mask_probs(ctx: Ctx, step: int) -> tuple[float, ...]:
    """Pure form of the source mask schedule used by CUDA-graph signatures."""
    power = float(ctx.train_model.model_args.poly_power)
    result = []
    for start_iter, final_iter in zip(
        ctx.train_model.model_args.attn_mask_annealing_steps_start,
        ctx.train_model.model_args.attn_mask_annealing_steps_end,
    ):
        if step < start_iter:
            value = 1.0
        elif step >= final_iter:
            value = 0.0
        else:
            value = (1.0 - (step - start_iter) / (final_iter - start_iter)) ** power
        result.append(float(value))
    return tuple(result)


def train_step(ctx: Ctx, batch: Any, step: int) -> StepOut:
    from .fast_forward import _range
    images = batch["image"]
    autocast = ctx.fabric.autocast() if ctx.meta.get("enable_autocast") else nullcontext()
    with autocast:
        forward_fn = ctx.meta.get("forward_train_fn", forward_train_v2)
        with _range(ctx, "fwd"):
            mask_logits, class_logits = forward_fn(ctx, images, return_logits_per_layer=True)
        with _range(ctx, "loss"):
            loss, losses = ctx.ops.loss_all_layers(ctx, mask_logits, class_logits, batch["binary_masks"])
        ctx.train_model.train_metrics.update_with_losses({"loss": loss.detach()}, weight=len(images))
        if not ctx.meta.get("phase_override"):
            update_mask_schedule(ctx, step)
    with _range(ctx, "bwd"):
        ctx.fabric.backward(loss)
    with _range(ctx, "clip_opt_sched"):
        grad_norm = ctx.ops.opt_step(ctx)
    for hook in ctx.post_opt_hooks:
        hook(ctx)
    ctx.train_model.on_train_batch_end()
    return StepOut(loss.detach(), {key: value.detach() for key, value in losses.items()}, None if grad_norm is None else grad_norm.detach())


def preload_batches(ctx: Ctx, k: int = 8) -> list[Any]:
    """Load real batches; callers that time them should hold the GPU lock."""
    model_name = MODEL_NAMES[ctx.cfg.model]
    loader = build_loader(
        ctx.fabric, ctx.meta["data_args"], ctx.meta["train_transform_args"],
        DINOv3EoMTSemanticSegmentationTrain, model_name, "train", ctx.meta["batch_size"],
        min(ctx.cfg.num_workers, 4), ctx.cfg.seed,
    )
    result = []
    for batch in loader:
        result.append(batch)
        if len(result) >= k:
            break
    return result


def make_loader(ctx: Ctx, split: str):
    """Build a real Lightly loader using the context's resolved arguments."""
    model_name = MODEL_NAMES[ctx.cfg.model]
    transform_args = ctx.meta["train_transform_args"] if split == "train" else ctx.meta["val_transform_args"]
    return build_loader(
        ctx.fabric,
        ctx.meta["data_args"],
        transform_args,
        DINOv3EoMTSemanticSegmentationTrain,
        model_name,
        split,
        ctx.meta["batch_size"],
        ctx.cfg.num_workers,
        ctx.cfg.seed,
    )


# Late import avoids importing the large train model module during type-only setup.
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.train_model import (  # noqa: E402
    DINOv3EoMTSemanticSegmentationTrain,
)
