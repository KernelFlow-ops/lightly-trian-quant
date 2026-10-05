"""Harness training loop with Lightly checkpoint and export contracts."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from common.harness import make_loader, train_step
from lightly_train._commands import train_task_helpers as helpers
from lightly_train._data.infinite_cycle_iterator import InfiniteCycleIterator


LICENSE_INFO = (
    "LightlyTrain License Notice\n\n"
    "Model training and inference in commercial settings require a valid Commercial License.\n"
    "Please ensure your usage complies with the applicable LightlyTrain license.\n"
    "See https://docs.lightly.ai/train/stable/index.html#license for more details.\n"
    "Contact us at https://www.lightly.ai/contact to discuss the best licensing option for your use case."
)


def _checkpoint_state(ctx: Any, train_loader: Any) -> dict[str, Any]:
    task_model = ctx.train_model.get_task_model()
    return {
        "train_model": ctx.train_model,
        "optimizer": ctx.optimizer,
        "scheduler": ctx.scheduler,
        "train_dataloader": train_loader,
        "step": -1,
        "model_class_path": task_model.class_path,
        "model_init_args": task_model.init_args,
        "license_info": LICENSE_INFO,
    }


def _export_dict(ctx: Any, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_class_path": state["model_class_path"],
        "model_init_args": state["model_init_args"],
        "train_model": ctx.train_model.get_export_state_dict(),
        "license_info": state["license_info"],
    }


def _save(ctx: Any, out: Path, state: dict[str, Any], which: str) -> None:
    helpers.save_checkpoint(fabric=ctx.fabric, out_dir=out, state=state, best_or_last=which)
    helpers.export_model(out_dir=out, model_dict=_export_dict(ctx, state), best_or_last=which)


def _advance_transform(loader: Any, iterator: InfiniteCycleIterator, step: int, workers: int) -> None:
    transform = loader.dataset.transform
    collate = loader.collate_fn
    transform.set_step(step)
    collate.set_step(step)
    needs_reinit = transform.requires_dataloader_reinitialization() or collate.requires_dataloader_reinitialization()
    if workers > 0 and needs_reinit:
        iterator.reset()
        transform.mark_dataloader_as_reinitialized()
        collate.mark_dataloader_as_reinitialized()


def _validate(ctx: Any, loader: Any) -> Any:
    ctx.train_model.eval()
    context_factory = ctx.meta.get("validation_context")
    context = context_factory() if context_factory is not None else torch.no_grad()
    try:
        with context:
            for val_step, batch in enumerate(loader):
                result = ctx.train_model.validation_step(fabric=ctx.fabric, batch=batch, step=val_step)
        aggregated = result.metrics.compute_aggregated_values()
        result.metrics.reset()
        return aggregated
    finally:
        ctx.train_model.set_train_mode()


def run_fast(ctx: Any, out: Path, *, epochs: int, save_checkpoints: bool = True) -> list[dict[str, Any]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if ctx.meta.get("checkpoint_writer_factory") is not None and save_checkpoints:
        raise RuntimeError("ckpt_async requires a validated Fabric-state snapshot adapter; use synchronous Lightly helpers for compatible checkpoints")
    loader_factory = ctx.meta.get("loader_factory", make_loader)
    train_loader = loader_factory(ctx, "train")
    val_loader = loader_factory(ctx, "val")
    iterator = InfiniteCycleIterator(iterable=train_loader)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    if total_steps != ctx.meta["total_steps"]:
        raise ValueError(f"ctx total_steps={ctx.meta['total_steps']} differs from {steps_per_epoch} x {epochs}")
    state = _checkpoint_state(ctx, train_loader)
    best = None
    history: list[dict[str, Any]] = []
    graph_runner = ctx.meta.get("full_graph_runner")
    if graph_runner is not None and save_checkpoints:
        # The graph owns optimizer buffers and checkpoints must snapshot them
        # after the replay.  Keep the same synchronous Lightly helper path.
        ctx.meta["full_graph_checkpointing"] = True
    metrics_path = out / "metrics.jsonl"
    val_log_path = out / "val_metrics.log"
    for epoch in range(epochs):
        train_start = time.perf_counter()
        for step in range(epoch * steps_per_epoch, (epoch + 1) * steps_per_epoch):
            _advance_transform(train_loader, iterator, step, ctx.cfg.num_workers)
            batch = next(iterator)
            if graph_runner is not None:
                graph_runner.ensure_phase(batch, step)
                if graph_runner.ready:
                    graph_loss = graph_runner.replay(batch, step=step)
                    ctx.train_model.train_metrics.update_with_losses({"loss": graph_loss}, weight=len(batch["image"]))
                    # Keep the post-step hook used by Lightly's training loop
                    # outside the graph; it mutates Python-side metric state.
                    ctx.train_model.on_train_batch_end()
                else:
                    raise RuntimeError(f"CUDA graph capture failed: {graph_runner.report.capture_error}")
            else:
                train_step(ctx, batch, step)
            state["step"] = step
        torch.cuda.synchronize()
        train_s = time.perf_counter() - train_start
        train_metrics = ctx.train_model.train_metrics.compute_aggregated_values().metric_values
        ctx.train_model.train_metrics.reset()

        ckpt_s = 0.0
        if save_checkpoints:
            begin = time.perf_counter()
            _save(ctx, out, state, "last")
            ckpt_s += time.perf_counter() - begin

        val_start = time.perf_counter()
        val_result = _validate(ctx, val_loader)
        torch.cuda.synchronize()
        val_s = time.perf_counter() - val_start
        best = helpers.get_best_metrics(
            best_agg_metric_values=best,
            last_agg_metric_values=val_result,
            step=state["step"],
            metric_args=ctx.train_model.metric_args,
        )
        if save_checkpoints and best.step == state["step"]:
            begin = time.perf_counter()
            _save(ctx, out, state, "best")
            ckpt_s += time.perf_counter() - begin

        row = {
            "epoch": epoch + 1,
            "step": state["step"],
            "train_metrics": train_metrics,
            "val_metrics": val_result.metric_values,
            "val_miou": val_result.metric_values.get("val_metric/miou"),
            "val_head_iou": val_result.metric_values.get("val_metric_classwise/iou_head"),
            "best_step": best.step,
            "epoch_s": {"train": train_s, "val": val_s, "ckpt": ckpt_s, "total": train_s + val_s + ckpt_s},
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        with val_log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"epoch: {row['epoch']}\n")
            stream.write(f"val_metric/miou: {row['val_miou']}\n")
            stream.write(f"val_metric_classwise/iou_head: {row['val_head_iou']}\n")
        print(json.dumps({"epoch": row["epoch"], "val_miou": row["val_miou"], "val_head_iou": row["val_head_iou"], "epoch_s": row["epoch_s"]}), flush=True)
    return history
