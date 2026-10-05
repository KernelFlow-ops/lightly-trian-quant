"""Full-step CUDA graphs with stable optimizer addresses and mask signatures.

Only host branch changes (one/interior/zero), not each probability value,
require recapture. One graph is kept resident to fit the 5090 memory budget.
Scheduler, metrics and the source's post-forward mask schedule stay outside.
"""
from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch

from common.fast_forward import forward_train_v2
from common.harness import scheduled_mask_probs


def padded_batch(batch: dict[str, Any], device: torch.device, *, max_targets: int | None = None) -> dict[str, Any]:
    images = batch["image"].to(device=device, non_blocking=True)
    targets = batch["binary_masks"]
    if not isinstance(targets, (list, tuple)) or not targets or len(targets) != len(images):
        raise ValueError("expected one target dictionary per image")
    counts = [int(item["masks"].shape[0]) for item in targets]
    spatial = tuple(targets[0]["masks"].shape[-2:])
    if any(tuple(item["masks"].shape[-2:]) != spatial for item in targets):
        raise ValueError("all target masks must have the same spatial shape")
    capacity = max(1, max(counts)) if max_targets is None else max_targets
    if capacity < max(1, max(counts)):
        raise ValueError(f"max_targets={capacity} is smaller than batch maximum {max(counts)}")
    if capacity > 4:
        raise ValueError("full graph requires at most four targets per image")
    masks = torch.zeros((len(targets), capacity, *spatial), dtype=torch.uint8, device=device)
    labels = torch.zeros((len(targets), capacity), dtype=torch.long, device=device)
    valid = torch.zeros((len(targets), capacity), dtype=torch.bool, device=device)
    for index, (item, count) in enumerate(zip(targets, counts)):
        masks[index, :count].copy_(item["masks"], non_blocking=True)
        labels[index, :count].copy_(item["labels"], non_blocking=True)
        valid[index, :count] = True
    return {"image": images.clone(), "binary_masks": {"masks": masks, "labels": labels, "valid": valid}}


@dataclass
class GraphReport:
    captured: bool = False
    capture_error: str | None = None
    replays: int = 0
    captures: int = 0
    capture_s: float = 0.0
    max_targets: int | None = None
    image_shape: tuple[int, ...] | None = None
    signatures: list[Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FullGraphRunner:
    def __init__(self, ctx: Any, *, warmup: int = 3, max_targets: int = 2) -> None:
        if warmup < 1 or not 1 <= max_targets <= 4:
            raise ValueError("warmup must be positive and target capacity must be 1..4")
        self.ctx = ctx
        self.warmup = warmup
        self.max_targets = max_targets
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_batch: dict[str, Any] | None = None
        self.static_loss: torch.Tensor | None = None
        self.report = GraphReport(signatures=[])
        self._phase_key: tuple[str, ...] | None = None

    @property
    def ready(self) -> bool:
        return self.graph is not None and self.static_loss is not None

    @staticmethod
    def phase_key(values: Any) -> tuple[str, ...]:
        return tuple("one" if float(v) == 1.0 else "zero" if float(v) == 0.0 else "middle" for v in values)

    def _check_contract(self) -> None:
        ctx = self.ctx
        if not torch.cuda.is_available() or next(ctx.train_model.parameters()).device.type != "cuda":
            raise RuntimeError("full graph requires a CUDA model")
        if ctx.rng.mode != "off":
            raise RuntimeError("full graph requires RngTape mode='off'")
        if ctx.meta.get("round3_loss_matcher") is None:
            raise RuntimeError("full graph requires round3 padded loss")
        if ctx.meta.get("round7_optimizer") not in {"capturable_torch", "capturable_shadow", "cuda_multi_tensor"}:
            raise RuntimeError("full graph requires round7 capturable optimizer")
        if not ctx.meta.get("round1_annealing_host"):
            raise RuntimeError("full graph requires round1 nosync mask branches")
        if "round2_graph_state" in ctx.meta:
            raise RuntimeError("full graph cannot nest segmented CUDA graphs")
        if ctx.meta.get("noncapturable_reason"):
            raise RuntimeError(ctx.meta["noncapturable_reason"])

    def _body(self) -> torch.Tensor:
        assert self.static_batch is not None
        ctx = self.ctx
        autocast = ctx.fabric.autocast() if ctx.meta.get("enable_autocast") else nullcontext()
        with autocast:
            forward_fn = ctx.meta.get("forward_train_fn", forward_train_v2)
            masks, classes = forward_fn(ctx, self.static_batch["image"], return_logits_per_layer=True)
            loss, _ = ctx.ops.loss_all_layers(ctx, masks, classes, self.static_batch["binary_masks"])
        ctx.fabric.backward(loss)
        ctx.ops.opt_step(ctx)
        for hook in ctx.post_opt_hooks:
            hook(ctx)
        return loss

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "model": {k: v.detach().cpu().clone() for k, v in self.ctx.train_model.state_dict().items()},
            "optimizer": {p: {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                              for k, v in state.items()} for p, state in self.ctx.optimizer.state.items()},
            "shadow": {k: v.detach().cpu().clone() for k, v in self.ctx.shadow.items()},
            "cpu_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }

    def _restore_state(self, saved: dict[str, Any]) -> None:
        # Do not replace any tensor that the graph references. Newly allocated
        # moment buffers must return to zero; pre-existing state must survive.
        with torch.no_grad():
            for key, tensor in self.ctx.train_model.state_dict().items():
                tensor.copy_(saved["model"][key])
            for param, state in self.ctx.optimizer.state.items():
                old = saved["optimizer"].get(param, {})
                for key, tensor in state.items():
                    if isinstance(tensor, torch.Tensor):
                        if key in old:
                            tensor.copy_(old[key])
                        else:
                            tensor.zero_()
                    elif key in old:
                        state[key] = copy.deepcopy(old[key])
            for key, tensor in self.ctx.shadow.items():
                tensor.copy_(saved["shadow"][key])
            self.ctx.optimizer.zero_grad(set_to_none=False)
        torch.random.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])

    def capture(self, batch: dict[str, Any], *, max_targets: int | None = None, step: int = 0,
                phase_key: tuple[str, ...] | None = None) -> GraphReport:
        try:
            self._check_contract()
        except Exception as exc:
            self.report.capture_error = f"{type(exc).__name__}: {exc}"
            return self.report
        key = phase_key or self.phase_key(self.ctx.host_probs)
        if self.ready and key == self._phase_key:
            return self.report
        self.graph = None
        self.static_loss = None
        self.report.captured = False
        self.report.capture_error = None
        device = next(self.ctx.train_model.parameters()).device
        self.static_batch = padded_batch(batch, device, max_targets=max_targets or self.max_targets)
        self.report.max_targets = self.static_batch["binary_masks"]["masks"].shape[1]
        self.report.image_shape = tuple(self.static_batch["image"].shape)
        snapshot = self._snapshot_state()
        old_capture = self.ctx.meta.get("round7_graph_capture", False)
        self.ctx.meta["round7_graph_capture"] = True
        started = time.perf_counter()
        try:
            for _ in range(self.warmup):
                self._body()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.static_loss = self._body()
            self.graph = graph
            self._phase_key = key
            self.report.captures += 1
            self.report.signatures.append(list(key))
            self.report.captured = True
        except Exception as exc:
            self.graph = None
            self.static_loss = None
            self.report.capture_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._restore_state(snapshot)
            self.ctx.meta["round7_graph_capture"] = old_capture
            self.report.capture_s += time.perf_counter() - started
        return self.report

    def ensure_phase(self, batch: dict[str, Any], step: int) -> GraphReport:
        # The source updates probabilities after forward(step). Forward(step+1)
        # therefore uses the values computed at step, including boundary draws.
        values = scheduled_mask_probs(self.ctx, max(0, step - 1))
        for i, value in enumerate(values):
            self.ctx.model.attn_mask_probs[i].fill_(value)
        self.ctx.host_probs[:] = values
        key = self.phase_key(values)
        if self.ready and key == self._phase_key:
            return self.report
        return self.capture(batch, step=step, phase_key=key)

    def replay(self, batch: dict[str, Any], *, step: int | None = None) -> torch.Tensor:
        if not self.ready or self.static_batch is None:
            raise RuntimeError("full graph has not been captured")
        incoming = padded_batch(batch, self.static_batch["image"].device, max_targets=self.report.max_targets)
        if incoming["image"].shape != self.static_batch["image"].shape:
            raise ValueError("full graph replay image shape differs from capture")
        self.static_batch["image"].copy_(incoming["image"], non_blocking=True)
        for key in ("masks", "labels", "valid"):
            self.static_batch["binary_masks"][key].copy_(incoming["binary_masks"][key], non_blocking=True)
        self.graph.replay()
        self.ctx.meta["round7_scheduler_after_graph"]()
        self.report.replays += 1
        return self.static_loss.detach().clone()
