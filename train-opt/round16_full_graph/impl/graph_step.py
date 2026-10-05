"""Whole-step CUDA graph runner (round16): forward, r3 padded loss, backward, clip + AdamW (round15 flat optimizer),
device-side lr / step counters, all in ONE graph replay per training step.

Design (see NOTES.md):

* Static input buffers (image, padded target masks / labels / valid).  The next batch is copied host -> pinned -> device
  staging buffers on a side stream while the current replay runs; the compute stream then does one cheap D2D copy staging
  -> static right before ``replay()`` (so the H2D never waits for the replay and the graph never reads a half-written input).
* One graph per mask-schedule *signature* = per joint block {one, middle, zero} (the host branch of the source's mask code).
  Probability values inside a state are device data (``attn_mask_probs``), refreshed with a tiny ``fill_`` per step in the
  ``middle`` state only.  Graphs are captured lazily, share ONE memory pool and are kept in an LRU (``max_resident``).
* Before the first capture every signature the schedule will visit is warmed up eagerly (JIT / library lazy init / cuDNN
  plans), without touching the optimizer (grads are re-zeroed).  Capture itself does not execute kernels, so it does not move
  the weights.
* The optimizer must be capturable: round15 ``FlatAdamW`` (``ctx.meta['round15_optimizer_flat']``).  The host scheduler and the
  train metrics stay outside the graph (the scheduler is not needed: lr is a device function of a device counter).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from common.fast_forward import forward_train_v2
from common.harness import StepOut, scheduled_mask_probs, train_step


def phase_key(values: Sequence[float]) -> tuple[str, ...]:
    return tuple("one" if float(v) == 1.0 else "zero" if float(v) == 0.0 else "middle" for v in values)


def target_counts(batch: dict[str, Any]) -> list[int]:
    return [int(item["masks"].shape[0]) for item in batch["binary_masks"]]


def padded_targets(batch: dict[str, Any], max_targets: int) -> dict[str, torch.Tensor]:
    """r3 padded target dict ``{masks uint8 [B,T,H,W], labels [B,T], valid [B,T]}`` for the eager reference path."""
    items = batch["binary_masks"]
    B, T = len(items), int(max_targets)
    H, W = items[0]["masks"].shape[-2:]
    dev = batch["image"].device
    masks = torch.zeros((B, T, H, W), dtype=torch.uint8, device=dev)
    labels = torch.zeros((B, T), dtype=torch.long, device=dev)
    valid = torch.zeros((B, T), dtype=torch.bool, device=dev)
    for i, item in enumerate(items):
        n = int(item["masks"].shape[0])
        if n > T:
            raise ValueError(f"image {i} has {n} targets > max_targets={T}")
        masks[i, :n] = item["masks"].to(dev)
        labels[i, :n] = item["labels"].to(dev)
        valid[i, :n] = True
    return {"masks": masks, "labels": labels, "valid": valid}


@dataclass
class GraphEntry:
    key: tuple[str, ...]
    graph: torch.cuda.CUDAGraph
    capture_s: float
    replays: int = 0


@dataclass
class Report:
    captures: int = 0
    capture_s: list[float] = field(default_factory=list)
    keys: list[list[str]] = field(default_factory=list)
    warm_s: float = 0.0
    evictions: int = 0
    eager_fallbacks: int = 0
    replays: int = 0
    reserved_gb_after_capture: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class StepGraphRunner:
    def __init__(self, ctx: Any, example: dict[str, Any], *, max_targets: int = 2, max_resident: int = 8,
                 warmup_iters: int = 2, metrics: str = "lazy", side_stream: bool = True, pin_host: bool = True) -> None:
        self.ctx = ctx
        self.flat = ctx.meta.get("round15_optimizer_flat")
        if self.flat is None:
            raise RuntimeError("StepGraphRunner needs the round15 flat optimizer (capturable clip + AdamW + device lr)")
        if getattr(self.flat, "external_lr", False):
            raise RuntimeError("the lr schedule is not a device function (external lr): not capturable")
        if ctx.rng.mode != "off":
            raise RuntimeError("graph mode requires RngTape(mode='off')")
        if ctx.meta.get("round3_loss_matcher") is None:
            raise RuntimeError("graph mode requires the round3 padded loss")
        self.device = next(ctx.train_model.parameters()).device
        self.max_targets = int(max_targets)
        self.max_resident = int(max_resident)
        self.warmup_iters = int(warmup_iters)
        # train metrics: "step" = update_with_losses every step (torchmetrics' nan check syncs the host), "lazy" = only on
        # ``update_metrics()`` (the training loss metric is overwritten each step, so the latest loss is all that is needed), "off"
        self.metrics = {True: "step", False: "off"}.get(metrics, metrics)
        if self.metrics not in ("step", "lazy", "off"):
            raise ValueError(self.metrics)
        self.use_side = side_stream
        self.pin_host = pin_host
        img = example["image"]
        B = img.shape[0]
        self.B = B
        self.image_shape = tuple(img.shape)
        H, W = example["binary_masks"][0]["masks"].shape[-2:]
        T = self.max_targets

        def buffers():
            return {
                "image": torch.zeros(self.image_shape, dtype=torch.float32, device=self.device),
                "masks": torch.zeros((B, T, H, W), dtype=torch.uint8, device=self.device),
                "labels": torch.zeros((B, T), dtype=torch.long, device=self.device),
                "valid": torch.zeros((B, T), dtype=torch.bool, device=self.device),
            }

        self.static = buffers()                # graph inputs
        self.staging = buffers()               # filled on the side stream
        self.targets = {k: self.static[k] for k in ("masks", "labels", "valid")}
        self.pinned = [self._pinned_set(B, T, H, W) for _ in range(2)] if pin_host else None
        self._pin_events: list[torch.cuda.Event | None] = [None, None]
        self._pin_i = 0
        self.side = torch.cuda.Stream(device=self.device) if side_stream else None
        self._staged = torch.cuda.Event()
        self._consumed = torch.cuda.Event()
        self._consumed.record(torch.cuda.current_stream())
        self._staged_id: int | None = None
        self.s_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        self.s_norm = torch.zeros((), dtype=torch.float32, device=self.device)
        self.probs = ctx.model.attn_mask_probs
        self._last_probs: list[float | None] = [None] * len(self.probs)
        self.mempool = torch.cuda.graph_pool_handle()
        self.graphs: OrderedDict[tuple[str, ...], GraphEntry] = OrderedDict()
        self.warmed: set[tuple[str, ...]] = set()
        self.report = Report()

    # ------------------------------------------------------------------ input staging
    def _pinned_set(self, B: int, T: int, H: int, W: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.zeros(self.image_shape, dtype=torch.float32).pin_memory(),
            "masks": torch.zeros((B, T, H, W), dtype=torch.uint8).pin_memory(),
            "labels": torch.zeros((B, T), dtype=torch.long).pin_memory(),
            "valid": torch.zeros((B, T), dtype=torch.bool).pin_memory(),
        }

    def fits(self, batch: dict[str, Any]) -> bool:
        return tuple(batch["image"].shape) == self.image_shape and max(target_counts(batch)) <= self.max_targets

    def _fill(self, dst: dict[str, torch.Tensor], batch: dict[str, Any], *, non_blocking: bool) -> None:
        """Write ``batch`` into ``dst`` buffers (padded targets).  ``dst`` and ``batch`` may live on different devices."""
        dst["image"].copy_(batch["image"], non_blocking=non_blocking)
        dst["masks"].zero_()
        dst["labels"].zero_()
        dst["valid"].zero_()
        for i, item in enumerate(batch["binary_masks"]):
            n = int(item["masks"].shape[0])
            if n:
                dst["masks"][i, :n].copy_(item["masks"], non_blocking=non_blocking)
                dst["labels"][i, :n].copy_(item["labels"], non_blocking=non_blocking)
                dst["valid"][i, :n] = True

    @torch.no_grad()
    def stage(self, batch: dict[str, Any], after: "torch.cuda.Event | None" = None) -> None:
        """Asynchronously load ``batch`` into the staging buffers on the side stream (overlaps the running replay).

        ``after``: event recorded on the compute stream once the producers of a GPU-resident ``batch`` are done (the
        caller records it BEFORE launching the replay, so the side stream does not end up waiting for the replay)."""
        if not self.fits(batch):
            raise ValueError("batch does not fit the static graph inputs")
        stream = self.side if self.use_side else torch.cuda.current_stream()
        on_gpu = batch["image"].is_cuda
        if self.use_side:
            if on_gpu:
                if after is not None:
                    stream.wait_event(after)
                else:
                    stream.wait_stream(torch.cuda.current_stream())
            stream.wait_event(self._consumed)                    # previous staging content already copied to static
        with torch.cuda.stream(stream):
            if on_gpu or not self.pin_host:
                if on_gpu:
                    for t in [batch["image"], *[x["masks"] for x in batch["binary_masks"]], *[x["labels"] for x in batch["binary_masks"]]]:
                        t.record_stream(stream)
                self._fill(self.staging, batch, non_blocking=True)
            else:
                i = self._pin_i
                self._pin_i ^= 1
                if self._pin_events[i] is not None:
                    self._pin_events[i].synchronize()            # earlier H2D from this pinned set finished
                self._fill(self.pinned[i], batch, non_blocking=False)  # host-side pad into pinned memory
                for k in self.staging:
                    self.staging[k].copy_(self.pinned[i][k], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(stream)
                self._pin_events[i] = ev
            self._staged.record(stream)
        self._staged_id = id(batch)

    def _consume(self) -> None:
        cur = torch.cuda.current_stream()
        if self.use_side:
            cur.wait_event(self._staged)
        for k in self.static:
            self.static[k].copy_(self.staging[k])
        self._consumed.record(cur)

    # ------------------------------------------------------------------ signatures / probabilities
    def values_for(self, step: int) -> tuple[float, ...]:
        if self.ctx.meta.get("phase_override"):
            return tuple(float(v) for v in self.ctx.host_probs)
        return scheduled_mask_probs(self.ctx, max(0, step - 1))

    def _set_probs(self, values: Sequence[float]) -> None:
        self.ctx.host_probs[:] = list(values)
        for j, v in enumerate(values):
            if self._last_probs[j] != v:
                self.probs[j].fill_(v)
                self._last_probs[j] = v

    def sync_probs_state(self) -> None:
        """Call after anything else (eager set_phase / update_mask_schedule) changed attn_mask_probs."""
        self._last_probs = [None] * len(self.probs)

    def keys_to_visit(self, start: int = 0, end: int | None = None) -> list[tuple[str, ...]]:
        if self.ctx.meta.get("phase_override"):
            return [phase_key(self.ctx.host_probs)]
        end = int(self.ctx.meta.get("total_steps", 1)) if end is None else end
        seen: dict[tuple[str, ...], None] = {}
        for s in range(start, end):
            seen.setdefault(phase_key(scheduled_mask_probs(self.ctx, max(0, s - 1))), None)
        return list(seen)

    # ------------------------------------------------------------------ the step body
    def _fwd_bwd(self) -> torch.Tensor:
        ctx = self.ctx
        autocast = ctx.fabric.autocast() if ctx.meta.get("enable_autocast") else nullcontext()
        with autocast:
            forward_fn = ctx.meta.get("forward_train_fn", forward_train_v2)
            masks, classes = forward_fn(ctx, self.static["image"], return_logits_per_layer=True)
            loss, _ = ctx.ops.loss_all_layers(ctx, masks, classes, self.targets)
        ctx.fabric.backward(loss)
        return loss

    def _body(self) -> None:
        ctx = self.ctx
        loss = self._fwd_bwd()
        norm = ctx.ops.opt_step(ctx)
        for hook in ctx.post_opt_hooks:
            hook(ctx)
        self.s_loss.copy_(loss.detach())
        if norm is not None:
            self.s_norm.copy_(norm.detach())

    # ------------------------------------------------------------------ warmup / capture
    @torch.no_grad()
    def _noop(self) -> None:
        pass

    def warm(self, keys: Sequence[tuple[str, ...]], values_of: Callable[[tuple[str, ...]], Sequence[float]]) -> None:
        """Eager fwd+bwd for every signature (no optimizer update; gradients re-zeroed)."""
        t0 = time.perf_counter()
        for key in keys:
            if key in self.warmed:
                continue
            self._set_probs(values_of(key))
            for _ in range(self.warmup_iters):
                with torch.enable_grad():
                    self._fwd_bwd()
                if not self.flat.adopted:
                    self.flat.adopt_grads()
                self.flat.zero_grad()
            self.warmed.add(key)
        torch.cuda.synchronize(self.device)
        self.report.warm_s += time.perf_counter() - t0

    @staticmethod
    def _rep_values(key: tuple[str, ...]) -> tuple[float, ...]:
        return tuple({"one": 1.0, "zero": 0.0, "middle": 0.5}[s] for s in key)

    def capture(self, key: tuple[str, ...], values: Sequence[float]) -> GraphEntry:
        if key not in self.warmed:
            self.warm([key], lambda k: values)
        while len(self.graphs) >= self.max_resident:
            torch.cuda.synchronize(self.device)
            old = self.graphs.popitem(last=False)[1]
            del old
            self.report.evictions += 1
            if not self.graphs:
                # the pool's use count dropped to 0: the allocator has released it, a new handle is required
                self.mempool = torch.cuda.graph_pool_handle()
        self._set_probs(values)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        self.flat.graph_mode = True
        try:
            with torch.cuda.graph(graph, pool=self.mempool):
                self._body()
        except Exception as exc:
            self.report.errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
            raise
        finally:
            self.flat.graph_mode = False
        torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        entry = GraphEntry(key, graph, dt)
        self.graphs[key] = entry
        self.report.captures += 1
        self.report.capture_s.append(dt)
        self.report.keys.append(list(key))
        self.report.reserved_gb_after_capture.append(torch.cuda.memory_reserved() / 2**30)
        return entry

    # ------------------------------------------------------------------ run
    def prepare(self, batch: dict[str, Any], start_step: int = 0) -> None:
        """Warm every signature the run will visit (call once, before timing)."""
        self.stage(batch)
        self._consume()
        keys = self.keys_to_visit(start_step)
        step_values: dict[tuple[str, ...], Sequence[float]] = {}
        if self.ctx.meta.get("phase_override"):
            step_values[keys[0]] = tuple(self.ctx.host_probs)
        else:
            for s in range(start_step, int(self.ctx.meta.get("total_steps", 1))):
                v = scheduled_mask_probs(self.ctx, max(0, s - 1))
                step_values.setdefault(phase_key(v), v)
        self.warm(keys, lambda k: step_values.get(k, self._rep_values(k)))
        self.sync_probs_state()

    def step(self, batch: dict[str, Any], step: int, next_batch: dict[str, Any] | None = None) -> StepOut:
        if not self.fits(batch):
            return self._eager_step(batch, step)
        values = self.values_for(step)
        key = phase_key(values)
        if self._staged_id != id(batch):
            self.stage(batch)
        self._consume()
        self._staged_id = None
        entry = self.graphs.get(key)
        if entry is None:
            entry = self.capture(key, values)
        else:
            self.graphs.move_to_end(key)
        self._set_probs(values)
        ready = None
        pipeline = next_batch is not None and self.fits(next_batch)
        if pipeline and next_batch["image"].is_cuda:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
        entry.graph.replay()
        entry.replays += 1
        self.report.replays += 1
        if pipeline:
            self.stage(next_batch, after=ready)
        if self.metrics == "step":
            self.ctx.train_model.train_metrics.update_with_losses({"loss": self.s_loss}, weight=self.B)
        return StepOut(self.s_loss.clone(), {}, self.s_norm.clone())

    def update_metrics(self) -> None:
        """Feed the latest loss to the train metrics (call before ``train_metrics.compute_aggregated_values()`` at log steps)."""
        self.ctx.train_model.train_metrics.update_with_losses({"loss": self.s_loss.detach().clone()}, weight=self.B)

    def _eager_step(self, batch: dict[str, Any], step: int) -> StepOut:
        """Overflow / odd-shape fallback: the ordinary eager step with the flat optimizer."""
        self.report.eager_fallbacks += 1
        self.sync_probs_state()
        return train_step(self.ctx, batch, step)

    def as_step_fn(self, batches: Sequence[dict[str, Any]]) -> Callable[[Any, Any, int], StepOut]:
        index = {id(b): i for i, b in enumerate(batches)}

        def step_fn(ctx: Any, batch: dict[str, Any], step: int) -> StepOut:
            i = index.get(id(batch))
            nxt = batches[(i + 1) % len(batches)] if i is not None else None
            return self.step(batch, step, nxt)

        return step_fn

    def release(self) -> None:
        torch.cuda.synchronize(self.device)
        self.graphs.clear()
