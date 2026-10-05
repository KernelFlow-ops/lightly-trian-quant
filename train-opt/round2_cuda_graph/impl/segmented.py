"""Partial-network CUDA graphs for the plain and joint EoMT block segments."""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from common.bench import assert_gpu_exclusive, gpu_lock, nvtx
from common.fast_forward import forward_train_v2


class PlainBlocks(nn.Module):
    def __init__(self, ctx: Any, H: int, W: int) -> None:
        super().__init__()
        self.ctx = ctx
        self.H, self.W = H, W
        self.blocks = nn.ModuleList(list(ctx.model.backbone.blocks[:-ctx.model.num_joint_blocks]))
        self.rope_embed = ctx.model.backbone.rope_embed
        self.rope_fn = ctx.ops.rope
        self.block_fn = ctx.ops.block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            rope = self.rope_fn(self.ctx, i, self.H, self.W) if self.rope_embed is not None else None
            x = self.block_fn(self.ctx, i, block, x, rope, None)
        return x


class JointBlocks(nn.Module):
    def __init__(self, ctx: Any, H: int, W: int, grid_size: tuple[int, int]) -> None:
        super().__init__()
        self.ctx = ctx
        self.H, self.W = H, W
        self.grid_size = grid_size
        self.first = len(ctx.model.backbone.blocks) - ctx.model.num_joint_blocks
        self.blocks = nn.ModuleList(list(ctx.model.backbone.blocks[self.first :]))
        self.rope_embed = ctx.model.backbone.rope_embed
        self.norm = ctx.model.backbone.norm
        self.class_head = ctx.model.class_head
        self.mask_head = ctx.model.mask_head
        self.upscale = ctx.model.upscale
        self.rope_fn = ctx.ops.rope
        self.predict_fn = ctx.ops.predict
        self.mask_fn = ctx.ops.build_mask
        self.block_fn = ctx.ops.block

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        for j, block in enumerate(self.blocks):
            i = self.first + j
            rope = self.rope_fn(self.ctx, i, self.H, self.W) if self.rope_embed is not None else None
            mask_logits, class_logits = self.predict_fn(self.ctx, x, self.grid_size)
            mask = self.mask_fn(self.ctx, j, mask_logits, self.grid_size, x.shape[1])
            x = self.block_fn(self.ctx, i, block, x, rope, mask)
            outputs.extend((mask_logits, class_logits))
        return (x, *outputs)


@dataclass
class GraphState:
    original: Callable[..., Any]
    variant: str
    max_joint_signatures: int
    plain: Callable[[torch.Tensor], torch.Tensor] | None = None
    plain_module: PlainBlocks | None = None
    plain_error: str | None = None
    joint: dict[tuple[str, ...], Callable[[torch.Tensor], tuple[torch.Tensor, ...]]] = field(default_factory=dict)
    joint_modules: dict[tuple[str, ...], JointBlocks] = field(default_factory=dict)
    joint_errors: dict[tuple[str, ...], str] = field(default_factory=dict)
    capture_s: dict[str, float] = field(default_factory=dict)
    compiled: Callable[[torch.Tensor], Any] | None = None
    compile_s: float | None = None
    compile_error: str | None = None
    compile_unique_graphs: int = 0
    input_signature: tuple[Any, ...] | None = None
    shape_fallbacks: int = 0

    def report(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "plain_captured": self.plain is not None,
            "plain_error": self.plain_error,
            "joint_signatures": [list(key) for key in self.joint],
            "joint_errors": {"/".join(key): value for key, value in self.joint_errors.items()},
            "capture_s": self.capture_s,
            "compile_s": self.compile_s,
            "compile_error": self.compile_error,
            "compile_unique_graphs": self.compile_unique_graphs,
            "input_signature": {"shape": list(self.input_signature[0]), "dtype": str(self.input_signature[1])} if self.input_signature is not None else None,
            "shape_fallbacks": self.shape_fallbacks,
        }


def _signature(ctx: Any) -> tuple[str, ...]:
    # prob=1 omits the source's rand call; 0<prob<1 includes it; prob=0 can
    # skip the dense mask. Keeping all three states preserves RNG semantics.
    return tuple("zero" if p == 0.0 else "one" if p == 1.0 else "middle" for p in ctx.host_probs)


def _capture(ctx: Any, module: nn.Module, actual: torch.Tensor, *, warmup: int) -> tuple[Callable, float]:
    sample = actual.detach().clone().requires_grad_(actual.requires_grad)
    started = time.perf_counter()
    with gpu_lock():
        assert_gpu_exclusive()
        device_index = actual.device.index if actual.device.index is not None else torch.cuda.current_device()
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if ctx.meta.get("enable_autocast") else nullcontext()
        with torch.random.fork_rng(devices=[device_index]), autocast:
            callable_graph = torch.cuda.make_graphed_callables(module, (sample,), num_warmup_iters=warmup)
        ctx.optimizer.zero_grad(set_to_none=True)
    return callable_graph, time.perf_counter() - started


def _plain(ctx: Any, x: torch.Tensor, state: GraphState, H: int, W: int) -> torch.Tensor:
    if state.plain is None and state.plain_error is None:
        module = PlainBlocks(ctx, H, W)
        try:
            state.plain, seconds = _capture(ctx, module, x, warmup=3)
            state.plain_module = module
            state.capture_s["plain"] = seconds
        except (RuntimeError, AssertionError) as exc:
            state.plain_error = str(exc)
            ctx.optimizer.zero_grad(set_to_none=True)
            state.plain_module = module
    if state.plain is not None:
        with nvtx("plain_graph") if ctx.nvtx else nullcontext():
            return state.plain(x)
    assert state.plain_module is not None
    return state.plain_module(x)


def _joint(ctx: Any, x: torch.Tensor, state: GraphState, H: int, W: int, grid_size: tuple[int, int]) -> tuple[torch.Tensor, ...]:
    signature = _signature(ctx)
    if signature not in state.joint and signature not in state.joint_errors:
        if len(state.joint) >= state.max_joint_signatures:
            state.joint_errors[signature] = f"signature cap {state.max_joint_signatures} reached; eager fallback"
        else:
            module = JointBlocks(ctx, H, W, grid_size)
            state.joint_modules[signature] = module
            try:
                state.joint[signature], seconds = _capture(ctx, module, x, warmup=3)
                state.capture_s["joint:" + "/".join(signature)] = seconds
            except (RuntimeError, AssertionError) as exc:
                state.joint_errors[signature] = str(exc)
                ctx.optimizer.zero_grad(set_to_none=True)
    callable_graph = state.joint.get(signature)
    if callable_graph is not None:
        with nvtx("joint_graph_" + "/".join(signature)) if ctx.nvtx else nullcontext():
            return callable_graph(x)
    module = state.joint_modules.get(signature)
    if module is None:
        module = JointBlocks(ctx, H, W, grid_size)
        state.joint_modules[signature] = module
    return module(x)


def segmented_forward(ctx: Any, images: torch.Tensor, return_logits_per_layer: bool = True) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    state: GraphState = ctx.meta["round2_graph_state"]
    if not return_logits_per_layer or not ctx.model.training or ctx.rng.mode != "off" or images.device.type != "cuda":
        return state.original(ctx, images, return_logits_per_layer)
    signature = (tuple(images.shape), images.dtype)
    if state.input_signature is None:
        state.input_signature = signature
    elif signature != state.input_signature:
        state.shape_fallbacks += 1
        return state.original(ctx, images, return_logits_per_layer)
    _, _, H, W = images.shape
    patch_size = ctx.model.backbone.patch_size
    if patch_size is None:
        return state.original(ctx, images, return_logits_per_layer)
    grid_size = (H // patch_size, W // patch_size)
    x, image_size = ctx.model.backbone.prepare_tokens_with_masks(images)
    x = _plain(ctx, x, state, image_size[0], image_size[1])
    x = torch.cat((ctx.model.queries.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)
    joint_outputs = _joint(ctx, x, state, image_size[0], image_size[1], grid_size)
    x = joint_outputs[0]
    mask_logits = list(joint_outputs[1::2])
    class_logits = list(joint_outputs[2::2])
    final_mask, final_class = ctx.ops.predict(ctx, x, grid_size)
    mask_logits.append(final_mask)
    class_logits.append(final_class)
    return mask_logits, class_logits


def _compile_forward(ctx: Any, state: GraphState, mode: str) -> Callable[..., Any]:
    def unique_graphs() -> int:
        from torch._dynamo.utils import counters

        return int(counters["stats"].get("unique_graphs", 0))

    def compiled_forward(c: Any, images: torch.Tensor, return_logits_per_layer: bool = True):
        if not return_logits_per_layer:
            return state.original(c, images, return_logits_per_layer)
        if state.compile_error is not None:
            return state.original(c, images, return_logits_per_layer)
        if state.compiled is None:
            with gpu_lock():
                assert_gpu_exclusive()
                started = time.perf_counter()
                before = unique_graphs()
                try:
                    state.compiled = torch.compile(lambda x: state.original(c, x, True), mode=mode, fullgraph=False)
                    result = state.compiled(images)
                except Exception as exc:
                    state.compile_error = str(exc)
                    state.compiled = None
                    return state.original(c, images, return_logits_per_layer)
                state.compile_s = time.perf_counter() - started
                state.compile_unique_graphs += unique_graphs() - before
                return result
        before = unique_graphs()
        try:
            result = state.compiled(images)
        except Exception as exc:
            state.compile_error = str(exc)
            state.compiled = None
            return state.original(c, images, return_logits_per_layer)
        state.compile_unique_graphs += unique_graphs() - before
        return result

    return compiled_forward


def install(ctx: Any, *, variant: str = "segmented", max_joint_signatures: int = 7) -> GraphState:
    if variant not in {"segmented", "compile_default", "compile_reduce_overhead", "compile_max_autotune_no_cudagraphs"}:
        raise ValueError(f"Unknown round2 variant: {variant}")
    if max_joint_signatures <= 0:
        raise ValueError("max_joint_signatures must be positive")
    original = ctx.meta.get("forward_train_fn", forward_train_v2)
    state = GraphState(original=original, variant=variant, max_joint_signatures=max_joint_signatures)
    ctx.meta["round2_graph_state"] = state
    if variant == "segmented":
        ctx.meta["forward_train_fn"] = segmented_forward
    else:
        mode = {
            "compile_default": "default",
            "compile_reduce_overhead": "reduce-overhead",
            "compile_max_autotune_no_cudagraphs": "max-autotune-no-cudagraphs",
        }[variant]
        ctx.meta["forward_train_fn"] = _compile_forward(ctx, state, mode)
    return state


def remove(ctx: Any) -> None:
    state: GraphState | None = ctx.meta.pop("round2_graph_state", None)
    if state is not None:
        ctx.meta["forward_train_fn"] = state.original
