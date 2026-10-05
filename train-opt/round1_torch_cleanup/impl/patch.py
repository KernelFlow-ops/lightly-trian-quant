"""Independent PyTorch cleanup variants for the source-fp32 training path."""
from __future__ import annotations

from contextlib import nullcontext
from types import MethodType

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .mask import attention_with_broadcast, build_mask
from .rope import plain_rope_batch

NAME = "round1_torch_cleanup"
REQUIRES: tuple[str, ...] = ()
CONFLICTS: tuple[str, ...] = ()

BACKENDS = {
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
    "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_math": SDPBackend.MATH,
}
COMPONENTS = {
    "autocast_bf16", "nosync", "mask_skip", "rope_batch", "adamw_fused",
    "cudnn_bench", "head_cl", "kmask_bias", "loader", "val_fast", "ckpt_async",
    *BACKENDS,
}


def _install_nosync(ctx) -> None:
    num_joint = len(ctx.model.attn_mask_probs)
    calls = {"anneal": 0, "disable": 0}
    power = float(ctx.train_model.model_args.poly_power)

    def annealing(self, start_iter: int, current_iter: int, final_iter: int) -> float:
        if current_iter < start_iter:
            value = 1.0
        elif current_iter >= final_iter:
            value = 0.0
        else:
            value = (1.0 - (current_iter - start_iter) / (final_iter - start_iter)) ** power
        j = calls["anneal"] % num_joint
        calls["anneal"] += 1
        ctx.host_probs[j] = value
        return value

    def disable(self, attn_mask: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        j = calls["disable"] % num_joint
        calls["disable"] += 1
        if ctx.host_probs[j] < 1.0:
            q = self.num_queries
            random_queries = ctx.rng.rand(f"mask/j{j}", (attn_mask.shape[0], q), device=attn_mask.device) > prob
            attn_mask[:, :q, q + 1 + self.backbone.n_storage_tokens :][random_queries] = True
        return attn_mask

    ctx.train_model.mask_annealing = MethodType(annealing, ctx.train_model)
    ctx.model._disable_attn_mask = MethodType(disable, ctx.model)
    ctx.meta["round1_annealing_host"] = True


def _install_mask(ctx, *, skip_zero: bool) -> None:
    def mask_fn(c, j, mask_logits, grid_size, n_tokens):
        return build_mask(c, j, mask_logits, grid_size, n_tokens, skip_zero=skip_zero, broadcast=skip_zero)

    ctx.ops.build_mask = mask_fn


def _install_attention(ctx, backend_name: str | None, *, broadcast: bool) -> None:
    original = ctx.ops.attn
    backend = BACKENDS.get(backend_name)

    def attn(c, block, x_normed, rope, mask):
        context = sdpa_kernel(backends=[backend]) if backend is not None else nullcontext()
        with context:
            if broadcast and mask is not None and mask.ndim == 4:
                return attention_with_broadcast(c, block, x_normed, rope, mask)
            return original(c, block, x_normed, rope, mask)

    ctx.ops.attn = attn


def _install_rope_batch(ctx) -> None:
    original = ctx.ops.rope
    plain_blocks = len(ctx.model.backbone.blocks) - ctx.model.num_joint_blocks
    cache: list[tuple[torch.Tensor, torch.Tensor]] = []

    def rope(c, block_idx: int, H: int, W: int):
        nonlocal cache
        module = c.model.backbone.rope_embed
        if block_idx >= plain_blocks or not module.training:
            return original(c, block_idx, H, W)
        if module.shift_coords is not None or module.jitter_coords is not None or module.rescale_coords is None:
            return original(c, block_idx, H, W)
        if block_idx != 0 and len(cache) != plain_blocks:
            return original(c, block_idx, H, W)
        if block_idx == 0:
            cache = plain_rope_batch(module, plain_blocks, H, W)
        return cache[block_idx]

    ctx.ops.rope = rope


def _install_adamw_fused(ctx) -> None:
    optimizer = getattr(ctx.optimizer, "optimizer", ctx.optimizer)
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("adamw_fused requires the source torch.optim.AdamW optimizer")
    optimizer.defaults["fused"] = True
    optimizer.defaults["foreach"] = None
    optimizer._step_supports_amp_scaling = True
    for group in optimizer.param_groups:
        group["fused"] = True
        group["foreach"] = None


def _install_head_cl(ctx) -> None:
    upscale = ctx.model.upscale
    for module in upscale.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            for parameter in module.parameters(recurse=False):
                if parameter.ndim == 4:
                    parameter.data = parameter.data.contiguous(memory_format=torch.channels_last)
    original = upscale.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        return original(x.contiguous(memory_format=torch.channels_last))

    upscale.forward = forward


def _install_kmask_bias(ctx) -> int:
    from lightly_train._models.dinov3.dinov3_src.layers.attention import LinearKMaskedBias

    installed = 0
    for module in ctx.model.modules():
        if not isinstance(module, LinearKMaskedBias) or module.bias is None:
            continue
        if not module.weight.requires_grad or not module.bias.requires_grad:
            continue
        if not bool(torch.all(module.bias_mask == 0)):
            continue
        # Source output and bias gradient are zero for this buffer. Keep a zero
        # grad present so AdamW still advances the bias's state and decay.
        zero_grad = torch.zeros_like(module.bias)

        def set_bias_grad(grad: torch.Tensor, *, bias=module.bias, zero=zero_grad) -> torch.Tensor:
            bias.grad = zero
            return grad

        module.weight.register_hook(set_bias_grad)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.weight, None)

        module.forward = MethodType(forward, module)
        installed += 1
    return installed


def _components(variant: str) -> list[str]:
    if variant in {"", "default"}:
        return []
    requested = [item.strip() for item in variant.replace(",", "+").split("+") if item.strip()]
    unknown = set(requested) - COMPONENTS
    if unknown:
        raise ValueError(f"Unsupported round1 variants: {sorted(unknown)}")
    if "mask_skip" in requested and "nosync" not in requested:
        requested.insert(0, "nosync")
    if len(set(requested) & BACKENDS.keys()) > 1:
        raise ValueError("Only one SDPA backend may be selected")
    return list(dict.fromkeys(requested))


def apply(ctx, variant: str = "default", *, workers: int | None = None, **kwargs) -> dict:
    if kwargs:
        raise ValueError(f"Unsupported round1 options: {sorted(kwargs)}")
    requested = _components(variant)
    prior = set(ctx.meta.get("round1_components", ()))
    new = [item for item in requested if item not in prior]
    if any(item.startswith("sdpa_") for item in new) and any(item.startswith("sdpa_") for item in prior):
        raise RuntimeError("Cannot change the SDPA backend after installing round1")
    changed_ops: list[str] = []
    if "autocast_bf16" in new:
        if ctx.cfg.precision != "bf16-mixed":
            raise ValueError("autocast_bf16 requires precision='bf16-mixed'")
        ctx.meta["enable_autocast"] = True
        ctx.meta["round1_autocast_experimental"] = True
    if "nosync" in new:
        _install_nosync(ctx)
        changed_ops.append("build_mask")
    if "nosync" in new or "mask_skip" in new:
        _install_mask(ctx, skip_zero="mask_skip" in prior or "mask_skip" in new)
        changed_ops.append("build_mask")
    active = prior | set(requested)
    backend_name = next((item for item in active if item in BACKENDS), None)
    if "mask_skip" in new or backend_name is not None and backend_name in new:
        _install_attention(ctx, backend_name, broadcast="mask_skip" in active)
        changed_ops.append("attn")
    if "rope_batch" in new:
        _install_rope_batch(ctx)
        changed_ops.append("rope")
    if "adamw_fused" in new:
        _install_adamw_fused(ctx)
    if "cudnn_bench" in new:
        torch.backends.cudnn.benchmark = True
    if "head_cl" in new:
        _install_head_cl(ctx)
    if "kmask_bias" in new:
        ctx.meta["round1_kmask_installed"] = _install_kmask_bias(ctx)
        ctx.meta["round1_kmask_experimental"] = True
    if "loader" in new:
        workers = ctx.cfg.num_workers if workers is None else workers
        if workers < 0:
            raise ValueError("loader variant requires workers >= 0")
        ctx.cfg.num_workers = workers
        ctx.meta["loader_workers"] = workers
    elif workers is not None:
        raise ValueError("workers is only valid with loader variant")
    if "val_fast" in new or "ckpt_async" in new:
        from .io import AsyncCheckpointWriter, validation_context

        if "val_fast" in new:
            ctx.meta["validation_context"] = validation_context
        if "ckpt_async" in new:
            ctx.meta["checkpoint_writer_factory"] = AsyncCheckpointWriter
    ctx.meta["round1_components"] = tuple(dict.fromkeys((*ctx.meta.get("round1_components", ()), *new)))
    return {"ops": list(dict.fromkeys(changed_ops)), "desc": f"round1: {variant or 'default'}", "experimental": any(item in requested for item in ("autocast_bf16", "kmask_bias"))}
