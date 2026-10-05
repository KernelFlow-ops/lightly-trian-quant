"""Default operation registry used by the harness and runtime patches."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


class Ops(SimpleNamespace):
    """Mutable namespace of training operations."""


@contextmanager
def _criterion_random_calls(ctx, layer: int, batch_size: int):
    if ctx.rng.mode == "off":
        yield
        return
    original_rand = torch.rand
    call = 0

    def named_rand(*shape, **kwargs):
        nonlocal call
        if call < batch_size:
            name = f"match/L{layer}/img{call}"
        elif call == batch_size:
            name = f"loss_over/L{layer}"
        elif call == batch_size + 1:
            name = f"loss_rand/L{layer}"
        else:
            raise RuntimeError(f"Unexpected random call in criterion layer {layer}")
        call += 1
        device = kwargs.get("device", ctx.model.queries.weight.device)
        dtype = kwargs.get("dtype", torch.float32)
        return ctx.rng.sample(name, lambda: original_rand(*shape, **kwargs), device, dtype)

    torch.rand = named_rand
    try:
        yield
    finally:
        torch.rand = original_rand
        if call != batch_size + 2:
            raise AssertionError(f"Expected {batch_size + 2} criterion random calls; got {call}")


def default_ops() -> Ops:
    def rope(ctx, block_idx: int, H: int, W: int):
        return ctx.train_model.model.backbone.rope_embed(H=H, W=W)

    def build_mask(ctx, j: int, mask_logits: torch.Tensor, grid_size: tuple[int, int], n_tokens: int):
        q = ctx.train_model.model.num_queries
        storage = ctx.train_model.model.backbone.n_storage_tokens
        mask = torch.ones(mask_logits.shape[0], n_tokens, n_tokens, dtype=torch.bool, device=mask_logits.device)
        interpolated = F.interpolate(mask_logits, size=grid_size, mode="bilinear")
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        mask[:, :q, q + 1 + storage :] = interpolated > 0
        prob = ctx.train_model.model.attn_mask_probs[j]
        # Preserve the source's conditional RNG consumption exactly.
        if bool(prob < 1):
            random_queries = ctx.rng.rand(
                f"mask/j{j}", (mask.shape[0], q), device=mask.device
            ) > prob
            mask[:, :q, q + 1 + storage :][random_queries] = True
        return mask

    def attn(ctx, block, x_normed, rope, mask):
        # Same op sequence as the source ``_attn``: qkv Linear, attention core, proj.
        module = block.attn
        qkv = module.qkv(x_normed)
        out = ctx.ops.attn_core(ctx, qkv, rope, mask, module.num_heads)
        return module.proj_drop(module.proj(out))

    def attn_core(ctx, qkv, rope, mask, num_heads):
        """Packed ``qkv`` ``[B, N, 3*C]`` -> attention output ``[B, N, C]`` (heads concatenated)."""
        B, N, C3 = qkv.shape
        C = C3 // 3
        qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        if rope is not None:
            q, k = _apply_rope(q, k, rope)
        if mask is not None:
            mask = mask[:, None, ...].expand(-1, num_heads, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(B, N, C)

    def block(ctx, i, block, x, rope, mask):
        x = x + block.ls1(ctx.ops.attn(ctx, block, block.norm1(x), rope, mask))
        return x + block.ls2(block.mlp(block.norm2(x)))

    def predict(ctx, x, grid_size):
        return ctx.train_model.model._predict(ctx.train_model.model.backbone.norm(x), grid_size=grid_size)

    def loss_all_layers(ctx, mask_logits_list, class_logits_list, binary_masks):
        losses: dict[str, torch.Tensor] = {}
        num_blocks = len(ctx.train_model.model.backbone.blocks)
        criterion = ctx.train_model.criterion
        for layer, (mask_logits, class_logits) in enumerate(zip(mask_logits_list, class_logits_list)):
            block_idx = num_blocks - len(mask_logits_list) + 1 + layer
            with _criterion_random_calls(ctx, layer, mask_logits.shape[0]):
                block_losses = criterion(
                    masks_queries_logits=mask_logits,
                    class_queries_logits=class_logits,
                    targets=binary_masks,
                )
            suffix = f"_block{block_idx}" if block_idx < num_blocks else ""
            losses.update({f"{key}{suffix}": value for key, value in block_losses.items()})
        return criterion.loss_total(losses_all_layers=losses), losses

    def opt_step(ctx):
        from .fast_forward import _range
        with _range(ctx, "clip"):
            grad_norm = ctx.train_model.clip_gradients(ctx.fabric, ctx.optimizer)
        with _range(ctx, "opt"):
            ctx.optimizer.step()
            ctx.optimizer.zero_grad()
        with _range(ctx, "sched"):
            ctx.scheduler.step()
        return grad_norm

    return Ops(
        rope=rope, build_mask=build_mask, attn=attn, attn_core=attn_core, block=block,
        predict=predict, loss_all_layers=loss_all_layers, opt_step=opt_step,
    )


def _apply_rope(q: torch.Tensor, k: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]):
    """Exact copy of DINOv3 ``SelfAttention.apply_rope`` (fp32 rotation of the patch tokens)."""
    from lightly_train._models.dinov3.dinov3_src.layers.attention import rope_apply

    q_dtype, k_dtype = q.dtype, k.dtype
    sin, cos = rope
    q = q.to(dtype=sin.dtype)
    k = k.to(dtype=sin.dtype)
    prefix = q.shape[-2] - sin.shape[-2]
    assert prefix >= 0
    q = torch.cat((q[:, :, :prefix, :], rope_apply(q[:, :, prefix:, :], sin, cos)), dim=-2)
    k = torch.cat((k[:, :, :prefix, :], rope_apply(k[:, :, prefix:, :], sin, cos)), dim=-2)
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)
