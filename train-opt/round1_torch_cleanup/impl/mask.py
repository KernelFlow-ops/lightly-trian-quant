"""Host-branch mask construction and SDPA's broadcast-mask call path."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_mask(ctx, j: int, mask_logits: torch.Tensor, grid_size: tuple[int, int], n_tokens: int, *, skip_zero: bool, broadcast: bool) -> torch.Tensor | None:
    prob_host = ctx.host_probs[j]
    q = ctx.model.num_queries
    if skip_zero and prob_host == 0.0:
        # The reference still draws (B,Q) at prob=0. Preserve all later RNG.
        ctx.rng.rand(f"mask/j{j}", (mask_logits.shape[0], q), device=mask_logits.device)
        return None

    shape = (mask_logits.shape[0], 1, n_tokens, n_tokens) if broadcast else (mask_logits.shape[0], n_tokens, n_tokens)
    mask = torch.ones(shape, dtype=torch.bool, device=mask_logits.device)
    interpolated = F.interpolate(mask_logits, size=grid_size, mode="bilinear")
    interpolated = interpolated.view(interpolated.shape[0], interpolated.shape[1], -1)
    first_patch = q + 1 + ctx.model.backbone.n_storage_tokens
    window = mask[:, 0, :q, first_patch:] if broadcast else mask[:, :q, first_patch:]
    window[:] = interpolated > 0
    if prob_host < 1.0:
        random_queries = ctx.rng.rand(f"mask/j{j}", (mask.shape[0], q), device=mask.device) > ctx.model.attn_mask_probs[j]
        window[random_queries] = True
    return mask


def attention_with_broadcast(ctx, block, x_normed: torch.Tensor, rope, mask: torch.Tensor | None) -> torch.Tensor:
    """Match the source _attn, letting an (B,1,N,N) mask broadcast over heads."""
    module = block.attn
    qkv = module.qkv(x_normed)
    batch, tokens, _ = qkv.shape
    channels = module.qkv.in_features
    qkv = qkv.reshape(batch, tokens, 3, module.num_heads, channels // module.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
    if rope is not None:
        q, k = module.apply_rope(q, k, rope)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    y = y.transpose(1, 2).reshape(batch, tokens, channels)
    return module.proj_drop(module.proj(y))
