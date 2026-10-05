"""Regional ``torch.compile`` (one compiled region per transformer block) -- reference only.

The compiled region is the default (unfused) block: LN, qkv Linear, RoPE, SDPA, proj, LayerScale,
MLP.  Inductor sees fwd+bwd of the whole block under autocast.  This is recorded in the candidate
matrix as the "what would the compiler do on its own" reference; it is not part of T-BF16.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _block_forward(blk, x, sin, cos, mask):
    from common.ops import _apply_rope

    h = blk.norm1(x)
    qkv = blk.attn.qkv(h)
    batch, tokens, c3 = qkv.shape
    heads = blk.attn.num_heads
    channels = c3 // 3
    q, k, v = (t.transpose(1, 2) for t in qkv.reshape(batch, tokens, 3, heads, channels // heads).unbind(2))
    if sin is not None:
        q, k = _apply_rope(q, k, (sin, cos))
    if mask is not None and mask.ndim == 3:
        mask = mask[:, None]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = out.transpose(1, 2).reshape(batch, tokens, channels)
    x = x + blk.ls1(blk.attn.proj_drop(blk.attn.proj(out)))
    return x + blk.ls2(blk.mlp(blk.norm2(x)))


def install_block_compile(ctx, mode: str) -> None:
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)
    compiled = torch.compile(_block_forward, mode=None if mode == "default" else mode, fullgraph=False, dynamic=False)

    def block(c, i, blk, x, rope, mask):
        sin, cos = (None, None) if rope is None else rope
        return compiled(blk, x, sin, cos, mask)

    ctx.ops.block = block
    ctx.meta["round11_compile_mode"] = mode
