"""autograd.Function and ``ops.attn_core`` override around the round12 fused attention kernels.

Forward : packed qkv [B,N,3C] bf16 (-> [B,N,3,H,64]) + RoPE tables + optional QueryBitmask -> O [B,N,C] bf16, lse (log2, fp32 [B,H,N]).
Backward: (qkv, O, lse, dO) -> packed dqkv [B,N,3C] with the inverse RoPE already applied.
Graph capturable: everything runs on the current stream, workspaces come from the torch caching allocator (static addresses
inside a captured graph), no host synchronisation.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from .attn_ext import load_ext
from .bitmask import QueryBitmask


class FusedAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, sin, cos, bits, heads: int, prefix: int, mask_rows: int, mask_key0: int):
        ext = load_ext()
        save_rot = bool(ctx.needs_input_grad[0]) and sin is not None
        o, lse2, qk_rot = ext.attn_fwd(qkv, heads, sin, cos, bits, prefix, mask_rows, mask_key0, save_rot)
        saved = [qkv, o, lse2]
        ctx.has_qk_rot = save_rot
        if save_rot:
            saved.append(qk_rot)
        ctx.has_rope = sin is not None
        ctx.has_bits = bits is not None
        if sin is not None:
            saved += [sin, cos]
        if bits is not None:
            saved += [bits]
        ctx.save_for_backward(*saved)
        ctx.cfg = (heads, prefix, mask_rows, mask_key0)
        return o

    @staticmethod
    def backward(ctx, do):
        ext = load_ext()
        heads, prefix, mask_rows, mask_key0 = ctx.cfg
        qkv, o, lse2, *rest = ctx.saved_tensors
        qk_rot = None
        if ctx.has_qk_rot:
            qk_rot, *rest = rest
        sin = cos = bits = None
        if ctx.has_rope:
            sin, cos, *rest = rest
        if ctx.has_bits:
            (bits,) = rest
        if do.dtype != torch.bfloat16:
            do = do.to(torch.bfloat16)
        if not do.is_contiguous():
            do = do.contiguous()
        dqkv = ext.attn_bwd(qkv, qk_rot, o, lse2, do, heads, sin, cos, bits, prefix, mask_rows, mask_key0)
        return dqkv.view(qkv.shape), None, None, None, None, None, None, None


def supported(qkv: torch.Tensor, rope: Any, mask: Any, num_heads: int) -> bool:
    if not (qkv.is_cuda and qkv.dtype == torch.bfloat16 and qkv.dim() == 3):
        return False
    c3 = qkv.shape[2]
    if c3 != 3 * num_heads * 64:
        return False
    n = qkv.shape[1]
    if rope is not None:
        sin, cos = rope
        if sin.dtype != torch.float32 or tuple(sin.shape) != (1024, 64) or tuple(cos.shape) != (1024, 64) or n < 1024:
            return False
    if mask is not None:
        if not isinstance(mask, QueryBitmask) or mask.rows != 100 or n != mask.n_prefix + 1024 or mask.bits.shape[0] != qkv.shape[0]:
            return False
    return True


def dense_attn_core(ctx, qkv, rope, mask, num_heads):
    """Reference path for everything the fused kernels do not take (fp32, odd shapes, dense bool masks of either layout)."""
    import torch.nn.functional as F
    from common.ops import _apply_rope

    b, n, c3 = qkv.shape
    c = c3 // 3
    q, k, v = (t.transpose(1, 2) for t in qkv.reshape(b, n, 3, num_heads, c // num_heads).unbind(2))
    if rope is not None:
        q, k = _apply_rope(q, k, rope)
    if isinstance(mask, QueryBitmask):
        mask = mask.to_dense(n)
    if mask is not None:
        mask = mask[:, None] if mask.ndim == 3 else mask  # [B,N,N] (source) or [B,1,N,N] (round1 broadcast) -> [B,1,N,N]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(b, n, c)


def make_attn_core(fallback: Callable[..., torch.Tensor] | None = None) -> Callable[..., torch.Tensor]:
    """``ops.attn_core(ctx, qkv, rope, mask, num_heads)``; everything unsupported (fp32, dense bool masks, odd shapes) goes to ``fallback``
    (default: ``dense_attn_core``; pass an earlier installed attn_core that understands the same mask layouts to chain)."""
    fallback = fallback or dense_attn_core

    def attn_core(ctx, qkv, rope, mask, num_heads):
        if not supported(qkv, rope, mask, num_heads):
            return fallback(ctx, qkv, rope, mask, num_heads)
        if not qkv.is_contiguous():
            qkv = qkv.contiguous()
        n = qkv.shape[1]
        sin = cos = None
        prefix = -1
        if rope is not None:
            sin, cos = rope
            if not sin.is_contiguous():
                sin = sin.contiguous()
            if not cos.is_contiguous():
                cos = cos.contiguous()
            prefix = n - sin.shape[0]
        bits = mask.bits if mask is not None else None
        rows = mask.rows if mask is not None else 0
        key0 = mask.n_prefix if mask is not None else 0
        return FusedAttn.apply(qkv, sin, cos, bits, num_heads, prefix, rows, key0)

    return attn_core
