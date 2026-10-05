"""fp32 / fp64 math references for the round12 attention gates (no fused code, no SDPA)."""
from __future__ import annotations

import torch


def rope_apply_fp(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """DINOv3 rope_apply: x * cos + rotate_half(x) * sin, x [..., N, 64] (same dtype as sin/cos or promoted)."""
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


def apply_rope_packed(qkv: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, prefix: int, dtype=torch.float32, round_bf16: bool = False):
    """qkv [B,N,3,H,64] -> q, k, v as [B,H,N,64] in ``dtype`` with rope on the last N-prefix tokens of q and k."""
    q, k, v = (t.transpose(1, 2).to(dtype) for t in qkv.unbind(2))
    if sin is not None and prefix >= 0:
        s, c = sin.to(dtype), cos.to(dtype)

        def rot(x):
            y = torch.cat((x[:, :, :prefix], rope_apply_fp(x[:, :, prefix:], s, c)), dim=2)
            return y.to(torch.bfloat16).to(dtype) if round_bf16 else y

        q, k = rot(q), rot(k)
    return q, k, v


def dense_mask_from_bits(bits: torch.Tensor, n: int, key0: int, rows: int = 100) -> torch.Tensor:
    """bool [B,N,N], True = allowed.  bits u32/i32 [B,rows,32]; row r (< rows) sees key key0 + p iff bit p is set."""
    b = bits.shape[0]
    words = bits.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(32, device=bits.device, dtype=torch.int64)
    allowed = ((words[:, :, :, None] >> shifts) & 1).bool().reshape(b, rows, 1024)
    mask = torch.ones(b, n, n, dtype=torch.bool, device=bits.device)
    mask[:, :rows, key0 : key0 + 1024] = allowed
    return mask


def ref_attention(qkv, rope, mask, prefix: int, dtype=torch.float64, round_bf16: bool = False):
    """Returns (o [B,N,H*64], lse natural-log [B,H,N]) in ``dtype``; autograd-capable w.r.t. qkv (if it requires grad)."""
    sin, cos = rope if rope is not None else (None, None)
    q, k, v = apply_rope_packed(qkv, sin, cos, prefix, dtype, round_bf16)
    s = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    if mask is not None:
        s = s.masked_fill(~mask[:, None], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None])
    o = torch.matmul(p, v)
    b, h, n, d = o.shape
    return o.transpose(1, 2).reshape(b, n, h * d), lse
