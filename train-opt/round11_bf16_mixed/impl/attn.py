"""``attn_core`` implementations for the torch-BF16 stack (round11).

Everything here is plain PyTorch SDPA plus the round4 fused RoPE kernel.  The goal is to
make the backend choice a measured, switchable property of ``ops.attn_core``:

* ``plain`` backend  - blocks without an attention mask (plain blocks, and joint blocks whose
  mask probability is 0 under ``mask_skip``): ``auto`` | ``flash`` | ``cudnn`` | ``eff``.
* ``joint`` backend  - blocks with a mask: ``auto`` | ``cudnn`` | ``eff``.  Flash cannot take
  a mask.
* ``bias``           - convert the bool mask to an additive bf16 bias (aligned to 8 elements)
  before SDPA instead of letting torch convert/pad it on every call.
* ``qsplit``         - only the ``nq`` query rows of a joint block are masked (rows >= nq see
  every key).  Rows are independent in attention, so the unmasked ``N-nq`` rows go to a
  mask-free backend (flash) and only the ``nq`` masked rows use the masked backend.
* ``rope``           - ``torch`` (copy of the source ``apply_rope``) or ``triton`` (round4
  fused packed-QKV RoPE, one launch forward and one backward).

Unknown mask types (for example round12's ``QueryBitmask``) are delegated to the previously
installed ``attn_core``.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS: dict[str, SDPBackend | None] = {
    "auto": None,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "eff": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


@dataclass(frozen=True)
class AttnCfg:
    rope: str = "torch"      # "torch" | "triton"
    plain: str = "auto"      # backend for mask-free attention
    joint: str = "auto"      # backend for masked attention
    bias: bool = False       # bool mask -> aligned additive bias
    qsplit: bool = False     # split the nq masked query rows from the unmasked rows
    nq: int = 100            # number of query tokens (rows 0..nq-1 are the only masked rows)

    def __post_init__(self) -> None:
        if self.rope not in ("torch", "triton"):
            raise ValueError(f"unknown rope mode {self.rope!r}")
        for name in (self.plain, self.joint):
            if name not in BACKENDS:
                raise ValueError(f"unknown SDPA backend {name!r}; known {tuple(BACKENDS)}")
        if self.plain == "math" or self.joint == "math":
            raise ValueError("math backend is reference-only")
        if self.joint == "flash":
            raise ValueError("flash cannot take an attention mask; use qsplit for the unmasked rows")


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, backend: str) -> torch.Tensor:
    chosen = BACKENDS[backend]
    context = sdpa_kernel(backends=[chosen]) if chosen is not None else nullcontext()
    with context:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """bool mask [..., Lq, Lk] -> additive bias (0 / -inf) whose row stride is a multiple of 8.

    The buffer is padded on the last dimension and sliced, so mem-efficient attention sees an
    aligned stride and does not have to pad (copy) the bias itself on every call.
    """
    *lead, rows, cols = mask.shape
    padded = (cols + 7) // 8 * 8
    buffer = torch.zeros((*lead, rows, padded), dtype=dtype, device=mask.device)
    bias = buffer[..., :cols]
    bias.masked_fill_(~mask, float("-inf"))
    return bias


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Accept [B,N,N] (source layout) or [B,1|H,N,N] (round1 broadcast layout) bool masks."""
    if mask.dtype != torch.bool:
        raise TypeError(f"expected a bool mask, got {mask.dtype}")
    if mask.ndim == 3:
        return mask[:, None]
    if mask.ndim == 4:
        return mask
    raise ValueError(f"unsupported mask rank {mask.ndim}")


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, cfg: AttnCfg) -> torch.Tensor:
    """q, k, v: [B,H,N,D] views; returns [B,H,N,D]."""
    if mask is None:
        return _sdpa(q, k, v, None, cfg.plain)
    mask = normalize_mask(mask)
    if cfg.qsplit:
        nq = cfg.nq
        masked_rows = mask[:, :, :nq]
        if cfg.bias:
            masked_rows = to_bias(masked_rows, q.dtype)
        out_masked = _sdpa(q[:, :, :nq], k, v, masked_rows, cfg.joint)
        out_free = _sdpa(q[:, :, nq:], k, v, None, cfg.plain)
        # concatenate in [B,N,H,D] order so the following reshape to [B,N,C] is a view
        return torch.cat((out_masked.transpose(1, 2), out_free.transpose(1, 2)), dim=1).transpose(1, 2)
    if cfg.bias:
        mask = to_bias(mask, q.dtype)
    return _sdpa(q, k, v, mask, cfg.joint)


def make_attn_core(cfg: AttnCfg, fallback: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Build ``ops.attn_core(ctx, qkv, rope, mask, num_heads)`` for ``cfg``."""
    rope_fn = None
    if cfg.rope == "triton":
        from round4_attention.impl.rope import rope_qkv

        rope_fn = rope_qkv

    def attn_core(ctx: Any, qkv: torch.Tensor, rope: Any, mask: Any, num_heads: int) -> torch.Tensor:
        if mask is not None and not isinstance(mask, torch.Tensor):
            return fallback(ctx, qkv, rope, mask, num_heads)
        batch, tokens, c3 = qkv.shape
        channels = c3 // 3
        head_dim = channels // num_heads
        qkv5 = qkv.reshape(batch, tokens, 3, num_heads, head_dim)
        if rope is not None and rope_fn is not None and head_dim == 64 and qkv5.is_cuda and qkv5.is_contiguous():
            qkv5 = rope_fn(qkv5, rope)
            q, k, v = (t.transpose(1, 2) for t in qkv5.unbind(2))
        else:
            q, k, v = (t.transpose(1, 2) for t in torch.unbind(qkv5, 2))
            if rope is not None:
                from common.ops import _apply_rope

                q, k = _apply_rope(q, k, rope)
        out = attention(q, k, v, mask, cfg)
        return out.transpose(1, 2).reshape(batch, tokens, channels)

    return attn_core
