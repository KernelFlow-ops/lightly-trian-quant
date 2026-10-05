"""QueryBitmask (CONTRACTS section 4) and the fused ``ops.build_mask`` override.

``bits`` is ``uint32 [B, 100, 32]``: bit ``p % 32`` of word ``p // 32`` set means query ``r`` may see patch ``p``
(key index ``n_prefix + p``).  Query rows never mask the query / cls / register keys, and rows >= 100 are never masked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .attn_ext import load_ext


@dataclass
class QueryBitmask:
    bits: torch.Tensor  # uint32 (or int32 view) [B, rows, 32]
    n_prefix: int       # token index of patch 0 (= num_queries + 1 + n_storage_tokens = 105)
    rows: int = 100     # number of masked query rows

    @property
    def batch(self) -> int:
        return self.bits.shape[0]

    def to_dense(self, n_tokens: int) -> torch.Tensor:
        """bool [B, N, N], True = allowed (reference semantics, used by tests and the fallback path)."""
        b = self.bits.shape[0]
        words = self.bits.to(torch.int64) & 0xFFFFFFFF
        shifts = torch.arange(32, device=self.bits.device, dtype=torch.int64)
        allowed = ((words[:, :, :, None] >> shifts) & 1).bool().reshape(b, self.rows, 1024)
        mask = torch.ones(b, n_tokens, n_tokens, dtype=torch.bool, device=self.bits.device)
        mask[:, : self.rows, self.n_prefix : self.n_prefix + 1024] = allowed
        return mask


def build_query_bitmask(mask_logits: torch.Tensor, n_prefix: int, rand: torch.Tensor | None = None, prob: torch.Tensor | None = None) -> QueryBitmask:
    """Exact source semantics: (2x2 average of the 128x128 logits > 0) with rows ``rand > prob`` fully enabled."""
    ext = load_ext()
    logits = mask_logits if mask_logits.is_contiguous() else mask_logits.contiguous()
    if rand is not None:
        assert prob is not None
        prob = prob.reshape(1)
        if prob.dtype != torch.float32:
            prob = prob.float()
        if rand.dtype != torch.float32 or not rand.is_contiguous():
            rand = rand.float().contiguous()
    bits = ext.build_bits(logits, rand, prob)
    return QueryBitmask(bits, n_prefix, mask_logits.shape[1])


def make_build_mask(fallback: Callable[..., Any], *, device_prob: bool = False, skip_zero: bool = True) -> Callable[..., Any]:
    """Build the ``ops.build_mask`` override.

    * host mode (default): identical RNG consumption to the source / round1: ``ctx.rng.rand("mask/j{j}", (B, Q))`` is drawn only when
      the host probability is < 1 (``ctx.host_probs`` when round1's nosync schedule is installed, else ``bool(prob < 1)`` like the
      source).  With ``skip_zero`` and a known host probability of exactly 0 the mask is dropped (returns ``None``) after drawing the
      random numbers, like round1's mask_skip (all rows are re-enabled unless a draw is exactly 0.0).
    * device_prob mode (round16, phase-agnostic graphs): always draws the random numbers and always builds the bitmask; the
      probability is read on the device (``rand > prob``), no host branch and no sync.
    """

    def build_mask(ctx, j: int, mask_logits: torch.Tensor, grid_size: tuple[int, int], n_tokens: int):
        model = ctx.model
        q = model.num_queries
        n_prefix = q + 1 + model.backbone.n_storage_tokens
        if (
            tuple(grid_size) != (32, 32)
            or mask_logits.dtype not in (torch.float32, torch.bfloat16)
            or tuple(mask_logits.shape[1:]) != (q, 128, 128)
            or q != 100
            or n_tokens != n_prefix + 1024
        ):
            return fallback(ctx, j, mask_logits, grid_size, n_tokens)
        prob = model.attn_mask_probs[j]
        b = mask_logits.shape[0]
        rand = None
        if device_prob:
            rand = ctx.rng.rand(f"mask/j{j}", (b, q), device=mask_logits.device)
        else:
            host_known = bool(ctx.meta.get("round1_annealing_host"))
            host_prob = ctx.host_probs[j] if host_known else None
            draw = host_prob < 1.0 if host_known else bool(prob < 1)
            if draw:
                rand = ctx.rng.rand(f"mask/j{j}", (b, q), device=mask_logits.device)
            if skip_zero and host_known and host_prob == 0.0:
                return None
        return build_query_bitmask(mask_logits, n_prefix, rand, prob if rand is not None else None)

    return build_mask
