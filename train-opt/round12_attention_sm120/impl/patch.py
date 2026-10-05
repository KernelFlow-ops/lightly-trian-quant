"""round12_attention_sm120: fused sm_120 attention (``ops.attn_core`` + ``ops.build_mask``).

Variants (joined with ``+``):

  fused     packed-qkv attention forward/backward CUDA kernels with fused RoPE and the QueryBitmask (host-probability mask builder
            with source-identical RNG consumption; mask dropped when the host probability is exactly 0 like round1's mask_skip)
  devprob   mask builder reads the probability on the device and always builds the bitmask (phase-agnostic graphs, round16)
  maskdense keep the previously installed ``build_mask`` (dense bool masks go to the fallback attn_core); attention kernels only

``attn_core`` accepts ``None``, dense bool masks (delegated to the previously installed ``attn_core``) and ``QueryBitmask``.
Patch order: after round1/round3/round11 (it keeps their ops as fallbacks).  Parameters, modules and state_dict are untouched.
"""
from __future__ import annotations

from .bitmask import make_build_mask
from .fused import make_attn_core

NAME = "round12_attention_sm120"
REQUIRES: tuple[str, ...] = ()
CONFLICTS: tuple[str, ...] = ()
VARIANTS = ("fused", "devprob", "maskdense")


def apply(ctx, variant: str = "fused", **kw):
    parts = [p.strip() for p in variant.replace(",", "+").split("+") if p.strip()]
    unknown = [p for p in parts if p not in VARIANTS]
    if unknown:
        raise ValueError(f"{NAME}: unknown variants {unknown}; known {VARIANTS}")
    if ctx.meta.get(NAME) == tuple(parts):
        return {"ops": [], "desc": f"{NAME}: {variant} (already applied)"}
    from common.ops import default_ops

    changed = ["attn_core"]
    # an earlier attention override (round11's SDPA stack) understands round1's [B,1,N,N] masks; the default op does not
    ctx.ops.attn_core = make_attn_core(fallback=ctx.ops.attn_core if ctx.meta.get("round11_attn_cfg") is not None else None)
    # route the attention block through attn_core (round1's mask_skip wrapper would bypass it)
    ctx.ops.attn = default_ops().attn
    changed.append("attn")
    if "maskdense" not in parts:
        ctx.ops.build_mask = make_build_mask(fallback=ctx.ops.build_mask, device_prob="devprob" in parts)
        changed.append("build_mask")
    ctx.meta[NAME] = tuple(parts)
    # warm the JIT build and the kernels' dynamic-smem attributes outside any CUDA graph capture
    from .attn_ext import load_ext

    load_ext()
    return {"ops": changed, "desc": f"{NAME}: {variant}"}
