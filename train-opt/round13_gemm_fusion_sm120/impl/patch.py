"""round13_gemm_fusion_sm120: fused BF16 transformer block (``ops.block``).

Variants (join with ``+``)::

    full          QuACK sm_120 GEMMs with fused epilogues for fwd/dgrad, per-shape cuBLAS/QuACK choice for wgrad (fp32 out),
                  round13 CUDA kernels for LN / residual-gradient / LayerScale / bias-gradient (default)
    wgrad_quack   every wgrad through QuACK (default: per-shape choice between cuBLAS and QuACK, see gemm_ops.WGRAD_AUTO)
    torch_gemm    every GEMM through torch.mm (cuBLAS); epilogues become eager torch ops  (ablation only)

The patch must be applied after round11 ``autocast`` (it is a bf16 block); it keeps the previously installed
``ops.block`` as the fallback for blocks the fused path does not cover (see ``fused_block.eligible``) and
calls ``ctx.ops.attn_core`` for attention (round11/12 own it).
"""
from __future__ import annotations

NAME = "round13_gemm_fusion_sm120"
REQUIRES: tuple[str, ...] = ()
CONFLICTS: tuple[str, ...] = ()
VARIANTS = ("full", "wgrad_quack", "torch_gemm")


def apply(ctx, variant: str = "full", **kw) -> dict:
    from round13_gemm_fusion_sm120.impl import gemm_ops
    from round13_gemm_fusion_sm120.impl.fused_block import make_block

    parts = [p.strip() for p in variant.replace(",", "+").split("+") if p.strip()]
    unknown = [p for p in parts if p not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown round13 variants {unknown}; known {VARIANTS}")
    if ctx.cfg.precision != "bf16-mixed" or not ctx.meta.get("enable_autocast"):
        raise ValueError("round13 is a bf16 block: apply round11_bf16_mixed (autocast) first")
    policy = dict(gemm_ops.DEFAULT_POLICY)
    if "wgrad_quack" in parts:
        policy.update({k: "quack" for k in policy if k[0] == "wgrad"})
    if "torch_gemm" in parts:
        policy = {k: "torch" for k in policy}
    gemm_ops.POLICY.update(policy)
    ctx.ops.block = make_block(ctx, ctx.ops.block)
    ctx.meta[NAME] = tuple(parts) or ("full",)
    return {"ops": ["block"], "desc": f"round13: {variant}"}
