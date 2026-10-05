"""round14_head_sm120: fused EoMT head (``ops.predict``): GEMM-based ConvT + fused GELU/dwconv3x3/LN2D Triton kernels + QuACK einsum.

Variants (``+`` joined)::

    fused        bf16 GEMMs (QuACK fwd/dgrad, cuBLAS wgrad, QuACK einsum bwd), Triton elementwise kernels (default)
    torch_gemm   every GEMM through cuBLAS (ablation)
    quack_gemm   every GEMM through QuACK (ablation)
    mx8          ConvT fwd/dgrad/wgrad GEMMs in MXFP8 (round17 quantizer + QuACK block-scaled GEMM); einsum stays bf16
    mx8_bwd | mx8_fwd | mx8_dgrad | mx8_wgrad   only that subset of the ConvT GEMM kinds in MXFP8

Stacks after round11 (autocast); independent of round13/17 (it only replaces ``ops.predict``).  The previous ``ops.predict`` is
kept as the fallback for shapes the fused path does not cover (see ``head.eligible``).  Parameters / state_dict are untouched.
"""
from __future__ import annotations

NAME = "round14_head_sm120"
REQUIRES: tuple[str, ...] = ("round11_bf16_mixed",)
CONFLICTS: tuple[str, ...] = ()
VARIANTS = ("fused", "torch_gemm", "quack_gemm", "mx8", "mx8_bwd", "mx8_fwd", "mx8_dgrad", "mx8_wgrad")


def apply(ctx, variant: str = "fused", **kw) -> dict:
    from round14_head_sm120.impl import head

    parts = [p.strip() for p in variant.replace(",", "+").split("+") if p.strip()]
    unknown = [p for p in parts if p not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown round14 variants {unknown}; known {VARIANTS}")
    if ctx.cfg.precision != "bf16-mixed" or not ctx.meta.get("enable_autocast"):
        raise ValueError("round14 head expects bf16 autocast: apply round11_bf16_mixed (autocast) first")
    head.reset_policy()
    if "torch_gemm" in parts:
        head.POLICY.update({k: "torch" for k in head.POLICY if k != "mx8"})
    if "quack_gemm" in parts:
        head.POLICY.update({k: "quack" for k in head.POLICY if k != "mx8"})
    mx = [p for p in parts if p.startswith("mx8")]
    if mx:
        head.POLICY["mx8"] = True if mx == ["mx8"] else "+".join(p[4:] for p in mx)
    fallback = ctx.ops.predict

    def predict(ctx_, x, grid_size):
        if head.eligible(ctx_.train_model.model, x, grid_size):
            return head.predict(ctx_, x, grid_size)
        return fallback(ctx_, x, grid_size)

    ctx.ops.predict = predict
    ctx.meta[NAME] = tuple(parts) or ("fused",)
    return {"ops": ["predict"], "desc": f"round14 head: {variant}"}
