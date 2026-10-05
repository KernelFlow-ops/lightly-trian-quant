"""round17_mxfp8_sm120: MXFP8 (E4M3 + UE8M0, block-scaled mma, fp32 accumulation) Linear GEMMs inside round13's fused block.

Variant = ``+``-joined tokens (see ``fused_block_mx8.Policy``)::

    fc | mlp | fc1 | fc2 | qkv | proj | attn | all     which Linears run in MXFP8 (default all four)
    nofirst nolast nojoint | blocks=a-b                 which blocks (default every block, joint blocks included)

Examples: ``r17:fc`` (fc1+fc2 everywhere), ``r17:all+nofirst+nolast``, ``r17:fc+proj+nojoint``.

Stacks on ``TB1_STACK + ",r13:full"``: the patch REPLACES ``ops.block`` with a block that has the same fused structure as round13's
and keeps round13's block as the fallback for blocks the fused path does not cover.  Parameters and state_dict are untouched
(weights are quantized each step from the fp32 Parameters at block 0).
"""
from __future__ import annotations

NAME = "round17_mxfp8_sm120"
REQUIRES: tuple[str, ...] = ("round13_gemm_fusion_sm120",)
CONFLICTS: tuple[str, ...] = ()


def apply(ctx, variant: str = "all", **kw) -> dict:
    from round17_mxfp8_sm120.impl.fused_block_mx8 import Policy, make_block

    parts = [p.strip() for p in variant.replace(",", "+").split("+") if p.strip()]
    policy = Policy(parts)
    if ctx.cfg.precision != "bf16-mixed" or not ctx.meta.get("enable_autocast"):
        raise ValueError("round17 builds on the bf16 round13 block: apply round11 autocast and round13 first")
    ctx.ops.block = make_block(ctx, ctx.ops.block, policy)
    ctx.meta[NAME] = policy.describe()
    return {"ops": ["block"], "desc": f"round17 mxfp8: {policy.describe()}"}
