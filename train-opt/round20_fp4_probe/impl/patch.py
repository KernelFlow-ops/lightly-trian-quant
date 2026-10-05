"""round20_fp4_probe (R20b): "half-real" FP4 training block + head with a runtime-switchable policy, stacked AFTER the FINAL stack (BEST_STACK_V2).

    BEST_STACK_V2 + ("round20_fp4_probe",)      variant = policy string (``impl/fp4_policy.py``); "" = FINAL's behaviour op for op

Replaces ``ops.block`` (r17's structure; the previous block is kept as the fallback for ineligible blocks) and ``ops.predict`` (r14's fused head; the previous
predict is the fallback for ineligible shapes).  nv4 / mx4 operands come from the torch simulation quantizer (``fp4_sim``), the GEMMs are the real QuACK
FP4 kernels.  Parameters / state_dict are untouched.  The handle ``ctx.meta['round20_fp4_probe']`` switches the policy at run time::

    ctx.meta["round20_fp4_probe"].set_policy("head.dw=nv4+sr,all.w=nv4+sr+rht,prefix")
"""
from __future__ import annotations

NAME = "round20_fp4_probe"
REQUIRES: tuple[str, ...] = ("round13_gemm_fusion_sm120", "round14_head_sm120", "round17_mxfp8_sm120")
CONFLICTS: tuple[str, ...] = ()


def apply(ctx, variant: str = "", **kw) -> dict:
    from round20_fp4_probe.impl import fp4_head, fp4_block

    if ctx.cfg.precision != "bf16-mixed" or not ctx.meta.get("enable_autocast"):
        raise ValueError("round20 builds on the bf16 fused block: apply the FINAL stack (round11 autocast, r13, r14, r17, r15) first")
    handle = fp4_block.Handle(ctx, variant)
    ctx.ops.block = fp4_block.make_block(ctx, ctx.ops.block, handle)
    fallback = ctx.ops.predict

    def predict(ctx_, x, grid_size):
        if fp4_head.eligible(ctx_.train_model.model, x, grid_size):
            return fp4_head.predict(ctx_, handle, x, grid_size)
        return fallback(ctx_, x, grid_size)

    ctx.ops.predict = predict
    ctx.meta[NAME] = handle
    return {"ops": ["block", "predict"], "desc": f"round20 fp4 sim: policy={handle.policy.describe()!r}"}
