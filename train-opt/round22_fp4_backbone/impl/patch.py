"""round22_fp4_backbone (R22): REAL-kernel FP4 (NVFP4) backbone dgrad / wgrad, stacked AFTER the FINAL stack (BEST_STACK_V2).

    BEST_STACK_V2 + ("round22_fp4_backbone",)       variant = policy string (round20 grammar, ``+``-joined; see ``impl/fp4_block22.py``)

Replaces ``ops.block`` (the previous block stays as the fallback for ineligible blocks) with r17's fused block whose dgrad / wgrad GEMMs of the policy's
Linears run on NVFP4 operands produced by round21's fused quantizers and consumed by QuACK's nv4 GEMMs.  The forward is unchanged (qkv bf16,
proj / fc1 / fc2 mx8).  The head (``ops.predict``) is not touched (FINAL's r14 bf16 head).  Parameters / state_dict are untouched.

Policies (CONTRACTS 6b 13.2):
    ViT-L (B4):  all.d=nv4+sr+all.w=nv4+sr+nofirst        ViT-B (B3):  all.d=nv4+sr+all.w=nv4+sr
    optional round22 tokens: dyhr=8 xhr=2 whr=2 (amax headroom of dY / X / weights), amax=exact|delayed (default delayed)

Per-patch ``FP4State`` (not shared with other patches): ``ctx.ops.opt_step`` is wrapped so that every step (eager or graph replay) ends with ``state.roll()``
(prev <- next amax, seed += 1).  ``ctx.meta['round22_fp4_backbone']`` is the ``Handle`` (``set_policy`` at run time, ``stats()``).
"""
from __future__ import annotations

NAME = "round22_fp4_backbone"
REQUIRES: tuple[str, ...] = ("round13_gemm_fusion_sm120", "round17_mxfp8_sm120")
CONFLICTS: tuple[str, ...] = ()


class RollingOptStep:
    """``ctx.ops.opt_step`` wrapper: the wrapped step, then ``state.roll()`` -- exactly once per training step, eager or CUDA graph (captured together with the optimizer)."""

    def __init__(self, inner, handle):
        self.inner, self.handle = inner, handle

    def __call__(self, c):
        out = self.inner(c)
        self.handle.state.roll()
        return out


def apply(ctx, variant: str = "", **kw) -> dict:
    from round22_fp4_backbone.impl import fp4_block22

    if ctx.cfg.precision != "bf16-mixed" or not ctx.meta.get("enable_autocast"):
        raise ValueError("round22 builds on the bf16 fused block: apply the FINAL stack (round11 autocast, r13, r17, r15) first")
    handle = fp4_block22.Handle(ctx, variant)
    ctx.ops.block = fp4_block22.make_block(ctx, ctx.ops.block, handle)
    ctx.ops.opt_step = RollingOptStep(ctx.ops.opt_step, handle)
    ctx.meta[NAME] = handle
    return {"ops": ["block", "opt_step"], "desc": f"round22 real FP4 backbone: policy={handle.describe()!r}"}
