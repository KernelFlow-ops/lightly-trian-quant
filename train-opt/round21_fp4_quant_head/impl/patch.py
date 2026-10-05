"""round21_fp4_quant_head (R21b): REAL-kernel NVFP4 / MXFP8 head ConvT GEMMs, stacked AFTER the FINAL stack (BEST_STACK_V2).

    BEST_STACK_V2 + ("round21_fp4_quant_head",)       variant = round20b policy string (``round20_fp4_probe/impl/fp4_policy.py``)

    H3  "head.f=mx8,head.dw=nv4+sr"   (default; the CONTRACTS 6b 13.2 strategy: mx8 forward, nv4 + SR dgrad / wgrad)
    H1  "head.dw=nv4+sr"              (forward stays bf16)
    H2  "head.dw=mx8"                 (r14's MXFP8 backward path)
    "head.f=mx8,head.dw=nv4+rn"       RN (no SR): bit-aligned with the round20b simulated head when combined with ``amax=exact``

Replaces ``ops.predict`` (r14's fused head structure with the ConvT GEMMs of ``impl/head_fp4.py``); the previous ``ops.predict`` is kept as the fallback for ineligible shapes
and when autograd is disabled.  Wraps ``ctx.ops.opt_step``: the original runs first, then the own ``FP4State`` is rolled (exactly once per step in eager and in the CUDA graph).
Parameters / state_dict are untouched.  Extra patch kwargs: ``dy_headroom`` (default 8.0, CONTRACTS 13.5), ``x_headroom`` / ``w_headroom`` (2.0), ``impl`` (nv4 GEMM flavour).

    ctx.meta["round21_fp4_quant_head"].state.stats()     # host-side saturation statistics of the delayed amax (NOT inside a graph)
"""
from __future__ import annotations

import torch

NAME = "round21_fp4_quant_head"
STATES: list = []   # (policy text, FP4State) of every context that applied this patch (tests read the saturation stats after the context was freed)
REQUIRES: tuple[str, ...] = ("round14_head_sm120", "round15_optimizer_flat")
CONFLICTS: tuple[str, ...] = ()


def apply(ctx, variant: str = "head.f=mx8,head.dw=nv4+sr", **kw) -> dict:
    from round21_fp4_quant_head.impl import head_fp4

    if ctx.cfg.precision != "bf16-mixed" or not ctx.meta.get("enable_autocast"):
        raise ValueError("round21 builds on the bf16 fused head: apply the FINAL stack (round11 autocast, r14, r15, r17) first")
    handle = head_fp4.Handle(ctx, variant, **kw)
    fallback = ctx.ops.predict
    original_opt = ctx.ops.opt_step

    def predict(ctx_, x, grid_size):
        if torch.is_grad_enabled() and head_fp4.eligible(ctx_.train_model.model, x, grid_size):
            return head_fp4.predict(ctx_, handle, x, grid_size)
        return fallback(ctx_, x, grid_size)

    def opt_step(*a, **k):
        out = original_opt(*a, **k)
        handle.end_step()
        return out

    ctx.ops.predict = predict
    ctx.ops.opt_step = opt_step
    ctx.meta[NAME] = handle
    STATES.append((handle.policy.describe(), handle.state))
    ctx.meta[NAME + "_fallback"] = (fallback, original_opt)
    return {"ops": ["predict", "opt_step"], "desc": f"round21 fp4 head: policy={handle.policy.describe()!r} dy_headroom={handle.dy_headroom}"}
