"""Stack presets of round23 (the combined phase-3 stack) on top of round19's preset mechanism (which sits on round16's run.py parser).

    P3          FINAL + round21 FP4 head + round22 FP4 backbone     (common.config.P3_STACK; ViT-L / ViT-B policies differ: ViT-L has ``nofirst``)
    P3H         FINAL + round21 FP4 head only                       (common.config.P3H_STACK; the strict tier of ViT-B)
    P3B         FINAL + round22 FP4 backbone only                   (common.config.P3B_STACK; ViT-L B4 / ViT-B B3)
    P3_RN       P3 with RN instead of SR in the head AND the backbone (control only)
    P3_q8       P3 + ``qkv.f=mx8`` in the backbone policy (optional 5a; recorded only, not published)
    P3_strict   the strict tier of the model: ViT-B = P3H; ViT-L = P3 (no stricter tier is registered for ViT-L)
plus everything of round19 (R0, TB1 / P1, P2, ..., FINAL).

The presets depend on the model (``register(round_dir, model)``): ViT-L uses ``P3_PATCH_KWARGS_VITL`` / ``P3B_PATCH_KWARGS_VITL`` (policy B4 = ``+nofirst``), ViT-B the ``_VITB`` ones.

``--q2`` (``maybe_q2``): alias ``round21_fp4_quant_head.impl.fp4q`` -> ``round24_fp4_quant_opt.impl.fp4q2`` in ``sys.modules`` BEFORE any round21 / round22 patch module is imported
(both patches reach the quantizer only through that module name), so that the R24 quantizer replaces R21a's in the head AND the backbone.  Off by default.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import (  # noqa: E402
    P3_PATCH_KWARGS_VITB, P3_PATCH_KWARGS_VITL, P3_STACK, P3B_PATCH_KWARGS_VITB, P3B_PATCH_KWARGS_VITL, P3B_STACK, P3H_PATCH_KWARGS, P3H_STACK,
)

R21 = "round21_fp4_quant_head"
R22 = "round22_fp4_backbone"
Q2_ACTIVE = False


def maybe_q2(force: bool | None = None) -> bool:
    """Install the fp4q -> fp4q2 alias when ``--q2`` is on the command line (or R23_Q2=1, or force=True).  Must run before round21 / round22 are imported."""
    global Q2_ACTIVE
    want = force if force is not None else ("--q2" in sys.argv or os.environ.get("R23_Q2") == "1")
    if not want or Q2_ACTIVE:
        return Q2_ACTIVE
    if f"{R21}.impl.patch" in sys.modules or f"{R21}.impl.head_fp4" in sys.modules or f"{R22}.impl.fp4_block22" in sys.modules:
        raise RuntimeError("--q2 must be installed before the round21 / round22 modules are imported")
    import importlib

    q2 = importlib.import_module("round24_fp4_quant_opt.impl.fp4q2")
    pkg = importlib.import_module(f"{R21}.impl")
    sys.modules[f"{R21}.impl.fp4q"] = q2
    pkg.fp4q = q2
    Q2_ACTIVE = True
    print("[round23] --q2: round21_fp4_quant_head.impl.fp4q -> round24_fp4_quant_opt.impl.fp4q2", flush=True)
    return True


def model_presets(model: str) -> dict:
    vitl = model == "vitl16"
    p3_kw = P3_PATCH_KWARGS_VITL if vitl else P3_PATCH_KWARGS_VITB
    p3b_kw = P3B_PATCH_KWARGS_VITL if vitl else P3B_PATCH_KWARGS_VITB
    rn_head = "head.f=mx8,head.dw=nv4+rn"
    rn_bb = p3_kw[R22]["variant"].replace("+sr", "+rn")
    p3_rn_kw = {**p3_kw, R21: {"variant": rn_head}, R22: {"variant": rn_bb}}
    p3_q8_kw = {**p3_kw, R22: {"variant": p3_kw[R22]["variant"] + "+qkv.f=mx8"}}      # optional 5a: qkv forward MXFP8 (r17's path, no new kernel); recorded only, not published
    return {
        "P3": (P3_STACK, p3_kw),
        "P3H": (P3H_STACK, P3H_PATCH_KWARGS),
        "P3B": (P3B_STACK, p3b_kw),
        "P3_RN": (P3_STACK, p3_rn_kw),
        "P3_q8": (P3_STACK, p3_q8_kw),
        "P3_strict": (P3H_STACK, P3H_PATCH_KWARGS) if not vitl else (P3_STACK, p3_kw),
    }


def register(round_dir: Path, model: str) -> None:
    """r19 presets (+ redirect of round16's outputs to ``round_dir``) and the model-dependent phase-3 presets."""
    from round16_full_graph import run as r16
    from round19_final.impl import presets as p19

    p19.register(round_dir)
    r16.SHORT["r21"] = R21
    r16.SHORT["r22"] = R22
    r16.PRESET_CFG.update(model_presets(model))
    r16.ROUND = round_dir
    r16.TMP = Path("/root/autodl-tmp/train-opt-tmp") / round_dir.name


def fp4_handles(ctx) -> dict:
    """{'head': round21 Handle | None, 'backbone': round22 Handle | None}."""
    return {"head": ctx.meta.get(R21), "backbone": ctx.meta.get(R22)}


def opt_step_chain(ctx) -> dict:
    """Checks of the ``ctx.ops.opt_step`` wrapper chain (P3_STACK order: FINAL's step <- round21 closure <- round22 ``RollingOptStep``, outermost = last applied patch)."""
    outer = ctx.ops.opt_step
    h21, h22 = ctx.meta.get(R21), ctx.meta.get(R22)
    inner = getattr(outer, "inner", None)
    mid = inner if inner is not None else outer
    fb = ctx.meta.get(R21 + "_fallback")
    return {
        "outer": f"{type(outer).__name__}",
        "outer_is_r22_rolling_with_r22_handle": bool(inner is not None and getattr(outer, "handle", None) is h22 and h22 is not None),
        "middle_is_r21_closure": bool(getattr(mid, "__module__", "") == f"{R21}.impl.patch" and getattr(mid, "__qualname__", "").endswith("opt_step") and h21 is not None),
        "r21_original_opt_step": (type(fb[1]).__name__ + ":" + getattr(fb[1], "__qualname__", "")) if fb is not None else None,
        "r21_original_is_not_a_fp4_wrapper": bool(fb is not None and getattr(fb[1], "__module__", "") != f"{R21}.impl.patch" and not hasattr(fb[1], "handle")),
    }
