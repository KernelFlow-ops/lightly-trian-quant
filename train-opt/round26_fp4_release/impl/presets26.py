"""Release presets of phase 3 (round26_fp4_release): the published defaults of the final stack.  NO new kernels, no new measurements of its own beyond the ones listed in NOTES.md.

    V3        the release default of the model (``--stack V3``):
                ViT-L : head ``head.f=mx8,head.dw=nv4+sr``  + backbone ``all.d=nv4+sr+all.w=nv4+sr+nofirst+dyhr=32``      (= round25's P3_hr32)
                ViT-B : head ``head.dw=mx8``                + backbone ``all.d=nv4+sr+all.w=nv4+sr+dyhr=32``              (round25's P3_H2 with backbone dY headroom 32)
    V3_speed  the speed tier: head ``head.f=mx8,head.dw=nv4+sr`` + the same backbone  (ViT-L: identical to V3; ViT-B: round25's P3_hr32)

Why these (CONTRACTS Amendment 6d): backbone dY headroom 32 removes the dY saturation of the delayed amax (ViT-L S gate: 232 / 36800 calls at headroom 8 -> 0 at 32) with identical gradient
cosines and no speed cost; on ViT-B the NVFP4 head (H3) is what drives the worst-batch cosine below the relaxed 0.97 line, the MXFP8-backward head (H2) keeps it above.
Everything else (R0, FINAL, P3, P3H, P3B, P3_hr*, P3_H2, ...) is inherited from round25 / round23 / round19.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round25_fp4_harden.impl import presets25 as p25  # noqa: E402
from round25_fp4_harden.impl.presets25 import H2, H3, R21, R22, fp4_handles, opt_step_chain  # noqa: E402,F401
from common.config import P3_PATCH_KWARGS_VITB, P3_PATCH_KWARGS_VITL, P3_STACK  # noqa: E402

DYHR = 32


def release_kwargs(model: str, *, speed: bool = False) -> dict:
    """Patch kwargs of the release stack of ``model`` (``speed=True``: the NVFP4-head speed tier)."""
    vitl = model == "vitl16"
    p3_kw = P3_PATCH_KWARGS_VITL if vitl else P3_PATCH_KWARGS_VITB
    backbone = p3_kw[R22]["variant"] + f"+dyhr={DYHR}"
    head = H3 if (vitl or speed) else H2
    return {**p3_kw, R21: {"variant": head}, R22: {"variant": backbone}}


def model_presets(model: str) -> dict:
    out = dict(p25.model_presets(model))
    out["V3"] = (P3_STACK, release_kwargs(model))
    out["V3_speed"] = (P3_STACK, release_kwargs(model, speed=True))
    return out


def register(round_dir: Path, model: str) -> None:
    """round25's registration (round23 / round19 presets, outputs redirected to ``round_dir``) + the release presets."""
    from round16_full_graph import run as r16

    p25.register(round_dir, model)
    r16.PRESET_CFG.update(model_presets(model))


Q2_ACTIVE = False


def maybe_q2(force=None) -> bool:
    return False
