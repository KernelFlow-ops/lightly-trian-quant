"""Stack presets of round25 (hardening of the phase-3 final stack P3; NO new kernels).  Built on round23's presets (``impl/presets23.py``), which sit on round19 / round16's preset mechanism.

dY headroom / amax mode study (task 1; the head's dZ headroom stays 8: 0 saturations in every round23 S-gate scenario, so only the BACKBONE dY ``dyhr`` is swept):
    P3_hr16   P3 with backbone ``dyhr=16``          P3_hr32   ... ``dyhr=32``          P3_hr64   ... ``dyhr=64``
    P3_ex     P3 with ``amax=exact`` in BOTH patches (separate amax pass; x / dy / w all exact)
    (P3 = default: dY headroom 8, x / w headroom 2, delayed amax)
ViT-B conservative tier (task 2):
    P3_H2     P3 backbone + head ``head.dw=mx8`` (R21b's H2 = r14-style MXFP8 head backward; head forward stays bf16 as in R21b's ``P3H_H2``)
    P3_H2f    P3 backbone + head ``head.f=mx8,head.dw=mx8``  (same with the MXFP8 head forward of H3)
Optional ViT-L tier (task 3):  P3_q8 (round23's: P3 + ``qkv.f=mx8``).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from round23_fp4_final.impl import presets23 as p23  # noqa: E402
from round23_fp4_final.impl.presets23 import R21, R22, fp4_handles, opt_step_chain  # noqa: E402,F401
from common.config import P3_PATCH_KWARGS_VITB, P3_PATCH_KWARGS_VITL, P3_STACK  # noqa: E402

H3 = "head.f=mx8,head.dw=nv4+sr"
H2 = "head.dw=mx8"
H2F = "head.f=mx8,head.dw=mx8"


def model_presets(model: str) -> dict:
    vitl = model == "vitl16"
    p3_kw = P3_PATCH_KWARGS_VITL if vitl else P3_PATCH_KWARGS_VITB
    bb = p3_kw[R22]["variant"]
    assert p3_kw[R21]["variant"] == H3

    def with_(head=None, backbone=None, **extra_head):
        return {**p3_kw, R21: {"variant": head or H3, **extra_head}, R22: {"variant": backbone or bb}}

    out = dict(p23.model_presets(model))
    out.update({
        "P3_hr16": (P3_STACK, with_(backbone=bb + "+dyhr=16")),
        "P3_hr32": (P3_STACK, with_(backbone=bb + "+dyhr=32")),
        "P3_hr64": (P3_STACK, with_(backbone=bb + "+dyhr=64")),
        "P3_ex": (P3_STACK, with_(head=H3 + ",amax=exact", backbone=bb + "+amax=exact")),
        "P3_H2": (P3_STACK, with_(head=H2)),
        "P3_H2f": (P3_STACK, with_(head=H2F)),
    })
    return out


def register(round_dir: Path, model: str) -> None:
    """round23's registration (r19 presets, outputs redirected to ``round_dir``) + this round's presets."""
    from round16_full_graph import run as r16

    p23.register(round_dir, model)
    r16.PRESET_CFG.update(model_presets(model))


# round24's ``--q2`` quantizer switch is NOT used in round25 (all numbers are R21a's quantizer); keep the names the copied round23 scripts expect.
Q2_ACTIVE = False


def maybe_q2(force=None) -> bool:
    return False
