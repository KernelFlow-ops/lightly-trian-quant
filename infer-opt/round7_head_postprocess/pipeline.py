"""Round 7 pipeline: fused GPU prep + eager backbone (bf16 autocast) + fused post.

    init(model)                       # once; builds FastPrep / FastPost singletons
    batch, meta = fast_prep(frame_bgr)
    ml, cl      = model_lowres(model, batch, precision)     # (B,Q,128,128), (B,Q,3); no 512x512 interpolate
    mask, conf  = fast_post(ml, cl, meta)                   # uint8 (H,W) numpy (pinned view), float
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "impl"))
sys.path.insert(0, str(_HERE.parent / "common"))

import model_factory as mf  # noqa: E402
import postfuse as pf  # noqa: E402

_PREP: pf.FastPrep | None = None
_POST: pf.FastPost | None = None


def init(model: torch.nn.Module) -> None:
    global _PREP, _POST
    dtype = next(model.parameters()).dtype
    norm = model.image_normalize
    crop = min(model.image_size)
    _PREP = pf.FastPrep(norm["mean"], norm["std"], crop, dtype)
    _POST = pf.FastPost(model.internal_class_to_class.tolist(), mf.HEAD_CLASS_ID, crop)


def fast_prep(frame_bgr: np.ndarray):
    assert _PREP is not None, "call init(model) first"
    return _PREP(frame_bgr)


def model_lowres(model, batch: torch.Tensor, precision: str = "bf16-mixed"):
    with mf.autocast_ctx(precision):
        ml, cl = model.forward_train(batch, return_logits_per_layer=False)
    return ml[-1], cl[-1]


def fast_post(mask_logits_lr, class_logits, meta, return_logits: bool = False):
    assert _POST is not None, "call init(model) first"
    return _POST(mask_logits_lr, class_logits, meta, return_logits)


def fast_e2e(model, frame_bgr, precision: str = "bf16-mixed"):
    with torch.inference_mode():
        batch, meta = fast_prep(frame_bgr)
        ml, cl = model_lowres(model, batch, precision)
        return fast_post(ml, cl, meta)
