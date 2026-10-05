"""Correctness checks shared by all rounds.

Level A: single kernel vs torch op (same dtype) -> `tensor_diff`.
Level B: whole model vs fp32 eager reference -> `compare_raw`, `compare_predictions`.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def tensor_diff(out: Tensor, ref: Tensor) -> dict[str, float]:
    # float64: fp32 reductions over millions of elements distort cos noticeably.
    o = out.detach().double().flatten()
    r = ref.detach().double().flatten()
    diff = (o - r).abs()
    denom = r.abs().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(o, r, dim=0).item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
        "cos": cos,
        "nan": bool(torch.isnan(o).any().item()),
    }


def assert_close(out: Tensor, ref: Tensor, *, atol: float, rtol: float, name: str) -> None:
    torch.testing.assert_close(
        out.float(), ref.float(), atol=atol, rtol=rtol, msg=lambda m: f"{name}: {m}"
    )


def compare_raw(
    mask_logits: Tensor, class_logits: Tensor, ref_mask: Tensor, ref_class: Tensor
) -> dict[str, Any]:
    """Compare forward_backend outputs (mask logits at input res, class logits)."""
    m = tensor_diff(mask_logits, ref_mask)
    c = tensor_diff(class_logits, ref_class)
    # Binary mask agreement at the logit>0 decision boundary.
    a = mask_logits.float() > 0
    b = ref_mask.float() > 0
    inter = (a & b).flatten(2).sum(-1).float()
    union = (a | b).flatten(2).sum(-1).float()
    iou = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))
    return {
        "mask_logits": m,
        "class_logits": c,
        "mask_iou_min": iou.min().item(),
        "mask_iou_mean": iou.mean().item(),
        "class_argmax_agree": (
            class_logits.argmax(-1) == ref_class.argmax(-1)
        ).float().mean().item(),
    }


def compare_predictions(
    pred: dict[str, Tensor], ref: dict[str, Tensor]
) -> dict[str, Any]:
    """Compare postprocessed outputs {labels, masks, scores}.

    Predictions are matched by descending score; counts may differ at the threshold.
    """
    n_pred, n_ref = pred["scores"].numel(), ref["scores"].numel()
    out: dict[str, Any] = {"n_pred": n_pred, "n_ref": n_ref}
    n = min(n_pred, n_ref)
    if n == 0:
        out["match"] = n_pred == n_ref
        return out
    ip = pred["scores"].float().argsort(descending=True)[:n]
    ir = ref["scores"].float().argsort(descending=True)[:n]
    mp = pred["masks"][ip].bool()
    mr = ref["masks"][ir].bool()
    inter = (mp & mr).flatten(1).sum(-1).float()
    union = (mp | mr).flatten(1).sum(-1).float()
    iou = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))
    out.update(
        {
            "labels_equal": bool((pred["labels"][ip] == ref["labels"][ir]).all().item()),
            "score_max_abs_diff": (
                pred["scores"][ip].float() - ref["scores"][ir].float()
            ).abs().max().item(),
            "mask_iou_min": iou.min().item(),
            "mask_iou_mean": iou.mean().item(),
        }
    )
    return out
