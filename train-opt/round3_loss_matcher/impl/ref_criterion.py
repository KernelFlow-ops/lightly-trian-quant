"""Mask2Former criterion with externally supplied random point coordinates.

The arithmetic and operation order intentionally match the installed transformers
implementation. This is a correctness oracle, not an optimized path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher,
    pair_wise_dice_loss,
    pair_wise_sigmoid_cross_entropy_loss,
    sample_point,
)

from lightly_train._task_models.dinov3_eomt_semantic_segmentation.mask_loss import (
    MaskClassificationLoss,
)


@dataclass
class LayerCoordinates:
    matcher: list[Tensor]
    oversample: Tensor
    random: Tensor | None


def draw_layer_coordinates(
    targets: list[dict[str, Tensor]],
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
    device: torch.device,
    layer_idx: int = 0,
    provider: Callable[[str, tuple[int, ...], torch.device], Tensor] | None = None,
) -> LayerCoordinates:
    """Consume RNG in the same order and shapes as one original loss layer."""
    draw = provider or (lambda _name, shape, dev: torch.rand(shape, device=dev))
    matcher = [
        draw(f"match/L{layer_idx}/img{i}", (1, num_points, 2), device)
        for i in range(len(targets))
    ]
    matched_masks = sum(target["masks"].shape[0] for target in targets)
    oversample = draw(
        f"loss_over/L{layer_idx}",
        (matched_masks, int(num_points * oversample_ratio), 2),
        device,
    )
    random_count = num_points - int(importance_sample_ratio * num_points)
    random = (
        draw(f"loss_rand/L{layer_idx}", (matched_masks, random_count, 2), device)
        if random_count > 0
        else None
    )
    return LayerCoordinates(matcher=matcher, oversample=oversample, random=random)


class CoordinateMatcher(Mask2FormerHungarianMatcher):
    coordinates: list[Tensor] | None = None
    last_indices: list[tuple[Tensor, Tensor]]

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
    ) -> list[tuple[Tensor, Tensor]]:
        if self.coordinates is None:
            self.last_indices = super().forward(
                masks_queries_logits, class_queries_logits, mask_labels, class_labels
            )
            return self.last_indices

        assignments = []
        for i in range(masks_queries_logits.shape[0]):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)[:, None]
            pred_mask = pred_mask[:, None]
            point_coordinates = self.coordinates[i]
            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = sample_point(
                target_mask, target_coordinates, align_corners=False
            ).squeeze(1)
            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = sample_point(
                pred_mask, pred_coordinates, align_corners=False
            ).squeeze(1)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            cost_matrix = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            assignments.append(linear_sum_assignment(cost_matrix.cpu()))

        self.last_indices = [
            (torch.as_tensor(src, dtype=torch.int64), torch.as_tensor(dst, dtype=torch.int64))
            for src, dst in assignments
        ]
        return self.last_indices


class InjectableMaskClassificationLoss(MaskClassificationLoss):
    _coordinates: LayerCoordinates | None = None

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.matcher = CoordinateMatcher(
            num_points=self.num_points,
            cost_mask=self.mask_coefficient,
            cost_dice=self.dice_coefficient,
            cost_class=self.class_coefficient,
        )

    def forward(
        self,
        masks_queries_logits: Tensor,
        targets: list[dict[str, Tensor]],
        class_queries_logits: Tensor | None = None,
        coords: LayerCoordinates | None = None,
    ) -> dict[str, Tensor]:
        self._coordinates = coords
        self.matcher.coordinates = coords.matcher if coords is not None else None
        try:
            return super().forward(masks_queries_logits, targets, class_queries_logits)
        finally:
            self._coordinates = None
            self.matcher.coordinates = None

    def sample_points_using_uncertainty(
        self,
        logits: Tensor,
        uncertainty_function: object,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
    ) -> Tensor:
        if self._coordinates is None:
            return super().sample_points_using_uncertainty(
                logits,
                uncertainty_function,
                num_points,
                oversample_ratio,
                importance_sample_ratio,
            )
        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)
        point_coordinates = self._coordinates.oversample
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        point_uncertainties = uncertainty_function(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(
            num_boxes, dtype=torch.long, device=logits.device
        )
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(
            num_boxes, num_uncertain_points, 2
        )
        if num_random_points > 0:
            assert self._coordinates.random is not None
            point_coordinates = torch.cat(
                [point_coordinates, self._coordinates.random], dim=1
            )
        self.last_point_coordinates = point_coordinates
        return point_coordinates
