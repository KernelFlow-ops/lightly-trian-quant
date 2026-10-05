"""Batched, synchronization-free Mask2Former loss for fixed-shape targets."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .matcher import assign_topk_exact
from .ref_criterion import LayerCoordinates, draw_layer_coordinates


class FastMaskClassificationLoss(nn.Module):
    def __init__(
        self, reference: nn.Module, num_blocks: int, variant: str = "batched_torch"
    ) -> None:
        super().__init__()
        if variant not in ("batched_torch", "fused", "cuda"):
            raise ValueError(f"Unknown loss variant: {variant}")
        self.variant = variant
        self.num_points = reference.num_points
        self.oversample_ratio = reference.oversample_ratio
        self.importance_sample_ratio = reference.importance_sample_ratio
        self.mask_coefficient = reference.mask_coefficient
        self.dice_coefficient = reference.dice_coefficient
        self.class_coefficient = reference.class_coefficient
        self.num_labels = reference.num_labels
        self.num_blocks = num_blocks
        self.register_buffer("empty_weight", reference.empty_weight.detach().clone())
        self.debug = False
        self.last_assignment: Tensor | None = None
        self.last_cost: Tensor | None = None

    def forward_all_layers(
        self,
        mask_list: list[Tensor],
        class_list: list[Tensor],
        targets: list[dict[str, Tensor]] | dict[str, Tensor],
        coords: list[LayerCoordinates] | None = None,
        coord_provider: Callable[[str, tuple[int, ...], torch.device], Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        layers = len(mask_list)
        if not layers or len(class_list) != layers:
            raise ValueError("Expected equally sized nonempty mask and class lists")
        batch, queries, height, width = mask_list[0].shape
        device, dtype = mask_list[0].device, mask_list[0].dtype
        if any(t.shape != mask_list[0].shape for t in mask_list):
            raise ValueError("All mask layers must have the same shape")
        padded = isinstance(targets, dict)
        if padded:
            target_masks_raw = targets["masks"].to(device=device, dtype=torch.uint8)
            target_labels = targets["labels"].to(device=device, dtype=torch.long)
            valid_base = targets["valid"].to(device=device, dtype=torch.bool)
            if target_masks_raw.ndim != 4 or target_masks_raw.shape[0] != batch:
                raise ValueError("Padded masks must have shape (B,Tmax,H,W)")
            max_targets = target_masks_raw.shape[1]
            if max_targets == 0 or max_targets > 4:
                raise ValueError("Capturable padded loss requires 1 <= Tmax <= 4")
            if target_labels.shape != (batch, max_targets) or valid_base.shape != (batch, max_targets):
                raise ValueError("Padded labels and valid must have shape (B,Tmax)")
            target_labels = target_labels.masked_fill(~valid_base, 0)
            total_masks = valid_base.sum().clamp_min(1)
        else:
            counts = [target["masks"].shape[0] for target in targets]
            if len(counts) != batch:
                raise ValueError("Target count must equal batch size")
            if any(target["masks"].shape[-2:] != targets[0]["masks"].shape[-2:] for target in targets):
                raise ValueError("batched_torch requires equal target spatial shapes")
            if not max(counts):
                if coords is None:
                    for layer in range(layers):
                        draw_layer_coordinates(
                            targets, self.num_points, self.oversample_ratio,
                            self.importance_sample_ratio, device, layer_idx=layer,
                            provider=coord_provider,
                        )
                return self._all_empty(mask_list, class_list)
            max_targets = max(counts)
            total_masks = float(sum(counts))
            target_masks_raw = torch.stack(
                [
                    F.pad(target["masks"].to(device=device, dtype=torch.uint8),
                          (0, 0, 0, 0, 0, max_targets - count))
                    for target, count in zip(targets, counts)
                ]
            )
            target_labels = torch.stack(
                [F.pad(target["labels"].to(device=device, dtype=torch.long),
                       (0, max_targets - count))
                 for target, count in zip(targets, counts)]
            )
            valid_base = (
                torch.arange(max_targets, device=device)[None, :]
                < torch.tensor(counts, device=device)[:, None]
            )

        if coords is None:
            if padded:
                draw = coord_provider or (
                    lambda _name, shape, dev: torch.rand(shape, device=dev)
                )
                sampled = int(self.num_points * self.oversample_ratio)
                random_count = self.num_points - int(self.importance_sample_ratio * self.num_points)
                coords = []
                for layer in range(layers):
                    matcher = [
                        draw(f"match/L{layer}/img{image}", (1, self.num_points, 2), device)
                        for image in range(batch)
                    ]
                    over = draw(
                        f"loss_over/L{layer}",
                        (batch * max_targets, sampled, 2), device,
                    )
                    random = (
                        draw(f"loss_rand/L{layer}",
                             (batch * max_targets, random_count, 2), device)
                        if random_count else None
                    )
                    coords.append(LayerCoordinates(matcher, over, random))
            else:
                coords = [
                    draw_layer_coordinates(
                        targets, self.num_points, self.oversample_ratio,
                        self.importance_sample_ratio, device, layer_idx=i,
                        provider=coord_provider,
                    )
                    for i in range(layers)
                ]
        if len(coords) != layers:
            raise ValueError("One coordinate set is required per layer")

        target_h, target_w = target_masks_raw.shape[-2:]
        target_masks = target_masks_raw.to(dtype)
        valid = valid_base.repeat(layers, 1).reshape(layers * batch, max_targets)
        labels = target_labels.repeat(layers, 1)
        masks = torch.stack(mask_list).reshape(layers * batch, queries, height, width)
        classes = torch.stack(class_list).reshape(layers * batch, queries, -1)
        match_coords = torch.stack(
            [torch.cat(layer.matcher, dim=0) for layer in coords]
        ).reshape(layers * batch, self.num_points, 2)

        with torch.no_grad():
            grid = (2.0 * match_coords - 1.0).unsqueeze(2)
            pred_points = F.grid_sample(
                masks, grid, align_corners=False
            ).squeeze(-1)
            target_points = F.grid_sample(
                target_masks.repeat(layers, 1, 1, 1).reshape(
                    layers * batch, max_targets, target_h, target_w
                ),
                grid, align_corners=False,
            ).squeeze(-1)
            x, y = pred_points.float(), target_points.float()
            cost_class = -classes.float().softmax(-1).gather(
                2, labels[:, None, :].expand(-1, queries, -1)
            )
            cost_mask = (
                F.softplus(x).sum(-1, keepdim=True)
                - torch.bmm(x, y.transpose(1, 2))
            ) / self.num_points
            prob = x.sigmoid()
            numerator = 2 * torch.bmm(prob, y.transpose(1, 2)) + 1
            denominator = prob.sum(-1, keepdim=True) + y.sum(-1)[:, None, :] + 1
            cost_dice = 1 - numerator / denominator
            cost = (
                self.mask_coefficient * cost_mask
                + self.class_coefficient * cost_class
                + self.dice_coefficient * cost_dice
            )
            cost = torch.nan_to_num(cost.clamp(-1e10, 1e10), nan=0)
            assignment = assign_topk_exact(cost, valid)
            self.last_assignment = assignment.reshape(layers, batch, max_targets)
            self.last_cost = cost.reshape(layers, batch, queries, max_targets)

        class_targets = torch.full(
            (layers * batch, queries + 1), self.num_labels,
            dtype=torch.long, device=device,
        )
        class_targets.scatter_(
            1, assignment.masked_fill(~valid, queries),
            labels.masked_fill(~valid, self.num_labels),
        )
        class_targets = class_targets[:, :queries]
        ce = F.cross_entropy(
            classes.transpose(1, 2), class_targets,
            weight=self.empty_weight.to(device), reduction="none",
        ).reshape(layers, batch, queries)
        denominator = self.empty_weight.to(device)[class_targets].reshape(
            layers, batch, queries
        ).sum(dim=(1, 2))
        class_loss = ce.sum(dim=(1, 2)) / denominator

        # SciPy returns row indices sorted by query, and the original criterion
        # assigns each oversampling-coordinate row in that same order.
        target_order = assignment.masked_fill(~valid, queries).argsort(dim=1)
        sorted_queries = assignment.gather(1, target_order).clamp_min(0)
        selected = masks.gather(
            1, sorted_queries[:, :, None, None].expand(
                -1, -1, height, width
            ),
        ).reshape(layers * batch * max_targets, 1, height, width)
        target_selected = target_masks_raw.repeat(layers, 1, 1, 1).reshape(
            layers * batch, max_targets, target_h, target_w
        ).gather(
            1, target_order[:, :, None, None].expand(
                -1, -1, target_h, target_w
            )
        ).reshape(
            layers * batch * max_targets, 1, target_h, target_w
        )
        sampled = int(self.num_points * self.oversample_ratio)
        if padded:
            over = torch.stack([layer.oversample for layer in coords]).reshape(
                layers * batch * max_targets, sampled, 2
            )
        else:
            positions = torch.tensor(
                [
                    (layer * batch + image) * max_targets + target
                    for layer in range(layers)
                    for image, count in enumerate(counts)
                    for target in range(count)
                ],
                dtype=torch.long, device=device,
            )
            over = torch.zeros(
                (layers * batch * max_targets, sampled, 2),
                device=device,
            ).index_copy_(0, positions, torch.cat([layer.oversample for layer in coords]))
        with torch.no_grad():
            over_logits = F.grid_sample(
                selected, (2.0 * over - 1.0).unsqueeze(2),
                align_corners=False,
            ).squeeze(1).squeeze(-1)
            uncertain = int(self.importance_sample_ratio * self.num_points)
            top_indices = (-over_logits.abs()).topk(uncertain, dim=1).indices
            point_coords = over.gather(
                1, top_indices[:, :, None].expand(-1, -1, 2)
            )
            random_count = self.num_points - uncertain
            if random_count:
                if padded:
                    random = torch.stack([layer.random for layer in coords]).reshape(
                        layers * batch * max_targets, random_count, 2
                    )
                else:
                    random = torch.zeros(
                        (layers * batch * max_targets, random_count, 2),
                        device=device,
                    ).index_copy_(
                        0, positions, torch.cat([layer.random for layer in coords])
                    )
                point_coords = torch.cat([point_coords, random], dim=1)
        if self.debug:
            self.last_point_coordinates = point_coords.detach()
        if self.variant == "fused":
            from .point_loss import fused_point_loss

            bce, dice = fused_point_loss(
                selected.squeeze(1), target_selected.squeeze(1), point_coords
            )
        elif self.variant == "cuda":
            from .point_loss import cuda_point_loss

            bce, dice = cuda_point_loss(
                selected.squeeze(1), target_selected.squeeze(1), point_coords
            )
        else:
            point_grid = (2.0 * point_coords - 1.0).unsqueeze(2)
            with torch.no_grad():
                point_labels = F.grid_sample(
                    target_selected.to(dtype), point_grid, align_corners=False
                ).squeeze(1).squeeze(-1)
            point_logits = F.grid_sample(
                selected, point_grid, align_corners=False
            ).squeeze(1).squeeze(-1)
            bce = F.binary_cross_entropy_with_logits(
                point_logits, point_labels, reduction="none"
            ).mean(-1)
            probs = point_logits.sigmoid()
            dice = 1 - (
                2 * (probs * point_labels).sum(-1) + 1
            ) / (probs.sum(-1) + point_labels.sum(-1) + 1)
        valid_loss = valid.reshape(layers, batch, max_targets)
        normalizer = total_masks
        mask_loss = (bce.reshape(layers, batch, max_targets) * valid_loss).sum(
            dim=(1, 2)
        ) / normalizer
        dice_loss = (dice.reshape(layers, batch, max_targets) * valid_loss).sum(
            dim=(1, 2)
        ) / normalizer

        losses: dict[str, Tensor] = {}
        first_block = self.num_blocks - (layers - 1)
        for layer in range(layers):
            suffix = f"_block{first_block + layer}" if layer < layers - 1 else ""
            losses[f"loss_mask{suffix}"] = mask_loss[layer]
            losses[f"loss_dice{suffix}"] = dice_loss[layer]
            losses[f"loss_cross_entropy{suffix}"] = class_loss[layer]
        total: Tensor | None = None
        for key, value in losses.items():
            if "mask" in key:
                weighted = value * self.mask_coefficient
            elif "dice" in key:
                weighted = value * self.dice_coefficient
            else:
                weighted = value * self.class_coefficient
            total = weighted if total is None else torch.add(total, weighted)
        assert total is not None
        return total, losses

    def _all_empty(
        self, mask_list: list[Tensor], class_list: list[Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        losses: dict[str, Tensor] = {}
        first_block = self.num_blocks - (len(mask_list) - 1)
        for i, (mask, classes) in enumerate(zip(mask_list, class_list)):
            suffix = f"_block{first_block + i}" if i < len(mask_list) - 1 else ""
            zero = mask.sum() * 0
            losses[f"loss_mask{suffix}"] = zero
            losses[f"loss_dice{suffix}"] = zero
            losses[f"loss_cross_entropy{suffix}"] = F.cross_entropy(
                classes.transpose(1, 2),
                torch.full(classes.shape[:2], self.num_labels, device=classes.device),
                weight=self.empty_weight.to(classes.device),
            )
        total = sum(
            value * (
                self.mask_coefficient if "mask" in key else
                self.dice_coefficient if "dice" in key else self.class_coefficient
            )
            for key, value in losses.items()
        )
        return total, losses
