"""Register the batched Mask2Former loss with the common ops table."""

from __future__ import annotations

from .fast_criterion import FastMaskClassificationLoss

NAME = "round3_loss_matcher"
REQUIRES: tuple[str, ...] = ()
CONFLICTS: tuple[str, ...] = ()


def apply(ctx, variant: str = "batched_torch", **kw) -> dict:
    if kw:
        raise ValueError(f"Unknown round3 options: {sorted(kw)}")
    if variant not in ("batched_torch", "fused", "cuda"):
        raise ValueError(f"Unknown round3 variant: {variant}")
    installed = ctx.meta.get(NAME)
    if installed is not None:
        if installed.variant != variant:
            raise RuntimeError("round3 already installed with a different variant")
        return {"ops": ["loss_all_layers"], "desc": f"{NAME}:{variant} (already installed)"}

    num_blocks = len(ctx.model.backbone.blocks)
    criterion = FastMaskClassificationLoss(
        ctx.train_model.criterion, num_blocks=num_blocks, variant=variant
    ).to(next(ctx.train_model.parameters()).device)

    def loss_all_layers(ctx, mask_logits_list, class_logits_list, binary_masks):
        return criterion.forward_all_layers(
            mask_logits_list, class_logits_list, binary_masks,
            coord_provider=lambda name, shape, device: ctx.rng.rand(
                name, shape, device=device
            ),
        )

    ctx.ops.loss_all_layers = loss_all_layers
    ctx.meta[NAME] = criterion
    return {"ops": ["loss_all_layers"], "desc": f"{NAME}:{variant}"}
