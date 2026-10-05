"""Forward path split into the public ops registry."""
from __future__ import annotations

from contextlib import nullcontext

import torch


def _range(ctx, name: str):
    if not ctx.nvtx:
        return nullcontext()
    from .bench import nvtx
    return nvtx(name)


def forward_train_v2(ctx, x: torch.Tensor, return_logits_per_layer: bool = True):
    model = ctx.train_model.model
    _, _, H, W = x.shape
    patch_size = model.backbone.patch_size
    assert patch_size is not None
    grid_size = (H // patch_size, W // patch_size)
    x, image_size = model.backbone.prepare_tokens_with_masks(x)
    mask_logits_per_layer: list[torch.Tensor] = []
    class_logits_per_layer: list[torch.Tensor] = []
    num_blocks = len(model.backbone.blocks)
    for i, block in enumerate(model.backbone.blocks):
        rope = None
        if model.backbone.rope_embed is not None:
            with _range(ctx, f"rope_L{i}"):
                rope = ctx.ops.rope(ctx, i, image_size[0], image_size[1])
        if i == num_blocks - model.num_joint_blocks:
            x = torch.cat((model.queries.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)
        if return_logits_per_layer and i >= num_blocks - model.num_joint_blocks:
            with _range(ctx, f"predict_L{i}"):
                mask_logits, class_logits = ctx.ops.predict(ctx, x, grid_size)
            mask_logits_per_layer.append(mask_logits)
            class_logits_per_layer.append(class_logits)
            attn_mask = None
            if model.training:
                j = i - num_blocks + model.num_joint_blocks
                with _range(ctx, f"mask_L{i}"):
                    attn_mask = ctx.ops.build_mask(ctx, j, mask_logits, grid_size, x.shape[1])
        else:
            attn_mask = None
        with _range(ctx, f"blk_{i:02d}"):
            x = ctx.ops.block(ctx, i, block, x, rope, attn_mask)
    with _range(ctx, "predict_final"):
        mask_logits, class_logits = ctx.ops.predict(ctx, x, grid_size)
    mask_logits_per_layer.append(mask_logits)
    class_logits_per_layer.append(class_logits)
    return mask_logits_per_layer, class_logits_per_layer
