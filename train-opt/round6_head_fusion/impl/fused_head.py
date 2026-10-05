"""Training-capable GEMM ConvT and channels-last EoMT head variants."""

from __future__ import annotations

import torch
from torch import Tensor

from .dwln import depthwise_norm


def convt2x2_gemm(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """Exact k=2,s=2 ConvTranspose2d as GEMM plus pixel rearrangement."""
    batch, channels_in, height, width = x.shape
    if weight.shape[0] != channels_in or weight.shape[-2:] != (2, 2):
        raise ValueError("Expected ConvTranspose2d weight (Cin,Cout,2,2)")
    channels_out = weight.shape[1]
    tokens = x.permute(0, 2, 3, 1).reshape(-1, channels_in)
    matrix = weight.permute(0, 2, 3, 1).reshape(channels_in, 4 * channels_out)
    z = torch.mm(tokens, matrix)
    z = z.reshape(batch, height, width, 2, 2, channels_out)
    nhwc = z.permute(0, 1, 3, 2, 4, 5).reshape(
        batch, 2 * height, 2 * width, channels_out
    )
    if bias is not None:
        nhwc = nhwc + bias
    return nhwc.permute(0, 3, 1, 2)


def predict(model, x: Tensor, grid_size: tuple[int, int], variant: str) -> tuple[Tensor, Tensor]:
    normalized = model.backbone.norm(x)
    query = normalized[:, :model.num_queries]
    class_logits = model.class_head(query)
    prefix = model.num_queries + 1 + model.backbone.n_storage_tokens
    patch = normalized[:, prefix:].transpose(1, 2).reshape(
        x.shape[0], -1, *grid_size
    )
    if variant in ("channels_last_bmm", "convt_gemm", "dwln", "all", "fused_dwln"):
        patch = patch.contiguous(memory_format=torch.channels_last)
    if variant in ("convt_gemm", "dwln", "all", "fused_dwln"):
        for stage in model.upscale:
            patch = convt2x2_gemm(patch, stage.conv1.weight, stage.conv1.bias)
            if variant == "fused_dwln":
                from .fused_dwln import gelu_depthwise_norm
                patch = gelu_depthwise_norm(patch, stage.conv2, stage.norm)
                continue
            patch = stage.act(patch)
            if variant in ("dwln", "all"):
                patch = depthwise_norm(patch, stage.conv2, stage.norm)
            else:
                patch = stage.norm(stage.conv2(patch))
    else:
        patch = model.upscale(patch)
    mask_embedding = model.mask_head(query)
    batch, channels, height, width = patch.shape
    nhwc = patch.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    mask_logits = torch.bmm(mask_embedding, nhwc.transpose(1, 2)).reshape(
        batch, model.num_queries, height, width
    ).contiguous()
    return mask_logits, class_logits
