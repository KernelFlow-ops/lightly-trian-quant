"""Batch the deterministic RoPE arithmetic for plain ViT blocks.

Each block still draws its own scalar with the source's uniform_ call. Joint
blocks use the source implementation so mask RNG stays interleaved correctly.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def plain_rope_batch(rope, count: int, H: int, W: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if count <= 0:
        return []
    if rope.shift_coords is not None or rope.jitter_coords is not None or rope.rescale_coords is None:
        raise ValueError("plain_rope_batch requires rescale-only training RoPE")
    dd = {"device": rope.periods.device, "dtype": rope.dtype}
    if rope.normalize_coords == "max":
        denominator = max(H, W)
        h = torch.arange(0.5, H, **dd) / denominator
        w = torch.arange(0.5, W, **dd) / denominator
    elif rope.normalize_coords == "min":
        denominator = min(H, W)
        h = torch.arange(0.5, H, **dd) / denominator
        w = torch.arange(0.5, W, **dd) / denominator
    elif rope.normalize_coords == "separate":
        h = torch.arange(0.5, H, **dd) / H
        w = torch.arange(0.5, W, **dd) / W
    else:
        raise ValueError(f"unknown normalize_coords: {rope.normalize_coords}")
    coords = torch.stack(torch.meshgrid(h, w, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0
    maximum = np.log(rope.rescale_coords)
    minimum = -maximum
    scales = [torch.empty(1, **dd).uniform_(minimum, maximum).exp() for _ in range(count)]
    batched_coords = coords[None, :, :] * torch.stack(scales).reshape(count, 1, 1)
    angles = (2 * math.pi * batched_coords[:, :, :, None] / rope.periods[None, None, None, :]).flatten(2, 3)
    angles = torch.cat((angles, angles), dim=-1)
    sin, cos = torch.sin(angles), torch.cos(angles)
    return [(sin[i], cos[i]) for i in range(count)]
