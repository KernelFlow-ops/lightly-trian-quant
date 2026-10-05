"""Training-safe depthwise convolution and LayerNorm head building block.

This module keeps NHWC/channels-last storage through the depthwise and norm
boundary while delegating arithmetic and backward to PyTorch's tested CUDA
kernels. It is the correctness fallback used by the round6 ``dwln`` variant;
custom fusion remains an optional follow-up once a profiler window is available.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def depthwise_norm(x: Tensor, conv: nn.Conv2d, norm: nn.Module) -> Tensor:
    if conv.groups != x.shape[1] or conv.kernel_size != (3, 3) or conv.bias is not None:
        raise ValueError("Expected bias-free depthwise 3x3 convolution")
    channels_last = x.is_cuda
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    y = F.conv2d(x, conv.weight, None, stride=conv.stride,
                 padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
    # LayerNorm2D uses the last dimension as its normalized dimension.
    y = y.permute(0, 2, 3, 1).contiguous()
    y = F.layer_norm(y, (y.shape[-1],), norm.weight, norm.bias, norm.eps)
    y = y.permute(0, 3, 1, 2)
    return y.contiguous(memory_format=torch.channels_last) if channels_last else y.contiguous()
