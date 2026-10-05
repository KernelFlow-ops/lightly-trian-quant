"""Head (``ScaleBlock`` upscale path) variants for BF16 autocast.

* ``head_cl``      - channels_last for the ConvTranspose / depthwise convolutions (round1 helper).
* ``head_ln_bf16`` - ``LayerNorm2D`` that keeps bf16 activations in and out (fp32 statistics inside
  the kernel, bf16-rounded affine parameters).  Under autocast the stock ``F.layer_norm`` is an fp32
  op: it casts the bf16 conv output to fp32, writes an fp32 result, and the next bf16 op (ConvT /
  einsum) casts it back.
"""
from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F


def install_head_cl(ctx) -> None:
    from round1_torch_cleanup.impl.patch import _install_head_cl

    _install_head_cl(ctx)


def _ln2d_bf16_forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float32 or not x.is_cuda:
        # fp32 / CPU path: exactly the source implementation
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)
    # torch's CUDA layer_norm needs weight/bias in the activation dtype (no mixed fp32-param path in
    # this torch build); the kernel still accumulates statistics in fp32.  weight/bias are rounded
    # to bf16 (relative error 2^-9), the only numerical difference to the autocast path.
    with torch.autocast("cuda", enabled=False):
        y = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)
    return y.permute(0, 3, 1, 2)


def install_ln2d_bf16(ctx) -> int:
    from lightly_train._task_models.dinov3_eomt_semantic_segmentation.scale_block import LayerNorm2D

    count = 0
    for module in ctx.model.upscale.modules():
        if isinstance(module, LayerNorm2D):
            module.forward = MethodType(_ln2d_bf16_forward, module)
            count += 1
    if not count:
        raise RuntimeError("no LayerNorm2D modules found in model.upscale")
    return count
