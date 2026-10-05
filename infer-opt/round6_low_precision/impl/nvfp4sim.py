"""NVFP4 W4A4 fake quantization for model-level sensitivity screening.

Each K-axis group of 16 has an E4M3 block scale and each operand has one fp32
global scale. E2M1 values are dequantized for fp32 GEMM, then rounded to bf16
like the existing FP8 sensitivity simulator. This does not time a real FP4 GEMM.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
FP8 = torch.float8_e4m3fn


def fake_quant(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    K = x.shape[-1]
    if K % 16:
        raise ValueError("NVFP4 requires K divisible by 16")
    blocks = x.reshape(*x.shape[:-1], K // 16, 16)
    max_abs = blocks.abs().amax()
    global_scale = (max_abs / (6.0 * 448.0)).clamp_min(1e-30)
    block_abs = blocks.abs().amax(-1, keepdim=True)
    block_scale = (block_abs / (6.0 * global_scale)).clamp(max=448).to(FP8).float()
    block_scale = block_scale.clamp_min(2**-9)  # smallest positive E4M3 subnormal
    effective_scale = global_scale * block_scale
    scaled = (blocks / effective_scale).abs().clamp(max=6)
    thresholds = scaled.new_tensor(FP4_THRESHOLDS)
    levels = scaled.new_tensor(FP4_LEVELS)
    indices = torch.bucketize(scaled.contiguous(), thresholds)
    quant = levels[indices] * blocks.sign()
    return (quant * effective_scale).reshape(x.shape)


TYPES = {
    "fc1": lambda blk: blk.mlp.fc1,
    "fc2": lambda blk: blk.mlp.fc2,
}


class NVFP4SimLinear:
    def __init__(self, lin: torch.nn.Linear):
        self.wq = fake_quant(lin.weight.detach()).contiguous()
        self.bias = None if lin.bias is None else lin.bias.detach().to(torch.bfloat16).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xq = fake_quant(x)
        y = F.linear(xq, self.wq)
        if self.bias is not None:
            y = y + self.bias
        return y.to(torch.bfloat16)


@contextmanager
def patched(model, units):
    saved = []
    try:
        for i, t in units:
            lin = TYPES[t](model.backbone.blocks[i])
            lin.forward = NVFP4SimLinear(lin)
            saved.append(lin)
        yield
    finally:
        for lin in saved:
            lin.__dict__.pop("forward", None)


if __name__ == "__main__":
    x = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.] * 2)
    assert torch.equal(fake_quant(x), x)
