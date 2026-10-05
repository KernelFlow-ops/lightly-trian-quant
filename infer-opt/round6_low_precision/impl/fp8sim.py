"""FP8 (e4m3) simulation of nn.Linear layers for sensitivity studies (no real FP8 GEMM).

Granularities (activations / weights):
  tensor : dynamic per-tensor absmax / static per-tensor
  row    : dynamic per-token absmax / static per-output-channel      (== torch._scaled_mm rowwise)
  b32    : 1x32 blocks along K, power-of-two (E8M0) scales, MXFP8-style, both operands
  b128   : 1x128 blocks along K, fp32 scales, both operands (DeepSeek-style)
The quantized values are exactly representable e4m3; the GEMM is done in fp32 on the dequantized
values (exact emulation of an fp32-accumulating FP8 tensor core GEMM up to summation order), then
the bf16 output rounding of the autocast baseline is applied.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP8 = torch.float8_e4m3fn


def _pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    s = (amax / FP8_MAX).clamp_min(1e-30)
    return torch.exp2(torch.ceil(torch.log2(s)))


def fake_quant(x: torch.Tensor, gran: str) -> torch.Tensor:
    """Quantize->dequantize x (..., K) in fp32 with the given granularity (scales along last dim)."""
    x = x.float()
    if gran == "tensor":
        s = (x.abs().amax() / FP8_MAX).clamp_min(1e-30)
    elif gran == "row":
        s = (x.abs().amax(-1, keepdim=True) / FP8_MAX).clamp_min(1e-30)
    elif gran in ("b32", "b128"):
        bs = 32 if gran == "b32" else 128
        K = x.shape[-1]
        assert K % bs == 0
        xb = x.reshape(*x.shape[:-1], K // bs, bs)
        amax = xb.abs().amax(-1, keepdim=True)
        s = _pow2_scale(amax) if gran == "b32" else (amax / FP8_MAX).clamp_min(1e-30)
        q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(FP8).float()
        return (q * s).reshape(x.shape)
    else:
        raise ValueError(gran)
    q = (x / s).clamp(-FP8_MAX, FP8_MAX).to(FP8).float()
    return q * s


class FP8SimLinear:
    def __init__(self, lin: torch.nn.Linear, gran: str, act_gran: str | None = None):
        self.gran = gran
        self.act_gran = act_gran or gran
        w = lin.weight.detach()
        # weights: per-tensor static / per-output-channel static / blocks along K
        wg = {"tensor": "tensor", "row": "row"}.get(gran, gran)
        self.wq = fake_quant(w, wg).contiguous()  # fp32 dequantized
        self.bias = None if lin.bias is None else lin.bias.detach().to(torch.bfloat16).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xq = fake_quant(x, self.act_gran)
        y = F.linear(xq, self.wq)
        if self.bias is not None:
            y = y + self.bias
        return y.to(torch.bfloat16)


TYPES = {
    "qkv": lambda blk: blk.attn.qkv,
    "proj": lambda blk: blk.attn.proj,
    "fc1": lambda blk: blk.mlp.fc1,
    "fc2": lambda blk: blk.mlp.fc2,
}


@contextmanager
def patched(model, units: list[tuple[int, str]], gran: str, act_gran: str | None = None):
    """Temporarily replace forwards of the (layer, type) units by FP8 simulation."""
    blocks = model.backbone.blocks
    saved = []
    try:
        for i, t in units:
            lin = TYPES[t](blocks[i])
            sim = FP8SimLinear(lin, gran, act_gran)
            saved.append(lin)
            lin.forward = sim  # instance attribute shadows Module.forward
        yield
    finally:
        for lin in saved:
            lin.__dict__.pop("forward", None)
