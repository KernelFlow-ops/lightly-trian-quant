"""Triton point-sampled sigmoid CE and dice loss with logits backward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _bilinear(
    pointer, mask_id, px, py, height: tl.constexpr, width: tl.constexpr, active
):
    fx = px * width - 0.5
    fy = py * height - 0.5
    x0 = tl.floor(fx).to(tl.int32)
    y0 = tl.floor(fy).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = fx - x0
    dy = fy - y0
    base = pointer + mask_id * height * width
    a = tl.load(
        base + y0 * width + x0,
        mask=active & (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height),
        other=0,
    ).to(tl.float32)
    b = tl.load(
        base + y0 * width + x1,
        mask=active & (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height),
        other=0,
    ).to(tl.float32)
    c = tl.load(
        base + y1 * width + x0,
        mask=active & (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height),
        other=0,
    ).to(tl.float32)
    d = tl.load(
        base + y1 * width + x1,
        mask=active & (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height),
        other=0,
    ).to(tl.float32)
    value = ((a * (1 - dx) + b * dx) * (1 - dy) +
             (c * (1 - dx) + d * dx) * dy)
    return value, x0, x1, y0, y1, dx, dy


@triton.jit
def _points_forward(
    pred, target, coords, sampled_x, sampled_y, partial,
    pred_h: tl.constexpr, pred_w: tl.constexpr,
    target_h: tl.constexpr, target_w: tl.constexpr,
    points: tl.constexpr, chunks: tl.constexpr, block: tl.constexpr,
):
    chunk = tl.program_id(0)
    mask_id = tl.program_id(1)
    offsets = chunk * block + tl.arange(0, block)
    active = offsets < points
    px = tl.load(coords + (mask_id * points + offsets) * 2, active, other=0)
    py = tl.load(coords + (mask_id * points + offsets) * 2 + 1, active, other=0)
    x, _, _, _, _, _, _ = _bilinear(
        pred, mask_id, px, py, pred_h, pred_w, active
    )
    y, _, _, _, _, _, _ = _bilinear(
        target, mask_id, px, py, target_h, target_w, active
    )
    tl.store(sampled_x + mask_id * points + offsets, x, active)
    tl.store(sampled_y + mask_id * points + offsets, y, active)
    probability = 1.0 / (1.0 + tl.exp(-x))
    bce = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x))) - x * y
    base = partial + (mask_id * chunks + chunk) * 4
    tl.store(base, tl.sum(tl.where(active, bce, 0.0), 0))
    tl.store(base + 1, tl.sum(tl.where(active, probability * y, 0.0), 0))
    tl.store(base + 2, tl.sum(tl.where(active, probability, 0.0), 0))
    tl.store(base + 3, tl.sum(tl.where(active, y, 0.0), 0))


@triton.jit
def _points_reduce(
    partial, out_bce, out_dice, numerator, denominator,
    points: tl.constexpr, chunks: tl.constexpr, reduce_block: tl.constexpr,
):
    mask_id = tl.program_id(0)
    chunk = tl.arange(0, reduce_block)
    active = chunk < chunks
    base = partial + (mask_id * chunks + chunk) * 4
    bce_sum = tl.sum(tl.load(base, active, other=0), 0)
    overlap = tl.sum(tl.load(base + 1, active, other=0), 0)
    pred_sum = tl.sum(tl.load(base + 2, active, other=0), 0)
    target_sum = tl.sum(tl.load(base + 3, active, other=0), 0)
    n = 2.0 * overlap + 1.0
    d = pred_sum + target_sum + 1.0
    tl.store(out_bce + mask_id, bce_sum / points)
    tl.store(out_dice + mask_id, 1.0 - n / d)
    tl.store(numerator + mask_id, n)
    tl.store(denominator + mask_id, d)


@triton.jit
def _points_backward(
    coords, sampled_x, sampled_y, numerator, denominator,
    grad_bce, grad_dice, grad_pred,
    pred_h: tl.constexpr, pred_w: tl.constexpr,
    points: tl.constexpr, block: tl.constexpr,
):
    chunk = tl.program_id(0)
    mask_id = tl.program_id(1)
    offsets = chunk * block + tl.arange(0, block)
    active = offsets < points
    px = tl.load(coords + (mask_id * points + offsets) * 2, active, other=0)
    py = tl.load(coords + (mask_id * points + offsets) * 2 + 1, active, other=0)
    x = tl.load(sampled_x + mask_id * points + offsets, active, other=0)
    y = tl.load(sampled_y + mask_id * points + offsets, active, other=0)
    n = tl.load(numerator + mask_id)
    d = tl.load(denominator + mask_id)
    gb = tl.load(grad_bce + mask_id)
    gd = tl.load(grad_dice + mask_id)
    p = 1.0 / (1.0 + tl.exp(-x))
    grad_x = gb * (p - y) / points + gd * p * (1.0 - p) * (n - 2.0 * y * d) / (d * d)
    fx = px * pred_w - 0.5
    fy = py * pred_h - 0.5
    x0 = tl.floor(fx).to(tl.int32)
    y0 = tl.floor(fy).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = fx - x0
    dy = fy - y0
    base = grad_pred + mask_id * pred_h * pred_w
    tl.atomic_add(
        base + y0 * pred_w + x0, grad_x * (1 - dx) * (1 - dy),
        sem="relaxed", mask=active & (x0 >= 0) & (x0 < pred_w) & (y0 >= 0) & (y0 < pred_h),
    )
    tl.atomic_add(
        base + y0 * pred_w + x1, grad_x * dx * (1 - dy),
        sem="relaxed", mask=active & (x1 >= 0) & (x1 < pred_w) & (y0 >= 0) & (y0 < pred_h),
    )
    tl.atomic_add(
        base + y1 * pred_w + x0, grad_x * (1 - dx) * dy,
        sem="relaxed", mask=active & (x0 >= 0) & (x0 < pred_w) & (y1 >= 0) & (y1 < pred_h),
    )
    tl.atomic_add(
        base + y1 * pred_w + x1, grad_x * dx * dy,
        sem="relaxed", mask=active & (x1 >= 0) & (x1 < pred_w) & (y1 >= 0) & (y1 < pred_h),
    )


class _FusedPointLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred: Tensor, target: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        if pred.ndim != 3 or target.ndim != 3 or coords.ndim != 3:
            raise ValueError("Expected pred (M,H,W), target (M,Ht,Wt), coords (M,P,2)")
        if pred.shape[0] != target.shape[0] or pred.shape[0] != coords.shape[0]:
            raise ValueError("Mask dimensions must match")
        if not pred.is_cuda or not target.is_cuda or not coords.is_cuda:
            raise ValueError("Fused point loss requires CUDA tensors")
        pred, target, coords = pred.contiguous(), target.contiguous(), coords.contiguous()
        masks, pred_h, pred_w = pred.shape
        _, target_h, target_w = target.shape
        points = coords.shape[1]
        block = 256
        chunks = triton.cdiv(points, block)
        sampled_x = torch.empty((masks, points), device=pred.device, dtype=torch.float32)
        sampled_y = torch.empty_like(sampled_x)
        partial = torch.empty((masks, chunks, 4), device=pred.device, dtype=torch.float32)
        bce = torch.empty(masks, device=pred.device, dtype=torch.float32)
        dice = torch.empty_like(bce)
        numerator = torch.empty_like(bce)
        denominator = torch.empty_like(bce)
        _points_forward[(chunks, masks)](
            pred, target, coords, sampled_x, sampled_y, partial,
            pred_h, pred_w, target_h, target_w, points, chunks, block,
        )
        _points_reduce[(masks,)](
            partial, bce, dice, numerator, denominator,
            points, chunks, triton.next_power_of_2(chunks),
        )
        ctx.save_for_backward(coords, sampled_x, sampled_y, numerator, denominator)
        ctx.pred_shape = pred.shape
        ctx.pred_dtype = pred.dtype
        return bce, dice

    @staticmethod
    def backward(ctx, grad_bce: Tensor, grad_dice: Tensor) -> tuple[Tensor, None, None]:
        coords, sampled_x, sampled_y, numerator, denominator = ctx.saved_tensors
        masks, pred_h, pred_w = ctx.pred_shape
        points = coords.shape[1]
        if grad_bce is None:
            grad_bce = torch.zeros_like(numerator)
        if grad_dice is None:
            grad_dice = torch.zeros_like(numerator)
        grad_pred = torch.zeros(
            (masks, pred_h, pred_w), device=coords.device, dtype=torch.float32
        )
        _points_backward[(triton.cdiv(points, 256), masks)](
            coords, sampled_x, sampled_y, numerator, denominator,
            grad_bce.contiguous(), grad_dice.contiguous(), grad_pred,
            pred_h, pred_w, points, 256,
        )
        return grad_pred.to(ctx.pred_dtype), None, None


def fused_point_loss(pred: Tensor, target: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Return per-mask BCE and dice, leaving aggregation to the caller."""
    return _FusedPointLoss.apply(pred, target, coords)


class _CudaPointLoss(torch.autograd.Function):
    """CUDA C++ implementation of the point sampled loss."""

    @staticmethod
    def forward(ctx, pred: Tensor, target: Tensor, coords: Tensor):
        from .cuda_extension import load_cuda_ops

        if not (pred.is_cuda and target.is_cuda and coords.is_cuda):
            raise ValueError("CUDA point loss requires CUDA tensors")
        target = target.to(torch.uint8).contiguous()
        pred, coords = pred.contiguous(), coords.contiguous().to(torch.float32)
        masks, ph, pw = pred.shape
        points = coords.shape[1]
        bce = torch.empty((masks,), device=pred.device, dtype=torch.float32)
        dice = torch.empty_like(bce)
        sx = torch.empty((masks, points), device=pred.device, dtype=torch.float32)
        sy = torch.empty_like(sx)
        num = torch.empty_like(bce)
        den = torch.empty_like(bce)
        load_cuda_ops().point_loss_forward(pred, target, coords, bce, dice, sx, sy, num, den)
        ctx.save_for_backward(coords, sx, sy, num, den)
        ctx.pred_shape, ctx.pred_dtype = (masks, ph, pw), pred.dtype
        return bce, dice

    @staticmethod
    def backward(ctx, grad_bce: Tensor, grad_dice: Tensor):
        from .cuda_extension import load_cuda_ops

        coords, sx, sy, num, den = ctx.saved_tensors
        masks, ph, pw = ctx.pred_shape
        grad_bce = torch.zeros_like(num) if grad_bce is None else grad_bce.contiguous()
        grad_dice = torch.zeros_like(num) if grad_dice is None else grad_dice.contiguous()
        grad_pred = torch.zeros((masks, ph, pw), device=coords.device, dtype=torch.float32)
        load_cuda_ops().point_loss_backward(coords, sx, sy, num, den, grad_bce, grad_dice, grad_pred)
        return grad_pred.to(ctx.pred_dtype), None, None


def cuda_point_loss(pred: Tensor, target: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Run the hand-written CUDA C++ point loss kernel."""
    return _CudaPointLoss.apply(pred, target, coords)
