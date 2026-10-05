from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_cpp_point_loss_small() -> None:
    from round3_loss_matcher.impl.point_loss import cuda_point_loss

    torch.cuda.set_per_process_memory_fraction(0.125)
    torch.manual_seed(0)
    pred = torch.randn(2, 16, 16, device="cuda", requires_grad=True)
    target = torch.randint(0, 2, (2, 32, 32), device="cuda", dtype=torch.uint8)
    coords = torch.rand(2, 257, 2, device="cuda")
    bce, dice = cuda_point_loss(pred, target, coords)
    (bce + dice).sum().backward()
    assert bool(torch.isfinite(bce).all())
    assert bool(torch.isfinite(dice).all())
    assert bool(torch.isfinite(pred.grad).all())

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round3_loss_matcher.impl.point_loss import fused_point_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale,points", [(1.0, 257), (8.0, 12544)])
def test_fused_point_loss_and_gradient(dtype: torch.dtype, scale: float, points: int) -> None:
    torch.cuda.set_per_process_memory_fraction(0.125)
    torch.manual_seed(82)
    masks = 2
    pred_ref = (scale * torch.randn(masks, 16, 16, device="cuda", dtype=dtype)).requires_grad_()
    pred_opt = pred_ref.detach().clone().requires_grad_()
    target = torch.randint(0, 2, (masks, 32, 32), device="cuda", dtype=torch.uint8)
    coords = torch.rand(masks, points, 2, device="cuda")
    coords[:, :4] = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device="cuda")

    point_grid = (2.0 * coords - 1.0).unsqueeze(2)
    pred_points = F.grid_sample(
        pred_ref[:, None].float(), point_grid, align_corners=False
    ).squeeze(1).squeeze(-1)
    target_points = F.grid_sample(
        target[:, None].float(), point_grid, align_corners=False
    ).squeeze(1).squeeze(-1)
    bce_ref = F.binary_cross_entropy_with_logits(
        pred_points.float(), target_points.float(), reduction="none"
    ).mean(-1)
    probability = pred_points.float().sigmoid()
    dice_ref = 1 - (
        2 * (probability * target_points.float()).sum(-1) + 1
    ) / (probability.sum(-1) + target_points.float().sum(-1) + 1)
    loss_ref = (bce_ref * 1.3 + dice_ref * 0.7).sum()
    loss_ref.backward()

    bce_opt, dice_opt = fused_point_loss(pred_opt, target, coords)
    loss_opt = (bce_opt * 1.3 + dice_opt * 0.7).sum()
    loss_opt.backward()

    tolerance = 1e-4 if dtype == torch.float32 else 5e-3
    assert torch.allclose(bce_ref, bce_opt, atol=tolerance, rtol=tolerance)
    assert torch.allclose(dice_ref, dice_opt, atol=tolerance, rtol=tolerance)
    assert torch.isfinite(pred_opt.grad).all()
    cosine = F.cosine_similarity(
        pred_ref.grad.float().flatten(), pred_opt.grad.float().flatten(), dim=0
    )
    assert cosine.item() >= 0.999
    assert torch.allclose(
        pred_ref.grad.float(), pred_opt.grad.float(), atol=tolerance, rtol=0.02
    )
