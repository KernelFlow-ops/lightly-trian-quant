"""A-level checks for the hand-written CUDA C++ round3 kernels."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round3_loss_matcher.impl.cuda_extension import load_cuda_ops
from round3_loss_matcher.impl.point_loss import cuda_point_loss


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_cuda_point_loss_matches_grid_sample_and_all_gradients() -> None:
    torch.cuda.set_per_process_memory_fraction(0.125)
    torch.manual_seed(501)
    pred = (8 * torch.randn(3, 17, 19, device="cuda", dtype=torch.float32)).requires_grad_()
    target = torch.randint(0, 2, (3, 33, 35), device="cuda", dtype=torch.uint8)
    coords = torch.rand(3, 257, 2, device="cuda")
    coords[:, :4] = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device="cuda")
    got_bce, got_dice = cuda_point_loss(pred, target, coords)
    got = (1.3 * got_bce + 0.7 * got_dice).sum()
    got.backward()

    ref_pred = pred.detach().clone().requires_grad_()
    grid = (2 * coords - 1).unsqueeze(2)
    px = F.grid_sample(ref_pred[:, None], grid, align_corners=False).squeeze(1).squeeze(-1)
    py = F.grid_sample(target[:, None].float(), grid, align_corners=False).squeeze(1).squeeze(-1)
    ref_bce = F.binary_cross_entropy_with_logits(px, py, reduction="none").mean(-1)
    p = px.sigmoid()
    ref_dice = 1 - (2 * (p * py).sum(-1) + 1) / (p.sum(-1) + py.sum(-1) + 1)
    (1.3 * ref_bce + 0.7 * ref_dice).sum().backward()
    assert torch.allclose(got_bce, ref_bce, atol=3e-5, rtol=3e-5)
    assert torch.allclose(got_dice, ref_dice, atol=3e-5, rtol=3e-5)
    assert torch.allclose(pred.grad, ref_pred.grad, atol=3e-5, rtol=3e-3)
    assert torch.nn.functional.cosine_similarity(pred.grad.flatten(), ref_pred.grad.flatten(), dim=0) > 0.9999


@pytest.mark.parametrize("targets", [1, 2, 3, 4])
def test_cuda_matcher_exact_cost_matches_scipy(targets: int) -> None:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    torch.cuda.set_per_process_memory_fraction(0.125)
    torch.manual_seed(502 + targets)
    cost = torch.randn(32, 100, targets, device="cuda", dtype=torch.float32)
    valid = torch.ones((32, targets), device="cuda", dtype=torch.bool)
    got = load_cuda_ops().matcher_exact(cost, valid).cpu().numpy()
    host = cost.cpu().numpy()
    for i in range(host.shape[0]):
        rows, cols = linear_sum_assignment(host[i])
        expected = rows[np.argsort(cols)]
        assert np.isclose(host[i, got[i], np.arange(targets)].sum(), host[i, expected, np.arange(targets)].sum(), atol=1e-5)

