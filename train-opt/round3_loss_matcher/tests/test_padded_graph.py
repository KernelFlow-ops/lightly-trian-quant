from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round3_loss_matcher.impl.fast_criterion import FastMaskClassificationLoss
from round3_loss_matcher.impl.ref_criterion import LayerCoordinates

from lightly_train._task_models.dinov3_eomt_semantic_segmentation.mask_loss import (
    MaskClassificationLoss,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_padded_loss_forward_backward_captures_and_replays() -> None:
    torch.cuda.set_per_process_memory_fraction(0.125)
    torch.manual_seed(94)
    reference = MaskClassificationLoss(
        num_points=32, oversample_ratio=3.0, importance_sample_ratio=0.75,
        mask_coefficient=5.0, dice_coefficient=5.0, class_coefficient=2.0,
        num_labels=2, no_object_coefficient=0.1,
    ).cuda()
    fast = FastMaskClassificationLoss(reference, num_blocks=24).cuda()
    masks = [torch.randn(2, 8, 16, 16, device="cuda", requires_grad=True)]
    classes = [torch.randn(2, 8, 3, device="cuda", requires_grad=True)]
    padded = {
        "masks": torch.randint(0, 2, (2, 2, 32, 32), device="cuda", dtype=torch.uint8),
        "labels": torch.tensor([[0, 1], [0, 1]], device="cuda"),
        "valid": torch.tensor([[True, False], [True, True]], device="cuda"),
    }
    coords = [LayerCoordinates(
        matcher=[torch.rand(1, 32, 2, device="cuda") for _ in range(2)],
        oversample=torch.rand(4, 96, 2, device="cuda"),
        random=torch.rand(4, 8, 2, device="cuda"),
    )]
    # Warm up on the same stream used for capture.  Running autograd on a
    # side stream creates AccumulateGrad nodes owned by that stream while the
    # leaf tensors were created on the default stream; PyTorch then warns on
    # the first replay about a stream mismatch and inserts a synchronization.
    # Same-stream warmup keeps the graph capture free of that hidden sync.
    for _ in range(3):
        value = fast.forward_all_layers(masks, classes, padded, coords=coords)[0]
        warmup_grads = torch.autograd.grad(value, masks + classes)
        del value, warmup_grads
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = fast.forward_all_layers(masks, classes, padded, coords=coords)[0]
        gradients = torch.autograd.grad(loss, masks + classes)
    expected = fast.forward_all_layers(masks, classes, padded, coords=coords)[0]
    expected_grads = torch.autograd.grad(expected, masks + classes)
    graph.replay()
    torch.cuda.synchronize()
    before = loss.detach().clone()
    assert torch.isfinite(before)
    torch.testing.assert_close(before, expected)
    for actual, reference_grad in zip(gradients, expected_grads):
        torch.testing.assert_close(actual, reference_grad)
    assert all(torch.isfinite(grad).all() for grad in gradients)

    padded["valid"].copy_(torch.tensor([[False, True], [True, False]], device="cuda"))
    expected = fast.forward_all_layers(masks, classes, padded, coords=coords)[0]
    expected_grads = torch.autograd.grad(expected, masks + classes)
    graph.replay()
    torch.cuda.synchronize()
    after = loss.detach().clone()
    assert torch.isfinite(after)
    torch.testing.assert_close(after, expected)
    for actual, reference_grad in zip(gradients, expected_grads):
        torch.testing.assert_close(actual, reference_grad)
    assert not torch.equal(before, after)
    assert all(torch.isfinite(grad).all() for grad in gradients)
