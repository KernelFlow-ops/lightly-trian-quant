from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round3_loss_matcher.impl.ref_criterion import (
    InjectableMaskClassificationLoss,
    draw_layer_coordinates,
)

from lightly_train._task_models.dinov3_eomt_semantic_segmentation.mask_loss import (
    MaskClassificationLoss,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_injection_matches_transformers_bitwise() -> None:
    torch.cuda.set_per_process_memory_fraction(0.125)
    device = torch.device("cuda")
    kwargs = dict(
        num_points=32,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=5.0,
        dice_coefficient=5.0,
        class_coefficient=2.0,
        num_labels=2,
        no_object_coefficient=0.1,
    )
    targets = [
        {
            "masks": torch.randint(0, 2, (t, 32, 32), device=device).bool(),
            "labels": torch.arange(t, device=device) % 2,
        }
        for t in (1, 2)
    ]
    torch.manual_seed(17)
    pred_base = torch.randn(2, 8, 16, 16, device=device, requires_grad=True)
    class_base = torch.randn(2, 8, 3, device=device, requires_grad=True)
    pred_ref = pred_base.detach().clone().requires_grad_()
    class_ref = class_base.detach().clone().requires_grad_()
    original = MaskClassificationLoss(**kwargs).to(device)
    injectable = InjectableMaskClassificationLoss(**kwargs).to(device)

    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(100)
        expected = original(pred_base, targets, class_base)
        sum(expected.values()).backward()
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(100)
        coords = draw_layer_coordinates(
            targets, kwargs["num_points"], kwargs["oversample_ratio"],
            kwargs["importance_sample_ratio"], device,
        )
        actual = injectable(pred_ref, targets, class_ref, coords=coords)
        sum(actual.values()).backward()

    assert expected.keys() == actual.keys()
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key
    assert torch.equal(pred_base.grad, pred_ref.grad)
    assert torch.equal(class_base.grad, class_ref.grad)

    # Assignment is also checked directly under identical point coordinates.
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(100)
        mask_labels = [target["masks"].float() for target in targets]
        class_labels = [target["labels"] for target in targets]
        expected_indices = original.matcher(
            pred_base.detach(), class_base.detach(), mask_labels, class_labels
        )
    for (want_src, want_dst), (got_src, got_dst) in zip(
        expected_indices, injectable.matcher.last_indices
    ):
        assert torch.equal(want_src, got_src)
        assert torch.equal(want_dst, got_dst)
