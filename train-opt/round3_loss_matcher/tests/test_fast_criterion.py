from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round3_loss_matcher.impl.fast_criterion import FastMaskClassificationLoss
from round3_loss_matcher.impl.ref_criterion import (
    InjectableMaskClassificationLoss,
    draw_layer_coordinates,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("variant", ["batched_torch", "fused"])
def test_batched_loss_matches_reference(dtype: torch.dtype, variant: str) -> None:
    torch.cuda.set_per_process_memory_fraction(0.125)
    device = torch.device("cuda")
    kwargs = dict(
        num_points=64, oversample_ratio=3.0, importance_sample_ratio=0.75,
        mask_coefficient=5.0, dice_coefficient=5.0, class_coefficient=2.0,
        num_labels=2, no_object_coefficient=0.1,
    )
    torch.manual_seed(202)
    targets = [
        {
            "masks": torch.randint(0, 2, (t, 32, 32), device=device).bool(),
            "labels": torch.arange(t, device=device),
        }
        for t in (1, 2)
    ]
    masks_ref = [torch.randn(2, 8, 16, 16, device=device, dtype=dtype).requires_grad_()
                 for _ in range(4)]
    classes_ref = [torch.randn(2, 8, 3, device=device, dtype=dtype).requires_grad_()
                   for _ in range(4)]
    with torch.no_grad():
        for classes in classes_ref:
            classes[1, :, :] = -16
            classes[1, :, 2] = 16
            classes[1, 0, 1] = 32
            classes[1, 0, 2] = -16
            classes[1, 7, 0] = 32
            classes[1, 7, 2] = -16
    masks_fast = [x.detach().clone().requires_grad_() for x in masks_ref]
    classes_fast = [x.detach().clone().requires_grad_() for x in classes_ref]
    reference = InjectableMaskClassificationLoss(**kwargs).to(device)
    fast = FastMaskClassificationLoss(reference, num_blocks=24, variant=variant).to(device)
    coords = [
        draw_layer_coordinates(targets, 64, 3.0, 0.75, device, layer_idx=i)
        for i in range(4)
    ]

    ref_losses = {}
    ref_assignments = []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        for i in range(4):
            block_losses = reference(
                masks_ref[i], targets, classes_ref[i], coords=coords[i]
            )
            ref_assignments.append(reference.matcher.last_indices)
            suffix = f"_block{21 + i}" if i < 3 else ""
            ref_losses.update({key + suffix: value for key, value in block_losses.items()})
        ref_total = reference.loss_total(ref_losses)
        fast_total, fast_losses = fast.forward_all_layers(
            masks_fast, classes_fast, targets, coords=coords
        )
    ref_total.backward()
    fast_total.backward()

    assert ref_losses.keys() == fast_losses.keys()
    for layer, indices_per_image in enumerate(ref_assignments):
        for image, (src, dst) in enumerate(indices_per_image):
            for query, target in zip(src.tolist(), dst.tolist()):
                assert fast.last_assignment[layer, image, target].item() == query
    tolerance = 1e-3 if dtype == torch.float32 else 1e-2
    for key, want in ref_losses.items():
        got = fast_losses[key]
        assert (got - want).abs().item() <= tolerance * want.abs().item() + 1e-4, (key, want, got)
    assert (fast_total - ref_total).abs().item() <= tolerance * ref_total.abs().item() + 1e-4
    for want, got in zip(masks_ref + classes_ref, masks_fast + classes_fast):
        cosine = torch.nn.functional.cosine_similarity(
            want.grad.float().flatten(), got.grad.float().flatten(), dim=0
        )
        assert cosine.item() >= 0.999, cosine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_real_spatial_and_point_shapes(dtype: torch.dtype) -> None:
    torch.cuda.set_per_process_memory_fraction(0.125)
    device = torch.device("cuda")
    kwargs = dict(
        num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75,
        mask_coefficient=5.0, dice_coefficient=5.0, class_coefficient=2.0,
        num_labels=2, no_object_coefficient=0.1,
    )
    torch.manual_seed(311)
    targets = [
        {
            "masks": torch.randint(0, 2, (t, 512, 512), device=device).bool(),
            "labels": torch.arange(t, device=device),
        }
        for t in (1, 2)
    ]
    masks_ref = [torch.randn(2, 100, 128, 128, device=device, dtype=dtype).requires_grad_()
                 for _ in range(4)]
    classes_ref = [torch.randn(2, 100, 3, device=device, dtype=dtype).requires_grad_()
                   for _ in range(4)]
    masks_fast = [x.detach().clone().requires_grad_() for x in masks_ref]
    classes_fast = [x.detach().clone().requires_grad_() for x in classes_ref]
    reference = InjectableMaskClassificationLoss(**kwargs).to(device)
    fast = FastMaskClassificationLoss(reference, num_blocks=24, variant="fused").to(device)
    fast.debug = True
    coords = [draw_layer_coordinates(targets, 12544, 3.0, 0.75, device, layer_idx=i)
              for i in range(4)]
    ref_losses = {}
    ref_assignments = []
    ref_points = []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        for i in range(4):
            item = reference(masks_ref[i], targets, classes_ref[i], coords=coords[i])
            ref_assignments.append(reference.matcher.last_indices)
            ref_points.append(reference.last_point_coordinates)
            suffix = f"_block{21 + i}" if i < 3 else ""
            ref_losses.update({key + suffix: value for key, value in item.items()})
        loss_ref = reference.loss_total(ref_losses)
        loss_fast, fast_losses = fast.forward_all_layers(
            masks_fast, classes_fast, targets, coords=coords
        )
    loss_ref.backward()
    loss_fast.backward()
    got_points = fast.last_point_coordinates.reshape(4, 2, 2, 12544, 2)
    for layer in range(4):
        want_layer = ref_points[layer]
        mask_index = 0
        for image, count in enumerate((1, 2)):
            for target_index in range(count):
                got = got_points[layer, image, target_index]
                want = want_layer[mask_index]
                assert torch.equal(want, got)
                mask_index += 1
    for layer, per_image in enumerate(ref_assignments):
        for image, (src, dst) in enumerate(per_image):
            for query, target in zip(src.tolist(), dst.tolist()):
                assert fast.last_assignment[layer, image, target].item() == query
    rtol, atol = (1e-3, 1e-4) if dtype == torch.float32 else (0.01, 0.005)
    for key, value in ref_losses.items():
        assert torch.allclose(value, fast_losses[key], rtol=rtol, atol=atol), key
    assert torch.allclose(loss_ref, loss_fast, rtol=rtol, atol=atol)
    for want, got in zip(masks_ref + classes_ref, masks_fast + classes_fast):
        cosine = torch.nn.functional.cosine_similarity(
            want.grad.float().flatten(), got.grad.float().flatten(), dim=0
        )
        assert cosine.item() >= 0.999, cosine
