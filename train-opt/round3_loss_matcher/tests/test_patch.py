from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.ops import default_ops
from common.rng import RngTape
from round3_loss_matcher.impl.fast_criterion import FastMaskClassificationLoss
from round3_loss_matcher.impl.patch import apply
from round3_loss_matcher.impl.ref_criterion import (
    InjectableMaskClassificationLoss,
    LayerCoordinates,
    draw_layer_coordinates,
)

from lightly_train._task_models.dinov3_eomt_semantic_segmentation.mask_loss import (
    MaskClassificationLoss,
)


class _MockTrainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.criterion = MaskClassificationLoss(
            num_points=32, oversample_ratio=3.0, importance_sample_ratio=0.75,
            mask_coefficient=5.0, dice_coefficient=5.0, class_coefficient=2.0,
            num_labels=2, no_object_coefficient=0.1,
        )


def test_patch_keeps_state_and_original_rng_order_on_cpu() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(9)
    train_model = _MockTrainModel()
    ctx = SimpleNamespace(
        train_model=train_model,
        model=SimpleNamespace(backbone=SimpleNamespace(blocks=range(24))),
        ops=SimpleNamespace(loss_all_layers=None),
        rng=RngTape(), meta={},
    )
    before = {key: value.shape for key, value in train_model.state_dict().items()}
    apply(ctx, variant="batched_torch")
    assert {key: value.shape for key, value in train_model.state_dict().items()} == before
    installed_op = ctx.ops.loss_all_layers
    apply(ctx, variant="batched_torch")
    assert ctx.ops.loss_all_layers is installed_op

    targets = [
        {"masks": torch.randint(0, 2, (t, 32, 32)).bool(), "labels": torch.arange(t)}
        for t in (1, 2)
    ]
    masks_ref = [torch.randn(2, 8, 16, 16).requires_grad_() for _ in range(4)]
    classes_ref = [torch.randn(2, 8, 3).requires_grad_() for _ in range(4)]
    with torch.no_grad():
        for classes in classes_ref:
            classes[1, :, :] = -16
            classes[1, :, 2] = 16
            classes[1, 0, 1] = 32
            classes[1, 0, 2] = -16
            classes[1, 7, 0] = 32
            classes[1, 7, 2] = -16
    masks_opt = [x.detach().clone().requires_grad_() for x in masks_ref]
    classes_opt = [x.detach().clone().requires_grad_() for x in classes_ref]

    torch.manual_seed(42)
    ref_losses = {}
    for layer in range(4):
        layer_losses = train_model.criterion(masks_ref[layer], targets, classes_ref[layer])
        suffix = f"_block{21 + layer}" if layer < 3 else ""
        ref_losses.update({key + suffix: value for key, value in layer_losses.items()})
    ref_total = train_model.criterion.loss_total(ref_losses)
    ref_total.backward()

    torch.manual_seed(42)
    opt_total, opt_losses = ctx.ops.loss_all_layers(ctx, masks_opt, classes_opt, targets)
    opt_total.backward()
    assert ref_losses.keys() == opt_losses.keys()
    for key, expected in ref_losses.items():
        assert torch.allclose(expected, opt_losses[key], rtol=1e-3, atol=1e-4), key
    for expected, actual in zip(masks_ref + classes_ref, masks_opt + classes_opt):
        cosine = torch.nn.functional.cosine_similarity(
            expected.grad.flatten(), actual.grad.flatten(), dim=0
        )
        assert cosine.item() >= 0.999, cosine


def test_default_ops_record_replays_into_round3_patch() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(19)
    train_model = _MockTrainModel()
    model = SimpleNamespace(
        backbone=SimpleNamespace(blocks=range(24)),
        queries=SimpleNamespace(weight=torch.empty(0)),
    )
    train_model.model = model
    targets = [
        {"masks": torch.randint(0, 2, (t, 32, 32)).bool(), "labels": torch.arange(t)}
        for t in (1, 2)
    ]
    masks_ref = [torch.randn(2, 8, 16, 16).requires_grad_() for _ in range(4)]
    classes_ref = [torch.randn(2, 8, 3).requires_grad_() for _ in range(4)]
    masks_opt = [x.detach().clone().requires_grad_() for x in masks_ref]
    classes_opt = [x.detach().clone().requires_grad_() for x in classes_ref]
    recorded = RngTape(mode="record")
    reference_ctx = SimpleNamespace(
        train_model=train_model, model=model, ops=default_ops(), rng=recorded, meta={}
    )
    loss_ref, losses_ref = reference_ctx.ops.loss_all_layers(
        reference_ctx, masks_ref, classes_ref, targets
    )
    loss_ref.backward()
    replayed = RngTape(mode="replay", values=recorded.values)
    fast_ctx = SimpleNamespace(
        train_model=train_model, model=model, ops=default_ops(), rng=replayed, meta={}
    )
    apply(fast_ctx, variant="batched_torch")
    loss_opt, losses_opt = fast_ctx.ops.loss_all_layers(
        fast_ctx, masks_opt, classes_opt, targets
    )
    loss_opt.backward()
    replayed.assert_complete()
    assert len(recorded.values) == 16
    assert losses_ref.keys() == losses_opt.keys()
    for key, expected in losses_ref.items():
        assert torch.allclose(expected, losses_opt[key], rtol=1e-3, atol=1e-4), key
    for expected, actual in zip(masks_ref + classes_ref, masks_opt + classes_opt):
        cosine = torch.nn.functional.cosine_similarity(
            expected.grad.flatten(), actual.grad.flatten(), dim=0
        )
        assert cosine.item() >= 0.999, cosine


def test_empty_and_scipy_fallback_targets_on_cpu() -> None:
    torch.set_num_threads(4)
    kwargs = dict(
        num_points=32, oversample_ratio=3.0, importance_sample_ratio=0.75,
        mask_coefficient=5.0, dice_coefficient=5.0, class_coefficient=2.0,
        num_labels=2, no_object_coefficient=0.1,
    )
    for counts in ((0, 1), (1, 5)):
        torch.manual_seed(70)
        targets = [
            {"masks": torch.randint(0, 2, (count, 32, 32)).bool(),
             "labels": torch.arange(count) % 2}
            for count in counts
        ]
        mask_ref = torch.randn(2, 8, 16, 16, requires_grad=True)
        class_ref = torch.randn(2, 8, 3, requires_grad=True)
        mask_opt = mask_ref.detach().clone().requires_grad_()
        class_opt = class_ref.detach().clone().requires_grad_()
        reference = InjectableMaskClassificationLoss(**kwargs)
        fast = FastMaskClassificationLoss(reference, num_blocks=24)
        coords = draw_layer_coordinates(targets, 32, 3.0, 0.75, torch.device("cpu"))
        losses_ref = reference(mask_ref, targets, class_ref, coords=coords)
        loss_ref = reference.loss_total(losses_ref)
        loss_opt, losses_opt = fast.forward_all_layers(
            [mask_opt], [class_opt], targets, coords=[coords]
        )
        assert losses_ref.keys() == losses_opt.keys()
        for key, expected in losses_ref.items():
            assert torch.allclose(expected, losses_opt[key], rtol=1e-3, atol=1e-4), (counts, key)
        assert torch.allclose(loss_ref, loss_opt, rtol=1e-3, atol=1e-4)
        loss_ref.backward()
        loss_opt.backward()
        for expected, actual in ((mask_ref, mask_opt), (class_ref, class_opt)):
            cosine = torch.nn.functional.cosine_similarity(
                expected.grad.flatten(), actual.grad.flatten(), dim=0
            )
            assert cosine.item() >= 0.999, (counts, cosine)


def test_padded_targets_match_list_with_injected_coordinates() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(74)
    kwargs = dict(
        num_points=32, oversample_ratio=3.0, importance_sample_ratio=0.75,
        mask_coefficient=5.0, dice_coefficient=5.0, class_coefficient=2.0,
        num_labels=2, no_object_coefficient=0.1,
    )
    targets = [
        {"masks": torch.randint(0, 2, (t, 32, 32)).bool(), "labels": torch.arange(t)}
        for t in (1, 2)
    ]
    padded = {
        "masks": torch.stack([
            torch.nn.functional.pad(target["masks"].to(torch.uint8),
                                    (0, 0, 0, 0, 0, 2 - target["masks"].shape[0]))
            for target in targets
        ]),
        "labels": torch.tensor([[0, -1], [0, 1]]),
        "valid": torch.tensor([[True, False], [True, True]]),
    }
    mask_list = [torch.randn(2, 8, 16, 16, requires_grad=True) for _ in range(4)]
    class_list = [torch.randn(2, 8, 3, requires_grad=True) for _ in range(4)]
    mask_padded = [x.detach().clone().requires_grad_() for x in mask_list]
    class_padded = [x.detach().clone().requires_grad_() for x in class_list]
    reference = InjectableMaskClassificationLoss(**kwargs)
    fast_list = FastMaskClassificationLoss(reference, num_blocks=24)
    fast_padded = FastMaskClassificationLoss(reference, num_blocks=24)
    coords = [draw_layer_coordinates(targets, 32, 3.0, 0.75, torch.device("cpu"), layer_idx=i)
              for i in range(4)]
    positions = torch.tensor([0, 2, 3])
    padded_coords = []
    for layer in coords:
        over = torch.zeros(4, 96, 2)
        random = torch.zeros(4, 8, 2)
        over[positions] = layer.oversample
        random[positions] = layer.random
        padded_coords.append(LayerCoordinates(layer.matcher, over, random))

    list_loss, list_terms = fast_list.forward_all_layers(
        mask_list, class_list, targets, coords=coords
    )
    padded_loss, padded_terms = fast_padded.forward_all_layers(
        mask_padded, class_padded, padded, coords=padded_coords
    )
    assert list_terms.keys() == padded_terms.keys()
    for key in list_terms:
        assert torch.allclose(list_terms[key], padded_terms[key], rtol=1e-6, atol=1e-6), key
    list_loss.backward()
    padded_loss.backward()
    for expected, actual in zip(mask_list + class_list, mask_padded + class_padded):
        assert torch.allclose(expected.grad, actual.grad, rtol=1e-5, atol=1e-6)


def test_all_empty_list_preserves_random_call_sites() -> None:
    torch.set_num_threads(4)
    reference = _MockTrainModel().criterion
    fast = FastMaskClassificationLoss(reference, num_blocks=24)
    targets = [
        {"masks": torch.empty(0, 16, 16, dtype=torch.bool),
         "labels": torch.empty(0, dtype=torch.long)}
        for _ in range(2)
    ]
    calls = []

    def draw(name, shape, device):
        calls.append((name, shape))
        return torch.rand(shape, device=device)

    masks = [torch.randn(2, 8, 8, 8, requires_grad=True)]
    classes = [torch.randn(2, 8, 3, requires_grad=True)]
    total, losses = fast.forward_all_layers(
        masks, classes, targets, coord_provider=draw
    )
    assert [name for name, _ in calls] == [
        "match/L0/img0", "match/L0/img1", "loss_over/L0", "loss_rand/L0"
    ]
    assert torch.isfinite(total)
    total.backward()
    assert masks[0].grad is not None
    assert classes[0].grad is not None
