from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round6_head_fusion.impl.fused_head import convt2x2_gemm, predict
from round6_head_fusion.impl.patch import apply

from lightly_train._task_models.dinov3_eomt_semantic_segmentation.scale_block import (
    ScaleBlock,
)


@pytest.mark.parametrize("channels", [8, 16])
def test_convt_gemm_forward_and_all_grads(channels: int) -> None:
    torch.set_num_threads(4)
    torch.manual_seed(32)
    x_ref = torch.randn(2, channels, 4, 5, requires_grad=True)
    w_ref = torch.randn(channels, channels, 2, 2, requires_grad=True)
    b_ref = torch.randn(channels, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_()
    w_opt = w_ref.detach().clone().requires_grad_()
    b_opt = b_ref.detach().clone().requires_grad_()
    want = torch.nn.functional.conv_transpose2d(x_ref, w_ref, b_ref, stride=2)
    got = convt2x2_gemm(x_opt, w_opt, b_opt)
    assert torch.allclose(want, got, rtol=1e-5, atol=1e-5)
    want.square().mean().backward()
    got.square().mean().backward()
    for ref, opt in ((x_ref, x_opt), (w_ref, w_opt), (b_ref, b_opt)):
        assert torch.allclose(ref.grad, opt.grad, rtol=1e-5, atol=1e-5)


class _Backbone(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.n_storage_tokens = 2


class _Head(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.backbone = _Backbone(channels)
        self.num_queries = 5
        self.class_head = nn.Linear(channels, 3)
        self.mask_head = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(),
            nn.Linear(channels, channels), nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.upscale = nn.Sequential(ScaleBlock(channels), ScaleBlock(channels))

    def original_predict(self, x: torch.Tensor, grid_size: tuple[int, int]):
        x = self.backbone.norm(x)
        query = x[:, :self.num_queries]
        class_logits = self.class_head(query)
        patch = x[:, self.num_queries + 1 + self.backbone.n_storage_tokens:]
        patch = patch.transpose(1, 2).reshape(x.shape[0], -1, *grid_size)
        mask_logits = torch.einsum(
            "bqc,bchw->bqhw", self.mask_head(query), self.upscale(patch)
        )
        return mask_logits, class_logits


@pytest.mark.parametrize("variant", ["bmm", "channels_last_bmm", "convt_gemm", "dwln", "all"])
def test_head_variants_match_original_forward_backward(variant: str) -> None:
    torch.set_num_threads(4)
    torch.manual_seed(34)
    reference = _Head(16)
    optimized = copy.deepcopy(reference)
    tokens = 5 + 1 + 2 + 4 * 4
    x_ref = torch.randn(2, tokens, 16, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_()
    expected_mask, expected_class = reference.original_predict(x_ref, (4, 4))
    actual_mask, actual_class = predict(optimized, x_opt, (4, 4), variant)
    assert actual_mask.is_contiguous()
    assert torch.allclose(expected_mask, actual_mask, rtol=1e-5, atol=1e-5)
    assert torch.equal(expected_class, actual_class)
    (expected_mask.square().mean() + expected_class.square().mean()).backward()
    (actual_mask.square().mean() + actual_class.square().mean()).backward()
    assert torch.allclose(x_ref.grad, x_opt.grad, rtol=1e-4, atol=1e-5)
    for (name, want), (opt_name, got) in zip(
        reference.named_parameters(), optimized.named_parameters()
    ):
        assert name == opt_name
        assert torch.allclose(want.grad, got.grad, rtol=1e-4, atol=1e-5), name


def test_patch_preserves_parameter_keys_and_is_idempotent() -> None:
    head = _Head(8)
    context = SimpleNamespace(model=head, ops=SimpleNamespace(predict=None), meta={})
    before = {key: value.shape for key, value in head.state_dict().items()}
    apply(context, variant="convt_gemm")
    first = context.ops.predict
    apply(context, variant="convt_gemm")
    assert context.ops.predict is first
    assert {key: value.shape for key, value in head.state_dict().items()} == before
