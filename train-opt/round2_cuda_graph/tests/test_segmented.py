from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
from torch import nn

from common.fast_forward import forward_train_v2
from common.rng import RngTape
from round2_cuda_graph.impl.patch import apply
from round2_cuda_graph.impl.segmented import GraphState, JointBlocks, PlainBlocks, remove


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 1
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.rope_embed = nn.Identity()
        self.norm = nn.Identity()

    def prepare_tokens_with_masks(self, images):
        return images.flatten(2).transpose(1, 2), images.shape[-2:]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.num_joint_blocks = 1
        self.num_queries = 1
        self.queries = nn.Embedding(1, 4)
        self.class_head = nn.Identity()
        self.mask_head = nn.Identity()
        self.upscale = nn.Identity()


def context():
    model = TinyModel()
    calls = []

    def rope(ctx, i, H, W):
        calls.append(f"rope{i}")
        return None

    def block(ctx, i, module, x, rope, mask):
        calls.append(f"block{i}")
        return x + module(x)

    def predict(ctx, x, grid):
        calls.append("predict")
        return x[:, :1, :1].reshape(x.shape[0], 1, 1, 1), x[:, :1, :2]

    def mask(ctx, j, logits, grid, tokens):
        calls.append(f"mask{j}")
        return None

    ctx = SimpleNamespace(
        model=model,
        train_model=SimpleNamespace(model=model),
        ops=SimpleNamespace(rope=rope, block=block, predict=predict, build_mask=mask),
        rng=RngTape(),
        host_probs=[1.0],
        nvtx=False,
        meta={"round1_components": ("nosync",)},
        cfg=SimpleNamespace(phase="early"),
    )
    return ctx, calls


def test_plain_and_joint_segments_preserve_forward_order_and_gradients():
    ctx, calls = context()
    image = torch.randn(2, 4, 2, 2, requires_grad=True)
    expected_mask, expected_class = forward_train_v2(ctx, image, True)
    expected_calls = list(calls)
    baseline_loss = sum(value.sum() for value in (*expected_mask, *expected_class))
    baseline_loss.backward()
    baseline_grad = image.grad.clone()
    baseline_params = [p.grad.clone() for p in ctx.model.parameters() if p.grad is not None]

    second, calls = context()
    second.model.load_state_dict(ctx.model.state_dict())
    image2 = image.detach().clone().requires_grad_()
    tokens, size = second.model.backbone.prepare_tokens_with_masks(image2)
    tokens = PlainBlocks(second, *size)(tokens)
    tokens = torch.cat((second.model.queries.weight[None].expand(tokens.shape[0], -1, -1), tokens), dim=1)
    outputs = JointBlocks(second, *size, (2, 2))(tokens)
    final_mask, final_class = second.ops.predict(second, outputs[0], (2, 2))
    mask = [*outputs[1::2], final_mask]
    classes = [*outputs[2::2], final_class]
    assert calls == expected_calls
    for a, b in zip((*expected_mask, *expected_class), (*mask, *classes)):
        assert torch.allclose(a, b)
    sum(value.sum() for value in (*mask, *classes)).backward()
    assert torch.allclose(image2.grad, baseline_grad)
    for ref, candidate in zip(baseline_params, (p.grad for p in second.model.parameters() if p.grad is not None)):
        assert torch.allclose(ref, candidate)


def test_patch_is_additive_and_requires_nosync():
    ctx, _ = context()
    original = ctx.meta.get("forward_train_fn")
    apply(ctx)
    assert ctx.meta["forward_train_fn"] is not original
    assert ctx.meta["round2_graph_state"].report()["plain_captured"] is False
    remove(ctx)
    assert ctx.meta["forward_train_fn"] is forward_train_v2
    ctx.meta["round1_components"] = ()
    try:
        apply(ctx)
    except RuntimeError as exc:
        assert "requires round1 nosync" in str(exc)
    else:
        raise AssertionError("round2 accepted an unsafe mask branch")


def test_capture_report_preserves_compile_failure():
    state = GraphState(original=forward_train_v2, variant="compile_default", max_joint_signatures=7)
    state.compile_error = "unsupported op"
    state.compile_unique_graphs = 2
    report = state.report()
    assert report["compile_error"] == "unsupported op"
    assert report["compile_unique_graphs"] == 2
    assert report["plain_captured"] is False
