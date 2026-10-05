from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from common.rng import RngTape
from lightly_train._models.dinov3.dinov3_src.layers.attention import LinearKMaskedBias, SelfAttention
from lightly_train._models.dinov3.dinov3_src.layers.rope_position_encoding import RopePositionEmbedding
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model import DINOv3EoMTSemanticSegmentation
from round1_torch_cleanup.impl.mask import attention_with_broadcast, build_mask
from round1_torch_cleanup.impl.io import AsyncCheckpointWriter, validation_context
from round1_torch_cleanup.impl.patch import _components, _install_adamw_fused, _install_head_cl, _install_kmask_bias, _install_nosync, apply
from round1_torch_cleanup.impl.rope import plain_rope_batch


def _mask_ctx(prob: float):
    model = SimpleNamespace(
        num_queries=2,
        backbone=SimpleNamespace(n_storage_tokens=1),
        attn_mask_probs=torch.tensor([prob]),
    )
    return SimpleNamespace(model=model, host_probs=[prob], rng=RngTape())


def _reference_mask(ctx, logits, grid, tokens):
    q = ctx.model.num_queries
    result = torch.ones(logits.shape[0], tokens, tokens, dtype=torch.bool)
    interpolated = F.interpolate(logits, size=grid, mode="bilinear").view(logits.shape[0], q, -1)
    patch = q + 1 + ctx.model.backbone.n_storage_tokens
    result[:, :q, patch:] = interpolated > 0
    if ctx.model.attn_mask_probs[0] < 1:
        disabled = torch.rand(logits.shape[0], q) > ctx.model.attn_mask_probs[0]
        result[:, :q, patch:][disabled] = True
    return result


def test_nosync_mask_matches_source_and_rng():
    logits = torch.randn(2, 2, 4, 4)
    for prob in (1.0, 0.5, 0.0):
        ctx = _mask_ctx(prob)
        torch.manual_seed(43)
        expected = _reference_mask(ctx, logits, (2, 2), 8)
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(43)
        actual = build_mask(ctx, 0, logits, (2, 2), 8, skip_zero=False, broadcast=False)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.random.get_rng_state(), expected_state)


def test_mask_skip_preserves_rng_and_broadcast_attention_gradient():
    logits = torch.randn(2, 2, 4, 4)
    ctx = _mask_ctx(0.0)
    torch.manual_seed(8)
    _reference_mask(ctx, logits, (2, 2), 8)
    expected_state = torch.random.get_rng_state()
    torch.manual_seed(8)
    assert build_mask(ctx, 0, logits, (2, 2), 8, skip_zero=True, broadcast=True) is None
    assert torch.equal(torch.random.get_rng_state(), expected_state)

    ctx = _mask_ctx(0.5)
    torch.manual_seed(9)
    dense = _reference_mask(ctx, logits, (2, 2), 8)
    torch.manual_seed(9)
    broadcast = build_mask(ctx, 0, logits, (2, 2), 8, skip_zero=True, broadcast=True)
    assert broadcast is not None and broadcast.shape == (2, 1, 8, 8)
    assert torch.equal(broadcast[:, 0], dense)
    module_a = SelfAttention(dim=16, num_heads=2)
    module_b = copy.deepcopy(module_a)
    x_a = torch.randn(2, 8, 16, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_()
    source = DINOv3EoMTSemanticSegmentation._attn(None, module_a, x_a, rope=None, mask=dense)
    optimized = attention_with_broadcast(None, SimpleNamespace(attn=module_b), x_b, None, broadcast)
    assert torch.allclose(source, optimized, atol=1e-6, rtol=1e-6)
    source.sum().backward()
    optimized.sum().backward()
    assert torch.allclose(x_a.grad, x_b.grad, atol=1e-6, rtol=1e-6)
    for a, b in zip(module_a.parameters(), module_b.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-6)


def test_rope_batch_preserves_random_calls_and_values():
    rope = RopePositionEmbedding(embed_dim=64, num_heads=1, base=100.0, rescale_coords=2.0)
    rope.train()
    torch.manual_seed(123)
    reference = [rope(H=4, W=4) for _ in range(3)]
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(123)
    batched = plain_rope_batch(rope, 3, 4, 4)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for (sin_ref, cos_ref), (sin, cos) in zip(reference, batched):
        assert torch.allclose(sin_ref, sin, atol=1e-6, rtol=1e-6)
        assert torch.allclose(cos_ref, cos, atol=1e-6, rtol=1e-6)


def test_host_annealing_updates_mirror_without_tensor_branch():
    model = SimpleNamespace(num_queries=2, backbone=SimpleNamespace(n_storage_tokens=1), attn_mask_probs=torch.ones(3))
    train_model = SimpleNamespace(model_args=SimpleNamespace(poly_power=0.9))
    ctx = SimpleNamespace(model=model, train_model=train_model, host_probs=[1.0, 1.0, 1.0], rng=RngTape(), meta={})
    _install_nosync(ctx)
    values = [train_model.mask_annealing(20 * (i + 1), 50, 20 * (i + 2)) for i in range(3)]
    assert all(isinstance(value, float) for value in values)
    assert values == ctx.host_probs
    assert values[0] == 0.0
    assert abs(values[1] - 0.5**0.9) < 1e-7
    assert values[2] == 1.0


def test_zero_kmask_bias_effective_gradient_and_update():
    module = LinearKMaskedBias(4, 12, bias=True)
    module.bias_mask.zero_()
    with torch.no_grad():
        module.bias.uniform_(-0.1, 0.1)
    baseline = copy.deepcopy(module)
    optimized = torch.nn.Sequential(module)
    ctx = SimpleNamespace(model=optimized)
    assert _install_kmask_bias(ctx) == 1
    x_ref = torch.randn(3, 4, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_()
    out_ref = baseline(x_ref)
    out_opt = optimized(x_opt)
    assert torch.equal(out_ref, out_opt)
    out_ref.square().sum().backward()
    out_opt.square().sum().backward()
    assert torch.equal(baseline.bias.grad, module.bias.grad)
    assert torch.allclose(baseline.weight.grad, module.weight.grad)
    opt_ref = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    opt_new = torch.optim.AdamW(module.parameters(), lr=1e-3)
    opt_ref.step()
    opt_new.step()
    assert torch.allclose(baseline.bias, module.bias)


def test_default_keeps_source_precision_and_patch_is_idempotent():
    model = SimpleNamespace(num_queries=2, backbone=SimpleNamespace(n_storage_tokens=1), attn_mask_probs=torch.ones(3))
    train_model = SimpleNamespace(model_args=SimpleNamespace(poly_power=0.9))
    ops = SimpleNamespace(build_mask=lambda *args: None, attn=lambda *args: None)
    ctx = SimpleNamespace(model=model, train_model=train_model, ops=ops, host_probs=[1.0] * 3, rng=RngTape(), meta={}, cfg=SimpleNamespace(precision="bf16-mixed"))
    apply(ctx)
    assert "enable_autocast" not in ctx.meta
    apply(ctx, "mask_skip")
    installed_mask = ctx.ops.build_mask
    installed_attn = ctx.ops.attn
    apply(ctx, "mask_skip")
    assert ctx.ops.build_mask is installed_mask
    assert ctx.ops.attn is installed_attn
    assert _components("mask_skip") == ["nosync", "mask_skip"]


def test_async_checkpoint_snapshot_is_immutable():
    with tempfile.TemporaryDirectory(dir="/root/autodl-tmp/train-opt-tmp/A") as directory:
        target = Path(directory) / "state.pt"
        state = {"train_model": {"weight": torch.tensor([1.0, 2.0])}, "step": 7}
        with AsyncCheckpointWriter() as writer:
            writer.submit(state, target)
            state["train_model"]["weight"].fill_(9.0)
        loaded = torch.load(target, weights_only=True)
        assert torch.equal(loaded["train_model"]["weight"], torch.tensor([1.0, 2.0]))
        with validation_context():
            assert not torch.is_grad_enabled()


def test_head_channels_last_keeps_parameter_identity_and_state_shapes():
    model = torch.nn.Module()
    model.upscale = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2),
        torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
    )
    before = {name: (id(parameter), tuple(parameter.shape)) for name, parameter in model.named_parameters()}
    _install_head_cl(SimpleNamespace(model=model))
    after = {name: (id(parameter), tuple(parameter.shape)) for name, parameter in model.named_parameters()}
    assert before == after
    model.upscale(torch.randn(2, 4, 2, 2)).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_fused_adamw_preserves_groups_and_one_step_update():
    baseline_param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    fused_param = torch.nn.Parameter(baseline_param.detach().clone())
    groups_baseline = [{"params": [baseline_param], "lr": 1e-3, "weight_decay": 0.05}]
    groups_fused = [{"params": [fused_param], "lr": 1e-3, "weight_decay": 0.05}]
    baseline = torch.optim.AdamW(groups_baseline)
    fused = torch.optim.AdamW(groups_fused)
    _install_adamw_fused(SimpleNamespace(optimizer=fused))
    assert fused._step_supports_amp_scaling is True
    assert [(group["lr"], group["weight_decay"]) for group in fused.param_groups] == [(group["lr"], group["weight_decay"]) for group in baseline.param_groups]
    baseline_param.grad = torch.tensor([0.4, -0.1])
    fused_param.grad = baseline_param.grad.clone()
    baseline.step()
    fused.step()
    assert torch.allclose(baseline_param, fused_param, atol=1e-7, rtol=1e-7)
