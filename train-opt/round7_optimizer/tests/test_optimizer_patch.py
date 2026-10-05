from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import torch
import pytest
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from round7_optimizer.impl.patch import apply
from round7_optimizer.impl.fused_optimizer import fused_clip_grad_norm_


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(4, 4, bias=False)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([_Block()])
        self.head = nn.Linear(4, 2)


class _TrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Model()
        self.model_args = SimpleNamespace(gradient_clip_val=0.01)

    def clip_gradients(self, fabric, optimizer):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 0.01)
        return norm.detach()


def _context(variant: str):
    train = _TrainModel()
    opt = torch.optim.AdamW(train.parameters(), lr=1e-3, foreach=False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    return SimpleNamespace(
        train_model=train,
        model=train.model,
        optimizer=opt,
        scheduler=sched,
        fabric=SimpleNamespace(),
        shadow={},
        post_opt_hooks=[],
        meta={},
        ops=SimpleNamespace(),
    )


def test_foreach_clip_matches_torch_global_norm() -> None:
    torch.manual_seed(5)
    first = [torch.randn(17, requires_grad=True), torch.randn(3, 4, requires_grad=True)]
    second = [value.detach().clone().requires_grad_() for value in first]
    for left, right in zip(first, second):
        grad = torch.randn_like(left)
        left.grad = grad.clone()
        right.grad = grad.clone()
    expected = torch.nn.utils.clip_grad_norm_(first, 0.01)
    actual = fused_clip_grad_norm_(second, 0.01)
    assert torch.allclose(expected, actual)
    for left, right in zip(first, second):
        assert torch.allclose(left.grad, right.grad)


def test_capturable_patch_keeps_device_lr_and_exports_compatible_state() -> None:
    ctx = _context("capturable_torch")
    apply(ctx, variant="capturable_torch")
    lr = ctx.optimizer.param_groups[0]["lr"]
    assert isinstance(lr, torch.Tensor)
    ptr = lr.data_ptr()
    for p in ctx.train_model.parameters():
        p.grad = torch.ones_like(p)
    old_step = ctx.scheduler.last_epoch
    ctx.ops.opt_step(ctx)
    assert lr.data_ptr() == ptr
    assert ctx.scheduler.last_epoch == old_step + 1
    exported = ctx.optimizer.state_dict()
    assert isinstance(exported["param_groups"][0]["lr"], float)


def test_capturable_patch_loads_a_regular_adamw_state_dict() -> None:
    source = _context("capturable_torch")
    for p in source.train_model.parameters():
        p.grad = torch.ones_like(p)
    source.optimizer.step()
    saved = source.optimizer.state_dict()
    target = _context("capturable_torch")
    apply(target, variant="capturable_torch")
    target.optimizer.load_state_dict(saved)
    assert torch.allclose(
        target.optimizer.param_groups[0]["lr"],
        torch.tensor(saved["param_groups"][0]["lr"]),
    )
    device = next(target.train_model.parameters()).device
    for state in target.optimizer.state.values():
        assert state["step"].device == device


def test_shadow_variant_populates_non_autograd_buffers_and_updates_after_step() -> None:
    ctx = _context("capturable_shadow")
    apply(ctx, variant="capturable_shadow")
    assert ctx.meta["shadow_writer"] == "round7_optimizer"
    assert ctx.shadow
    before = {key: value.clone() for key, value in ctx.shadow.items()}
    for p in ctx.train_model.parameters():
        p.grad = torch.ones_like(p)
    assert all(not value.requires_grad and value.grad_fn is None for value in ctx.shadow.values())
    ctx.ops.opt_step(ctx)
    assert any(not torch.equal(before[key], value) for key, value in ctx.shadow.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multi_tensor_variant_matches_adamw_one_step():
    torch.manual_seed(12)
    ctx = _context("multi_tensor")
    ctx.train_model.cuda()
    ctx.optimizer = torch.optim.AdamW(ctx.train_model.parameters(), lr=1e-3, foreach=False)
    ctx.scheduler = torch.optim.lr_scheduler.LambdaLR(ctx.optimizer, lambda _: 1.0)
    reference = [p.detach().clone().requires_grad_() for p in ctx.train_model.parameters()]
    ref_opt = torch.optim.AdamW(reference, lr=1e-3, foreach=False)
    for source, target in zip(ctx.train_model.parameters(), reference):
        grad = torch.randn_like(source)
        source.grad = grad.clone()
        target.grad = grad.clone()
    apply(ctx, variant="multi_tensor")
    ctx.ops.opt_step(ctx)
    torch.nn.utils.clip_grad_norm_(reference, 0.01)
    ref_opt.step()
    for source, target in zip(ctx.train_model.parameters(), reference):
        assert torch.allclose(source, target, atol=5e-7, rtol=5e-6)
