"""GPU correctness tests for the round11 attention/head variants (operator level, vs fp32 references).

Every GPU process holds the dual lock (module fixture).  Skipped when CUDA is unavailable.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common.bench import gpu_lock
from common.ops import default_ops
from round11_bf16_mixed.impl.attn import AttnCfg, make_attn_core

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

B, H, D, NQ = 2, 16, 64, 100
STORAGE = 4
PREFIX = NQ + 1 + STORAGE


@pytest.fixture(scope="module", autouse=True)
def _lock():
    with gpu_lock():
        yield


def cos(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def rel_l2(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm())


def rope_tables(scale=8.0):
    pos = torch.arange(1024, device="cuda", dtype=torch.float32)
    freq = torch.exp(torch.linspace(0, -3, D // 2, device="cuda")) * scale
    angles = torch.cat([(pos[:, None] * freq[None]) * 0.05] * 2, dim=-1)
    return torch.sin(angles), torch.cos(angles)


def inputs(tokens, *, spiky, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(B, tokens, 3 * H * D, device="cuda", generator=gen)
    if spiky:  # register-token style: large queries/keys and a few huge-norm keys
        view = qkv.view(B, tokens, 3, H, D)
        view[:, :, :2] *= 3.0
        view[:, PREFIX - STORAGE : PREFIX, 1] *= 8.0
    go = torch.randn(B, tokens, H * D, device="cuda", generator=gen)
    return qkv.to(torch.bfloat16).requires_grad_(True), go.to(torch.bfloat16)


def masked(tokens, density, seed=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mask = torch.ones(B, 1, tokens, tokens, dtype=torch.bool, device="cuda")
    mask[:, 0, :NQ, PREFIX:] = torch.rand(B, NQ, tokens - PREFIX, device="cuda", generator=gen) < density
    return mask


def fp32_reference(qkv, go, mask, rope):
    qkv32 = qkv.detach().float().requires_grad_(True)
    q, k, v = (t.transpose(1, 2) for t in qkv32.reshape(B, qkv.shape[1], 3, H, D).unbind(2))
    if rope is not None:
        from common.ops import _apply_rope

        q, k = _apply_rope(q, k, rope)
    expanded = None if mask is None else mask.expand(-1, H, -1, -1)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=expanded)
    out = out.transpose(1, 2).reshape(B, qkv.shape[1], H * D)
    (grad,) = torch.autograd.grad(out, qkv32, go.float())
    return out.detach(), grad


def check_close(out, out_ref, grad, grad_ref, *, spiky):
    """bf16 vs fp32 reference.  The spiky case (large q/k plus huge-norm keys, logits of order 50-100)
    amplifies the bf16 rounding of q/k that every bf16 path shares, so its thresholds are looser."""
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    if spiky:
        assert cos(out, out_ref) > 0.999 and cos(grad, grad_ref) > 0.99
    else:
        assert cos(out, out_ref) > 0.9999 and rel_l2(out, out_ref) < 1.5e-2
        assert cos(grad, grad_ref) > 0.9995 and rel_l2(grad, grad_ref) < 3e-2


PLAIN_CASES = [AttnCfg(plain=p, rope=r) for p in ("auto", "flash", "cudnn", "eff") for r in ("torch", "triton")]
JOINT_CASES = [
    AttnCfg(joint="auto"), AttnCfg(joint="eff"), AttnCfg(joint="cudnn"),
    AttnCfg(joint="eff", bias=True), AttnCfg(joint="cudnn", bias=True),
    AttnCfg(plain="flash", joint="eff", qsplit=True), AttnCfg(plain="flash", joint="cudnn", qsplit=True),
    AttnCfg(plain="flash", joint="cudnn", qsplit=True, bias=True, rope="triton"),
    AttnCfg(plain="flash", joint="eff", qsplit=True, bias=True, rope="triton"),
]


@pytest.mark.parametrize("cfg", PLAIN_CASES, ids=str)
@pytest.mark.parametrize("spiky", [False, True])
def test_plain_attention_vs_fp32(cfg, spiky):
    qkv, go = inputs(1029, spiky=spiky)
    rope = rope_tables()
    out_ref, grad_ref = fp32_reference(qkv, go, None, rope)
    core = make_attn_core(cfg, fallback=None)
    out = core(SimpleNamespace(), qkv, rope, None, H)
    (grad,) = torch.autograd.grad(out, qkv, go)
    check_close(out, out_ref, grad, grad_ref, spiky=spiky)


@pytest.mark.parametrize("cfg", JOINT_CASES, ids=str)
@pytest.mark.parametrize("density,spiky", [(0.0, False), (0.5, False), (1.0, False), (0.5, True)])
def test_joint_masked_attention_vs_fp32(cfg, density, spiky):
    qkv, go = inputs(1129, spiky=spiky)
    mask = masked(1129, density)
    rope = rope_tables()
    out_ref, grad_ref = fp32_reference(qkv, go, mask, rope)
    core = make_attn_core(cfg, fallback=None)
    out = core(SimpleNamespace(), qkv, rope, mask, H)
    (grad,) = torch.autograd.grad(out, qkv, go)
    check_close(out, out_ref, grad, grad_ref, spiky=spiky)


def test_bool3_source_layout_accepted():
    qkv, go = inputs(1129, spiky=False)
    mask = masked(1129, 0.5)
    ref = make_attn_core(AttnCfg(), fallback=None)(SimpleNamespace(), qkv, None, mask, H)
    out = make_attn_core(AttnCfg(plain="flash", joint="cudnn", qsplit=True), fallback=None)(SimpleNamespace(), qkv, None, mask[:, 0], H)
    assert cos(out, ref) > 0.9999


def test_default_ops_attn_core_agrees_in_bf16():
    qkv, go = inputs(1129, spiky=False)
    mask = masked(1129, 0.5)
    out_default = default_ops().attn_core(SimpleNamespace(), qkv, None, mask[:, 0], H)
    out = make_attn_core(AttnCfg(plain="flash", joint="cudnn", qsplit=True), fallback=None)(SimpleNamespace(), qkv, None, mask, H)
    assert cos(out, out_default) > 0.9999


def test_autocast_loss_wrapper_runs_fp32_without_autocast():
    from round11_bf16_mixed.impl import patch as r11

    seen = {}

    def base_loss(c, mask_logits, class_logits, targets):
        seen["dtype"] = mask_logits[0].dtype
        seen["autocast"] = torch.is_autocast_enabled("cuda")
        seen["matmul_dtype"] = (mask_logits[0] @ mask_logits[0].transpose(-1, -2)).dtype
        return torch.zeros((), device="cuda"), {}

    ctx = SimpleNamespace(cfg=SimpleNamespace(precision="bf16-mixed", patches=("round11_bf16_mixed",)), meta={}, ops=default_ops(), model=SimpleNamespace(num_queries=NQ))
    ctx.ops.loss_all_layers = base_loss
    r11.apply(ctx, variant="autocast")
    logits = torch.randn(2, 4, 4, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ctx.ops.loss_all_layers(ctx, [logits], [logits], None)
    assert seen == {"dtype": torch.float32, "autocast": False, "matmul_dtype": torch.float32}


def test_head_ln_bf16_matches_autocast_head():
    from lightly_train._task_models.dinov3_eomt_semantic_segmentation.scale_block import ScaleBlock
    from round11_bf16_mixed.impl.head import install_head_cl, install_ln2d_bf16

    torch.manual_seed(0)
    base = torch.nn.Sequential(ScaleBlock(64), ScaleBlock(64)).cuda()
    for module in base.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.normal_(module.weight, 1.0, 0.1)
            torch.nn.init.normal_(module.bias, 0.0, 0.1)
    x = torch.randn(2, 64, 8, 8, device="cuda", requires_grad=True)
    variants = {}
    for name, installs in {"ln_bf16": [install_ln2d_bf16], "cl+ln_bf16": [install_head_cl, install_ln2d_bf16]}.items():
        upscale = copy.deepcopy(base)
        for install in installs:
            install(SimpleNamespace(model=SimpleNamespace(upscale=upscale)))
        variants[name] = upscale
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = base(x)
    (g_ref,) = torch.autograd.grad(ref.float().square().sum(), x)
    ref32 = copy.deepcopy(base)(x)
    for name, upscale in variants.items():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = upscale(x)
        (g,) = torch.autograd.grad(out.float().square().sum(), x)
        assert out.dtype == torch.bfloat16
        assert cos(out, ref32) > 0.999 and cos(out, ref) > 0.9995, name
        assert cos(g, g_ref) > 0.999, name
