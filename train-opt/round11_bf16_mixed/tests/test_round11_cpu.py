"""CPU tests for the round11 patch protocol and the torch attention/head building blocks."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from common.ops import _apply_rope, default_ops
from round11_bf16_mixed.impl import patch as r11
from round11_bf16_mixed.impl.attn import AttnCfg, attention, make_attn_core, normalize_mask, to_bias

NQ, STORAGE, PATCHES, HEADS, DIM = 3, 1, 8, 2, 8   # tokens N = NQ + 1 + STORAGE + PATCHES = 13
TOKENS = NQ + 1 + STORAGE + PATCHES
PREFIX = NQ + 1 + STORAGE


def make_ctx(patches=("round11_bf16_mixed",), precision="bf16-mixed"):
    return SimpleNamespace(
        cfg=SimpleNamespace(precision=precision, patches=tuple(patches)),
        meta={}, ops=default_ops(), model=SimpleNamespace(num_queries=NQ),
    )


def make_mask(batch=2, density=0.5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mask = torch.ones(batch, TOKENS, TOKENS, dtype=torch.bool)
    mask[:, :NQ, PREFIX:] = torch.rand(batch, NQ, PATCHES, generator=gen) < density
    return mask


def make_rope(seed=1):
    gen = torch.Generator().manual_seed(seed)
    angles = torch.rand(PATCHES, DIM // HEADS // 2, generator=gen) * 6.0
    angles = torch.cat((angles, angles), dim=-1)
    return torch.sin(angles), torch.cos(angles)


def make_qkv(batch=2, seed=2):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, TOKENS, 3 * DIM, generator=gen).requires_grad_(True)


# ----------------------------------------------------------------------------- variant parsing
def test_parse_variant_rules():
    assert r11.parse_variant("autocast") == ["autocast"]
    assert r11.parse_variant("autocast+rope_triton+plain_flash") == ["autocast", "rope_triton", "plain_flash"]
    assert r11.parse_variant("autocast+autocast") == ["autocast"]
    with pytest.raises(ValueError, match="unknown"):
        r11.parse_variant("autocast+nope")
    with pytest.raises(ValueError, match="requires"):
        r11.parse_variant("plain_flash")
    with pytest.raises(ValueError, match="conflicts"):
        r11.parse_variant("autocast+plain_flash+plain_cudnn")
    with pytest.raises(ValueError, match="conflicts"):
        r11.parse_variant("autocast+joint_eff+joint_cudnn")
    with pytest.raises(ValueError, match="conflicts"):
        r11.parse_variant("autocast+compile_block+rope_triton")
    for variant in r11.VARIANTS:  # every variant parses together with its own requirements
        parts = [variant, *r11.VARIANT_REQUIRES.get(variant, ())]
        assert variant in r11.parse_variant("+".join(parts))


def test_attn_cfg_from_variants():
    cfg = r11.attn_cfg_from(r11.parse_variant("autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+joint_bias"), 100)
    assert (cfg.rope, cfg.plain, cfg.joint, cfg.bias, cfg.qsplit, cfg.nq) == ("triton", "flash", "cudnn", True, True, 100)
    cfg = r11.attn_cfg_from(r11.parse_variant("autocast+plain_eff+joint_eff"), 7)
    assert (cfg.rope, cfg.plain, cfg.joint, cfg.bias, cfg.qsplit) == ("torch", "eff", "eff", False, False)
    assert r11.attn_cfg_from(["attn_core"], 100) == AttnCfg()
    with pytest.raises(ValueError):
        AttnCfg(joint="flash")
    with pytest.raises(ValueError):
        AttnCfg(plain="math")


# ----------------------------------------------------------------------------- masks and attention
def test_to_bias_values_and_alignment():
    mask = make_mask()[:, None, :NQ, :]
    bias = to_bias(mask, torch.float32)
    assert bias.shape == mask.shape
    assert bias.stride(-2) % 8 == 0 and bias.stride(-1) == 1
    assert torch.equal(bias == 0, mask)
    assert torch.equal(torch.isinf(bias), ~mask)


def test_normalize_mask_layouts():
    mask = make_mask()
    assert normalize_mask(mask).shape == (2, 1, TOKENS, TOKENS)
    assert normalize_mask(mask[:, None]) is not None
    with pytest.raises(TypeError):
        normalize_mask(mask.float())


@pytest.mark.parametrize("cfg", [
    AttnCfg(bias=True),
    AttnCfg(qsplit=True, nq=NQ),
    AttnCfg(qsplit=True, bias=True, nq=NQ),
])
@pytest.mark.parametrize("layout", ["bool3", "bool4"])
def test_attention_variants_equal_reference(cfg, layout):
    qkv = make_qkv()
    mask = make_mask()
    if layout == "bool4":
        mask = mask[:, None]
    ref_core = default_ops().attn_core
    ctx = make_ctx()
    ref_mask = mask[:, 0] if layout == "bool4" else mask
    out_ref = ref_core(ctx, qkv, None, ref_mask, HEADS)
    (grad_ref,) = torch.autograd.grad(out_ref, qkv, torch.ones_like(out_ref) * 0.3)
    core = make_attn_core(cfg, fallback=ref_core)
    out = core(ctx, qkv, None, mask, HEADS)
    (grad,) = torch.autograd.grad(out, qkv, torch.ones_like(out) * 0.3)
    assert torch.allclose(out, out_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad, grad_ref, atol=1e-6, rtol=1e-5)


def test_attn_core_with_rope_matches_default_and_no_mask():
    qkv = make_qkv()
    rope = make_rope()
    ctx = make_ctx()
    ref = default_ops().attn_core(ctx, qkv, rope, None, HEADS)
    out = make_attn_core(AttnCfg(), fallback=None)(ctx, qkv, rope, None, HEADS)
    assert torch.equal(out, ref)
    # rope_triton falls back to the torch rotation on CPU tensors and must stay identical
    out = make_attn_core(AttnCfg(rope="triton"), fallback=None)(ctx, qkv, rope, None, HEADS)
    assert torch.equal(out, ref)
    q, k = _apply_rope(qkv.detach().reshape(2, TOKENS, 3, HEADS, DIM // HEADS)[:, :, 0].transpose(1, 2), qkv.detach().reshape(2, TOKENS, 3, HEADS, DIM // HEADS)[:, :, 1].transpose(1, 2), rope)
    assert q.shape == (2, HEADS, TOKENS, DIM // HEADS)


def test_unknown_mask_type_falls_back():
    calls = []

    def fallback(ctx, qkv, rope, mask, heads):
        calls.append(type(mask).__name__)
        return "fallback"

    class QueryBitmask:  # stand-in for round12's mask object
        pass

    core = make_attn_core(AttnCfg(), fallback=fallback)
    assert core(make_ctx(), make_qkv(), None, QueryBitmask(), HEADS) == "fallback"
    assert calls == ["QueryBitmask"]


def test_qsplit_only_changes_masked_rows_semantics():
    """The mask structure the split relies on: rows >= nq are all True (checked on the source builder)."""
    mask = make_mask()
    assert bool(mask[:, NQ:].all()) and bool(mask[:, :NQ, :PREFIX].all())
    q = torch.randn(2, HEADS, TOKENS, DIM // HEADS)
    k, v = torch.randn_like(q), torch.randn_like(q)
    full = attention(q, k, v, mask, AttnCfg())
    split = attention(q, k, v, mask, AttnCfg(qsplit=True, nq=NQ))
    assert torch.allclose(full, split, atol=1e-6, rtol=1e-5)


# ----------------------------------------------------------------------------- patch protocol
def test_apply_autocast_sets_flags_and_fp32_loss():
    ctx = make_ctx()
    seen = {}

    def base_loss(c, mask_logits, class_logits, targets):
        seen["dtypes"] = (mask_logits[0].dtype, class_logits[0].dtype)
        seen["autocast"] = torch.is_autocast_enabled("cuda")
        return torch.zeros(()), {}

    ctx.ops.loss_all_layers = base_loss
    result = r11.apply(ctx, variant="autocast")
    assert ctx.meta["enable_autocast"] is True
    assert result["ops"] == ["loss_all_layers"]
    with torch.autocast("cuda", dtype=torch.bfloat16):  # only the autocast *state* matters, no GPU work happens
        ctx.ops.loss_all_layers(ctx, [torch.zeros(1, dtype=torch.bfloat16)], [torch.zeros(1, dtype=torch.bfloat16)], None)
    assert seen["dtypes"] == (torch.float32, torch.float32)
    assert seen["autocast"] is False


def test_apply_autocast_rejects_non_mixed_precision_and_unknown_variant():
    with pytest.raises(ValueError, match="bf16-mixed"):
        r11.apply(make_ctx(precision="32-true"), variant="autocast")
    with pytest.raises(ValueError, match="unknown"):
        r11.apply(make_ctx(), variant="autocast+bogus")


def test_apply_attention_installs_attn_core_and_default_attn():
    ctx = make_ctx()
    original_attn = ctx.ops.attn
    ctx.ops.attn = lambda *a, **k: (_ for _ in ()).throw(AssertionError("r1-style attn wrapper must be replaced"))
    result = r11.apply(ctx, variant="autocast+plain_flash+joint_qsplit+joint_cudnn+joint_bias+rope_triton")
    assert "attn_core" in result["ops"]
    assert ctx.ops.attn.__name__ == original_attn.__name__ == "attn"
    cfg = ctx.meta["round11_attn_cfg"]
    assert (cfg.plain, cfg.joint, cfg.qsplit, cfg.bias, cfg.rope, cfg.nq) == ("flash", "cudnn", True, True, "triton", NQ)


def test_apply_order_and_conflicts():
    ctx = make_ctx(patches=("round11_bf16_mixed", "round3_loss_matcher"))
    with pytest.raises(RuntimeError, match="after round3_loss_matcher"):
        r11.apply(ctx, variant="autocast")
    ctx = make_ctx(patches=("round1_torch_cleanup", "round3_loss_matcher", "round11_bf16_mixed"))
    r11.apply(ctx, variant="autocast")  # correct order
    ctx = make_ctx(patches=("round4_attention", "round11_bf16_mixed"))
    with pytest.raises(RuntimeError, match="round4_attention"):
        r11.apply(ctx, variant="autocast+rope_triton")
    r11.apply(make_ctx(patches=("round4_attention", "round11_bf16_mixed")), variant="autocast")  # autocast alone is fine with r4
    # seeded semantics: coordinator users pass only the variant key
    r11.apply(make_ctx(), variant="autocast", ignored_option=1)


# ----------------------------------------------------------------------------- head and compile reference
def test_ln2d_bf16_cpu_path_equals_source():
    from lightly_train._task_models.dinov3_eomt_semantic_segmentation.scale_block import LayerNorm2D
    from round11_bf16_mixed.impl.head import install_ln2d_bf16

    module = LayerNorm2D(16)
    torch.nn.init.normal_(module.weight)
    x = torch.randn(2, 16, 5, 5)
    expected = module(x)
    ctx = SimpleNamespace(model=SimpleNamespace(upscale=torch.nn.Sequential(module)))
    assert install_ln2d_bf16(ctx) == 1
    assert torch.equal(module(x), expected)


def test_compile_reference_block_matches_default_block():
    from lightly_train._models.dinov3.dinov3_src.layers.block import SelfAttentionBlock
    from round11_bf16_mixed.impl.compile_ref import _block_forward

    blk = SelfAttentionBlock(dim=DIM, num_heads=HEADS, ffn_ratio=2.0, qkv_bias=True, init_values=0.1)
    torch.nn.init.normal_(blk.ls1.gamma)
    torch.nn.init.normal_(blk.ls2.gamma)
    x = torch.randn(2, TOKENS, DIM)
    rope = make_rope()
    ctx = make_ctx()
    for mask in (None, make_mask()):
        expected = ctx.ops.block(ctx, 0, blk, x, rope, mask)
        actual = _block_forward(blk, x, rope[0], rope[1], mask)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_bf16_shadow_protocol_cpu():
    """Shadow Linear on the fp32 path equals the source Linear; the post-optimizer hook refreshes the shadows."""
    from round11_bf16_mixed.impl.shadow import install_bf16_shadow

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv = torch.nn.Linear(4, 12)
            self.attn.proj = torch.nn.Linear(4, 4)
            self.mlp = torch.nn.Module()
            self.mlp.fc1 = torch.nn.Linear(4, 8)
            self.mlp.fc2 = torch.nn.Linear(8, 4)

    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.blocks = torch.nn.ModuleList([Block(), Block()])
    ctx = SimpleNamespace(model=model, shadow={}, post_opt_hooks=[], meta={})
    x = torch.randn(3, 4)
    expected = model.backbone.blocks[0].attn.qkv(x)
    assert install_bf16_shadow(ctx) == 8
    assert torch.equal(model.backbone.blocks[0].attn.qkv(x), expected)  # no autocast -> master F.linear
    assert len(ctx.shadow) == 8 and all(v.dtype == torch.bfloat16 for v in ctx.shadow.values())
    with torch.no_grad():
        model.backbone.blocks[0].mlp.fc1.weight.add_(1.0)
    ctx.post_opt_hooks[0](ctx)
    name = "backbone.blocks.0.mlp.fc1.weight"
    assert torch.equal(ctx.shadow[name], model.backbone.blocks[0].mlp.fc1.weight.detach().to(torch.bfloat16))
    ctx.meta["shadow_writer"] = "round7_optimizer"
    with torch.no_grad():
        model.backbone.blocks[0].mlp.fc1.weight.add_(1.0)
    ctx.post_opt_hooks[0](ctx)
    assert not torch.equal(ctx.shadow[name], model.backbone.blocks[0].mlp.fc1.weight.detach().to(torch.bfloat16))
