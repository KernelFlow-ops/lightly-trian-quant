"""Pluggable model implementations for the Round 9 pipeline.

Contract (capture-safe, no host sync, static shapes):
    model_fn(static_input: bf16 [B,3,512,512]) -> (mask_logits_lr [B,Q,128,128], class_logits [B,Q,3])
Both outputs must have the same dtype (bf16 / fp16 / fp32) and are consumed by the fused post kernels.

    make_model_fn(name, model, impl="lightly", fp8=False)

impl:
  lightly : lightly eager forward_train under bf16 autocast + Round-7 fused head (P3)
            (fp8=True adds real FP8 rowwise GEMMs on the Round-6 recommended units)
  opt     : graphopt Round-1 OptEoMT (bf16-resident folded weights, fp32 residual)   [read-only import]
  opt_gemmhead : OptEoMT with head_convt_as_gemm
  opt_tl / opt_tl_rope : OptEoMT with Round-4 TileLang attention (plain / fused RoPE)
  fused   : blockfuse Round-3 fused_block (TileLang GEMMs, fused LN/rope/epilogues) + Round-7 fused head
  fused_tl: fused block with Round-4 TileLang attention on the Round-3 QKV layout
  r5_v1  : Round-5 tuned GEMM tiles and LayerNorm kernels + Round-7 fused head
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "round7_head_postprocess" / "impl", ROOT / "round6_low_precision" / "impl", ROOT / "common"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import head as hd  # noqa: E402

FP8_UNITS = {
    "vitl16": ROOT / "round6_low_precision" / "units_vitl16_n6.json",
    "vitb16": ROOT / "round6_low_precision" / "units_vitb16_n14.json",
}
IMPLS = ("lightly", "opt", "opt_gemmhead", "opt_tl", "opt_tl_rope", "fused", "fused_tl", "r5_v1")


def _lightly(model, fp8_units):
    hd.enable(model)
    if fp8_units:
        import fp8linear as fl

        fl.enable(model, fp8_units)

    def model_fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            ml, cl = model.forward_train(x, return_logits_per_layer=False)
        return ml[-1], cl[-1]

    return model_fn


def _opt(model, **kw):
    p = str(ROOT / "round1_torch_cleanup")
    if p not in sys.path:
        sys.path.insert(0, p)
    from opt_model import OptEoMT

    o = OptEoMT(model, **kw)

    def model_fn(x):
        return o.forward_lowres(x)

    model_fn._keepalive = o
    return model_fn


def _fused(model, tl_attention: bool = False):
    p = str(ROOT / "round3_fusion")
    if p not in sys.path:
        sys.path.insert(0, p)
    from impl import ops  # blockfuse

    if tl_attention:
        p4 = str(ROOT / "round4_attention" / "impl")
        if p4 not in sys.path:
            sys.path.insert(0, p4)
        import ops as attn_ops  # Round-4 TileLang attention, distinct from impl.ops

        def block_fn(t, w, prefix):
            _, h = ops.add_layernorm(t, None, w["ln1_w"], w["ln1_b"], w["ln1_eps"])
            qkv = ops.qkv_rope(h, w["Wqkv"], w["bqkv"], sin, cos, prefix, heads)
            # qkv is a view of contiguous [3,B,N,H,64] storage.
            attended = attn_ops.attention_3bnhd(qkv.transpose(2, 3))
            t = ops.gemm_bias_residual(attended, w["Wproj"], w["bproj"], t, w["g1"])
            _, h = ops.add_layernorm(t, None, w["ln2_w"], w["ln2_b"], w["ln2_eps"])
            a = ops.gemm_bias_gelu(h, w["W1"], w["b1"])
            return ops.gemm_bias_residual(a, w["W2"], w["b2"], t, w["g2"])
    else:
        def block_fn(t, w, prefix):
            return ops.fused_block(t, w, sin, cos, prefix, heads, False)

    hd.enable(model)
    bb = model.backbone
    heads = bb.blocks[0].attn.num_heads
    nb = len(bb.blocks)
    W = [ops.prepare_block_weights(b) for b in bb.blocks]
    grid = (512 // bb.patch_size, 512 // bb.patch_size)
    with torch.no_grad():
        sin, cos = bb.rope_embed(H=grid[0], W=grid[1])
    queries = model.queries.weight.detach().float()
    first_joint = nb - model.num_joint_blocks

    def model_fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            t, _ = bb.prepare_tokens_with_masks(x)
        t = t.float().contiguous()
        for i in range(nb):
            if i == first_joint:
                t = torch.cat((queries[None].expand(t.shape[0], -1, -1), t), dim=1).contiguous()
            t = block_fn(t, W[i], t.shape[1] - 1024)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            ml, cl = model._predict(bb.norm(t), grid_size=grid)
        return ml, cl

    return model_fn


def _r5_v1(name, model):
    p = str(ROOT / "round5_gemm_sm120")
    if p not in sys.path:
        sys.path.insert(0, p)
    from r5_model_impls import make_model_fn_r5

    return make_model_fn_r5(name, model, impl="r5_v1")


def make_model_fn(name: str, model, impl: str = "lightly", fp8: bool = False):
    if fp8 and impl != "lightly":
        raise ValueError("FP8 is only wired for impl=lightly")
    units = None
    if fp8:
        units = [tuple(u) for u in json.loads(FP8_UNITS[name].read_text())["chosen_units"]]
    if impl == "lightly":
        return _lightly(model, units)
    if impl == "opt":
        return _opt(model)
    if impl == "opt_gemmhead":
        return _opt(model, head_convt_as_gemm=True)
    if impl == "opt_tl":
        return _opt(model, attn_impl="tl")
    if impl == "opt_tl_rope":
        return _opt(model, attn_impl="tl_rope")
    if impl == "fused":
        return _fused(model)
    if impl == "fused_tl":
        return _fused(model, tl_attention=True)
    if impl == "r5_v1":
        return _r5_v1(name, model)
    raise ValueError(impl)
