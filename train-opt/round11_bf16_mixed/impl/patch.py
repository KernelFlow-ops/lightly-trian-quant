"""round11_bf16_mixed: torch BF16 autocast mainline ("T-BF16").

Composable variants joined with ``+``, e.g. ``autocast+rope_triton+plain_flash+joint_qsplit``.

``autocast`` was seeded by the coordinator so that rounds 12-17 have a stable same-precision
reference from the start; its name and semantics must stay stable (see CONTRACTS.md): it turns on
Fabric bf16 autocast for the harness ``train_step`` and wraps ``ops.loss_all_layers`` so the loss and
matcher get fp32 logits and run with autocast disabled.  Every other variant is additive.

Variant            | what it does                                                      | requires   | conflicts
-------------------|-------------------------------------------------------------------|------------|----------------------
autocast           | bf16 autocast; fp32 loss                                          | -          | -
rope_triton        | attn_core: round4 fused packed-QKV RoPE (1 launch fwd, 1 bwd)      | -          | round4_attention
plain_flash        | attn_core: flash for mask-free attention                          | autocast   | plain_cudnn/eff
plain_cudnn        | attn_core: cuDNN SDPA for mask-free attention                     | autocast   | plain_flash/eff
plain_eff          | attn_core: mem-efficient SDPA for mask-free attention             | -          | plain_flash/cudnn
joint_cudnn        | attn_core: cuDNN SDPA for masked (joint) attention                | autocast   | joint_eff
joint_eff          | attn_core: mem-efficient SDPA for masked (joint) attention        | -          | joint_cudnn
joint_bias         | bool mask -> aligned additive bf16 bias before SDPA               | -          | -
joint_qsplit       | only the 100 query rows are masked: unmasked rows use the         | -          | -
                   | mask-free backend, the 100 masked rows use the joint backend      |            |
attn_core          | install the r11 attn_core with torch RoPE and default backends    | -          | round4_attention
head_cl            | channels_last head convolutions (round1 helper)                   | -          | -
head_ln_bf16       | LayerNorm2D in bf16 in/out instead of fp32 round trip             | autocast   | -
bf16_shadow        | bf16 shadow weights for block Linears (round5 Function), refreshed | autocast   | round7 capturable_shadow
                   | after the optimizer step; fp32 wgrad straight from the GEMM       |            |
cudnn_bench        | torch.backends.cudnn.benchmark = True                             | -          | -
compile_block      | torch.compile (default) per block (reference only)               | autocast   | attn tokens, shadow
compile_block_max  | torch.compile (max-autotune-no-cudagraphs) per block              | autocast   | attn tokens, shadow

Patch order: ``round11_bf16_mixed`` must come after ``round1_torch_cleanup`` and
``round3_loss_matcher`` (they replace ``ops.attn``/``ops.loss_all_layers``, which r11 overrides);
``apply`` raises if they come later.
"""
from __future__ import annotations

from typing import Any

import torch

REQUIRES: tuple[str, ...] = ()
CONFLICTS: tuple[str, ...] = ()

ATTN_VARIANTS = (
    "attn_core", "rope_triton", "plain_flash", "plain_cudnn", "plain_eff",
    "joint_cudnn", "joint_eff", "joint_bias", "joint_qsplit",
)
HEAD_VARIANTS = ("head_cl", "head_ln_bf16")
OTHER_VARIANTS = ("bf16_shadow", "cudnn_bench", "compile_block", "compile_block_max")
VARIANTS = ("autocast", *ATTN_VARIANTS, *HEAD_VARIANTS, *OTHER_VARIANTS)

VARIANT_REQUIRES: dict[str, tuple[str, ...]] = {
    "plain_flash": ("autocast",), "plain_cudnn": ("autocast",), "joint_cudnn": ("autocast",),
    "head_ln_bf16": ("autocast",), "bf16_shadow": ("autocast",),
    "compile_block": ("autocast",), "compile_block_max": ("autocast",),
}
VARIANT_CONFLICTS: dict[str, tuple[str, ...]] = {
    "plain_flash": ("plain_cudnn", "plain_eff"),
    "plain_cudnn": ("plain_flash", "plain_eff"),
    "plain_eff": ("plain_flash", "plain_cudnn"),
    "joint_cudnn": ("joint_eff",),
    "joint_eff": ("joint_cudnn",),
    "compile_block": ("compile_block_max", *ATTN_VARIANTS, "bf16_shadow"),
    "compile_block_max": ("compile_block", *ATTN_VARIANTS, "bf16_shadow"),
}
# patches whose ops this patch would silently bypass or be bypassed by
PATCH_CONFLICTS: dict[str, tuple[str, ...]] = {v: ("round4_attention",) for v in ATTN_VARIANTS}
MUST_PRECEDE = ("round1_torch_cleanup", "round3_loss_matcher")
NAME = "round11_bf16_mixed"


def parse_variant(variant: str) -> list[str]:
    parts = [part.strip() for part in variant.replace(",", "+").split("+") if part.strip()]
    unknown = [part for part in parts if part not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown round11 variants {unknown}; known {VARIANTS}")
    parts = list(dict.fromkeys(parts))
    for part in parts:
        missing = [req for req in VARIANT_REQUIRES.get(part, ()) if req not in parts]
        if missing:
            raise ValueError(f"round11 variant {part!r} requires {missing}")
        clash = [other for other in VARIANT_CONFLICTS.get(part, ()) if other in parts]
        if clash:
            raise ValueError(f"round11 variant {part!r} conflicts with {clash}")
    return parts


def attn_cfg_from(parts: list[str], num_queries: int):
    from round11_bf16_mixed.impl.attn import AttnCfg

    plain = next((p.split("_", 1)[1] for p in parts if p.startswith("plain_")), "auto")
    joint = next((p.split("_", 1)[1] for p in parts if p in ("joint_cudnn", "joint_eff")), "auto")
    return AttnCfg(
        rope="triton" if "rope_triton" in parts else "torch",
        plain=plain, joint=joint, bias="joint_bias" in parts, qsplit="joint_qsplit" in parts, nq=num_queries,
    )


def _check_order(ctx, parts: list[str]) -> None:
    patches = list(ctx.cfg.patches)
    me = patches.index(NAME) if NAME in patches else len(patches)
    for name in MUST_PRECEDE:
        if name in patches and patches.index(name) > me:
            raise RuntimeError(f"{NAME} must be applied after {name} (it replaces ops that {name} would otherwise overwrite)")
    for part in parts:
        for other in PATCH_CONFLICTS.get(part, ()):
            if other in patches:
                raise RuntimeError(f"round11 variant {part!r} conflicts with patch {other!r} (it overrides ops.attn and bypasses attn_core)")


def _install_autocast(ctx) -> list[str]:
    if ctx.cfg.precision != "bf16-mixed":
        raise ValueError("autocast requires precision='bf16-mixed'")
    ctx.meta["enable_autocast"] = True
    base_loss = ctx.ops.loss_all_layers

    def loss_all_layers(c, mask_logits_list, class_logits_list, binary_masks):
        # The loss/matcher stay fp32: fp32 inputs and no autocast inside.
        with torch.autocast("cuda", enabled=False):
            return base_loss(
                c,
                [logits.float() for logits in mask_logits_list],
                [logits.float() for logits in class_logits_list],
                binary_masks,
            )

    ctx.ops.loss_all_layers = loss_all_layers
    return ["loss_all_layers"]


def _install_attention(ctx, parts: list[str]) -> list[str]:
    from common.ops import default_ops
    from round11_bf16_mixed.impl.attn import make_attn_core

    cfg = attn_cfg_from(parts, ctx.model.num_queries)
    ctx.ops.attn_core = make_attn_core(cfg, fallback=ctx.ops.attn_core)
    # round1's mask_skip wraps ops.attn and handles masked blocks itself; route everything through
    # attn_core instead (default attn = qkv -> ops.attn_core -> proj).
    ctx.ops.attn = default_ops().attn
    ctx.meta["round11_attn_cfg"] = cfg
    return ["attn", "attn_core"]


def _install_compile(ctx, parts: list[str]) -> list[str]:
    from round11_bf16_mixed.impl.compile_ref import install_block_compile

    mode = "max-autotune-no-cudagraphs" if "compile_block_max" in parts else "default"
    install_block_compile(ctx, mode)
    return ["block"]


def apply(ctx, variant: str = "autocast", **kw) -> dict:
    # extra keyword arguments are ignored (same as the coordinator's seeded autocast-only version)
    parts = parse_variant(variant)
    _check_order(ctx, parts)
    changed: list[str] = []
    if "autocast" in parts:
        changed += _install_autocast(ctx)
    if any(part in ATTN_VARIANTS for part in parts):
        changed += _install_attention(ctx, parts)
    if "head_cl" in parts:
        from round11_bf16_mixed.impl.head import install_head_cl

        install_head_cl(ctx)
    if "head_ln_bf16" in parts:
        from round11_bf16_mixed.impl.head import install_ln2d_bf16

        install_ln2d_bf16(ctx)
    if "bf16_shadow" in parts:
        from round11_bf16_mixed.impl.shadow import install_bf16_shadow

        install_bf16_shadow(ctx)
    if "cudnn_bench" in parts:
        torch.backends.cudnn.benchmark = True
    if "compile_block" in parts or "compile_block_max" in parts:
        changed += _install_compile(ctx, parts)
    ctx.meta["round11_variants"] = tuple(parts)
    return {"ops": list(dict.fromkeys(changed)), "desc": f"round11: {variant}"}
