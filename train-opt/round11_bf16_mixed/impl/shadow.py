"""bf16 shadow weights for the block Linears (qkv, proj, fc1, fc2), refreshed after the optimizer step.

Adapted from round5's shadow Linear (``_ShadowLinear``: bf16 GEMM forward/dgrad from the shadow copy, fp32
wgrad written straight from the GEMM), but parameter discovery uses ``ctx.model`` names
(``backbone.blocks.N...``): under Fabric ``ctx.train_model`` is wrapped and its names carry a prefix, so
round5's ``setup`` matches nothing in this harness.
"""
from __future__ import annotations

import re

import torch

SHADOW_RE = re.compile(r"backbone\.blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)\.weight")


def install_bf16_shadow(ctx) -> int:
    from round5_block_fusion.impl.shadow import _ShadowLinear

    matched = {name: p for name, p in ctx.model.named_parameters() if SHADOW_RE.fullmatch(name)}
    if not matched:
        raise RuntimeError("bf16_shadow: no block Linear weights matched")
    for name, p in matched.items():
        ctx.shadow.setdefault(name, p.detach().to(torch.bfloat16).contiguous())
    ids = {id(p): name for name, p in matched.items()}

    @torch.no_grad()
    def refresh(c) -> None:
        if c.meta.get("shadow_writer"):  # another patch (round7 capturable_shadow) refreshes the shadows
            return
        torch._foreach_copy_([c.shadow[name] for name in matched], list(matched.values()))

    ctx.post_opt_hooks.append(refresh)
    count = 0
    for block in ctx.model.backbone.blocks:
        for lin in (block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2):
            name = ids[id(lin.weight)]

            def forward(x: torch.Tensor, lin=lin, name=name) -> torch.Tensor:
                bias = lin.bias
                if bias is not None and hasattr(lin, "bias_mask"):
                    bias = bias * lin.bias_mask.to(bias.dtype)
                if not torch.is_autocast_enabled("cuda") or not x.is_cuda:
                    return torch.nn.functional.linear(x, lin.weight, bias)
                return _ShadowLinear.apply(x, lin.weight, bias, ctx.shadow[name])

            lin.forward = forward
            count += 1
    ctx.meta["round11_shadow_names"] = tuple(matched)
    return count
