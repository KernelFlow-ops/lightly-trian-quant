"""Round13 fused BF16 training block: ``ops.block`` as a chain of ``torch.autograd.Function`` s.

Semantics are those of the source block (DINOv3 ``SelfAttentionBlock._forward`` without stochastic depth)::

    x1 = x  + gamma1 * proj(attn(LN1(x)))
    x2 = x1 + gamma2 * fc2(GELU_erf(fc1(LN2(x1))))

with the residual stream x fp32 and every GEMM operand bf16 (fp32 accumulation), Parameters fp32.

Forward chain (kernels, excluding attention):

    _LN1      x(fp32) -> ln1(bf16)                         1 CUDA kernel   (returns x as an alias output so that the
                                                                           residual-path gradient arrives in its backward)
    _Linear   qkv = ln1 @ Wqkv^T + b                       1 QuACK GEMM (bias epilogue)
    attn_core (ctx.ops.attn_core, owned by round11/12)     -
    _ProjRes  x1 = x + gamma1*(o @ Wp^T + bp); y1 saved   1 QuACK GEMM (bias+gamma+fp32 residual epilogue) + 1 LN2 kernel
    _FC1      z = ln2 @ W1^T + b1, h = GELU(z)             1 QuACK GEMM (bias + erf-GELU epilogue, z and h stored)
    _FC2Res   x2 = x1 + gamma2*(h @ W2^T + b2); y2 saved   1 QuACK GEMM (same epilogue as proj)

Backward chain (g = incoming fp32 gradient of the block output):

    _FC2Res.bwd  g -> dy2 = gamma2*g (bf16), dGamma2, db2 partials      1 kernel (+1 finalize)
                 dz = (dy2 @ W2) * GELU'(z), db1 = sum dz              1 GEMM, GELU'/bias-grad in the epilogue
                 dW2 = dy2^T h (fp32)                                  1 GEMM
    _FC1.bwd     dln2 = dz @ W1 ; dW1 = dz^T ln2 (fp32)                2 GEMMs
    _ProjRes.bwd LN2 bwd + residual add + dyp = gamma1*dx1 (bf16) + dGamma1, dbp, dLN2 partials   1 kernel (+1 finalize)
                 do = dyp @ Wp ; dWp = dyp^T o                         2 GEMMs
    attn_core.bwd (autograd)                                           -
    _Linear.bwd  dln1 = dqkv @ Wqkv ; dWqkv = dqkv^T ln1 ; db = colsum 2 GEMMs + 1 kernel (+1 finalize)
    _LN1.bwd     LN1 bwd + residual add (dx = dxr + LN_bwd(dln1))       1 kernel (+1 finalize)

Everything is fp32 where it matters (LN statistics, residual stream, gradient partial sums, all reductions are
fixed-order two-stage, bitwise deterministic).  The weights are consumed through bf16 shadow tensors
(``ctx.shadow[name]``), refreshed once per forward at block 0 unless another patch owns them
(``ctx.meta['shadow_writer']`` / round11 ``bf16_shadow``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import block_kernels as bk
from . import gemm_ops as ops

SUPPORTED_D = bk.SUPPORTED_D


def _need(ctx, idx):
    return ctx.needs_input_grad[idx]


# ------------------------------------------------------------------------------------------------ Functions
class _LN1(torch.autograd.Function):
    """LayerNorm of the fp32 residual stream with bf16 output; the input is returned as an alias output ``xr``."""

    @staticmethod
    def forward(ctx, x, w, b, eps):
        with torch.autocast("cuda", enabled=False):
            y, mean, rstd = bk.ln_fwd(x, w, b, eps)
        ctx.save_for_backward(x, mean, rstd, w)
        ctx.set_materialize_grads(False)
        return x, y

    @staticmethod
    def backward(ctx, dxr, dy):
        x, mean, rstd, w = ctx.saved_tensors
        if dxr is None:
            dxr = torch.zeros_like(x)
        if dy is None:
            dy = torch.zeros(x.shape, device=x.device, dtype=torch.bfloat16)
        dx, _, sums = bk.ln_bwd_res(x, mean, rstd, w, dy.contiguous(), dxr.contiguous())
        return dx, sums[0], sums[1], None


class _Linear(torch.autograd.Function):
    """``y = x @ W^T + b`` (bf16 out) with bf16 shadow weight, fp32 wgrad and fused bias gradient (qkv)."""

    @staticmethod
    def forward(ctx, x, weight, shadow, bias, name):
        with torch.autocast("cuda", enabled=False):
            y = ops.linear_bias(x, shadow, bias, name)
        ctx.save_for_backward(x, shadow)
        ctx.name = name
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x, shadow = ctx.saved_tensors
        dy = dy.contiguous()
        dx = ops.dgrad(dy, shadow, ctx.name) if _need(ctx, 0) else None
        dw = ops.wgrad(dy, x, ctx.name) if _need(ctx, 1) else None
        db = bk.colsum_bf16(dy) if (ctx.has_bias and _need(ctx, 3)) else None
        return dx, dw, None, db, None


class _ProjRes(torch.autograd.Function):
    """``x1 = xr + gamma1 * (o @ Wp^T + bp)`` (GEMM epilogue) followed by ``ln2 = LN2(x1)``."""

    @staticmethod
    def forward(ctx, o, xr, weight, shadow, bias, gamma, ln_w, ln_b, eps):
        with torch.autocast("cuda", enabled=False):
            x1, y1 = ops.linear_res(o, shadow, bias, gamma, xr, "proj")
            ln2, mean, rstd = bk.ln_fwd(x1, ln_w, ln_b, eps)
        ctx.save_for_backward(o, y1, x1, mean, rstd, shadow, gamma, ln_w)
        ctx.set_materialize_grads(False)
        return x1, ln2

    @staticmethod
    def backward(ctx, g_x1, d_ln2):
        o, y1, x1, mean, rstd, shadow, gamma, ln_w = ctx.saved_tensors
        if g_x1 is None:
            g_x1 = torch.zeros_like(x1)
        if d_ln2 is None:
            d_ln2 = torch.zeros(x1.shape, device=x1.device, dtype=torch.bfloat16)
        dx1, g, sums = bk.ln_bwd_res(x1, mean, rstd, ln_w, d_ln2.contiguous(), g_x1.contiguous(), y=y1, gamma=gamma)
        do = ops.dgrad(g, shadow, "proj") if _need(ctx, 0) else None
        dw = ops.wgrad(g, o, "proj") if _need(ctx, 2) else None
        # grads: o, xr, weight, shadow, bias, gamma, ln_w, ln_b, eps
        return do, dx1, dw, None, sums[3], sums[2], sums[0], sums[1], None


class _Bundle:
    """Per-block hand-over of the fc1 bias gradient from the fc2 dgrad epilogue to ``_FC1.backward``."""

    __slots__ = ("db1",)

    def __init__(self):
        self.db1 = None


class _FC1(torch.autograd.Function):
    """``z = ln2 @ W1^T + b1`` and ``h = GELU_erf(z)``; ``z`` is a non-differentiable carrier for ``_FC2Res``.

    Protocol: ``_FC2Res.backward`` applies GELU' (it owns ``z``), so the gradient arriving here for ``h`` is already
    the pre-activation gradient ``dz``.
    """

    @staticmethod
    def forward(ctx, ln2, weight, shadow, bias, bundle):
        with torch.autocast("cuda", enabled=False):
            z, h = ops.linear_gelu(ln2, shadow, bias)
        ctx.save_for_backward(ln2, shadow)
        ctx.bundle = bundle
        ctx.mark_non_differentiable(z)
        ctx.set_materialize_grads(False)
        return h, z

    @staticmethod
    def backward(ctx, dz, _unused):
        ln2, shadow = ctx.saved_tensors
        dz = dz.contiguous()
        dln2 = ops.dgrad(dz, shadow, "fc1") if _need(ctx, 0) else None
        dw = ops.wgrad(dz, ln2, "fc1") if _need(ctx, 1) else None
        db = ctx.bundle.db1
        ctx.bundle.db1 = None
        return dln2, dw, None, db, None


class _FC2Res(torch.autograd.Function):
    """``x2 = x1 + gamma2 * (h @ W2^T + b2)`` (GEMM epilogue)."""

    @staticmethod
    def forward(ctx, h, z, x1, weight, shadow, bias, gamma, bundle):
        with torch.autocast("cuda", enabled=False):
            x2, y2 = ops.linear_res(h, shadow, bias, gamma, x1, "fc2")
        ctx.save_for_backward(h, z, y2, shadow, gamma)
        ctx.bundle = bundle
        return x2

    @staticmethod
    def backward(ctx, g2):
        h, z, y2, shadow, gamma = ctx.saved_tensors
        g2 = g2.contiguous()
        g, sums = bk.gamma_prep(g2, y2, gamma)  # dy2 = gamma2*g2 (bf16); sums = [dGamma2, db2]
        dz, db1 = ops.dgrad_dgelu(g, shadow, z)
        ctx.bundle.db1 = db1
        dw = ops.wgrad(g, h, "fc2") if _need(ctx, 3) else None
        # grads: h, z, x1, weight, shadow, bias, gamma, bundle
        return dz, None, g2, dw, None, sums[1], sums[0], None


# ------------------------------------------------------------------------------------------------ shadows
class Shadows:
    """bf16 copies of the block Linear weights, keyed like round5/round11 (``backbone.blocks.N.attn.qkv.weight``)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.items: list[tuple[str, torch.nn.Parameter]] = []
        names = {id(p): n for n, p in ctx.model.named_parameters()}
        for blk in ctx.model.backbone.blocks:
            for lin in (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2):
                name = names[id(lin.weight)]
                ctx.shadow.setdefault(name, lin.weight.detach().to(torch.bfloat16).contiguous())
                self.items.append((name, lin.weight))
        self.names = {id(p): n for n, p in self.items}
        # round11 ``bf16_shadow`` (or the optimizer patch) maintains the shadows itself
        self.external = bool(ctx.meta.get("shadow_writer") or ctx.meta.get("round11_shadow_names"))

    @torch.no_grad()
    def refresh(self):
        if self.external:
            return
        torch._foreach_copy_([self.ctx.shadow[n] for n, _ in self.items], [p.detach() for _, p in self.items])

    def get(self, weight):
        return self.ctx.shadow[self.names[id(weight)]]


# ------------------------------------------------------------------------------------------------ block
def eligible(blk, x) -> str | None:
    """None if the fused path applies, else the reason for falling back to the previously installed block."""
    if not x.is_cuda:
        return "cpu tensor"
    d = x.shape[-1]
    if d not in SUPPORTED_D:
        return f"hidden size {d}"
    ls1, ls2 = getattr(blk, "ls1", None), getattr(blk, "ls2", None)
    if not (hasattr(ls1, "gamma") and hasattr(ls2, "gamma")):
        return "no LayerScale"
    if blk.training and getattr(blk, "sample_drop_ratio", 0.0) > 0.0:
        return "stochastic depth"
    attn, mlp = blk.attn, blk.mlp
    if not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2") and isinstance(mlp.act, nn.GELU) and mlp.act.approximate == "none"):
        return "mlp is not Linear-GELU(erf)-Linear"
    if attn.num_heads * 64 != d:
        return "head_dim != 64"
    if not (isinstance(blk.norm1, nn.LayerNorm) and isinstance(blk.norm2, nn.LayerNorm)):
        return "norm is not LayerNorm"
    for p in (attn.proj_drop, attn.attn_drop, mlp.drop):
        if getattr(p, "p", 0.0) != 0.0 and blk.training:
            return "dropout"
    return None


def make_block(ctx, previous):
    shadows = Shadows(ctx)

    def block(c, i, blk, x, rope, mask):
        reason = eligible(blk, x)
        if reason is not None:
            return previous(c, i, blk, x, rope, mask)
        if i == 0:
            shadows.refresh()
        B, N, d = x.shape
        M = B * N
        attn = blk.attn
        xc = (x if x.dtype == torch.float32 else x.float()).reshape(M, d).contiguous()
        bundle = _Bundle()

        qkv_lin = attn.qkv
        qkv_bias = qkv_lin.bias
        if qkv_bias is not None and hasattr(qkv_lin, "bias_mask"):  # LinearKMaskedBias (DINOv3 mask_k_bias)
            qkv_bias = qkv_bias * qkv_lin.bias_mask.to(qkv_bias.dtype)

        xr, ln1 = _LN1.apply(xc, blk.norm1.weight, blk.norm1.bias, blk.norm1.eps)
        qkv = _Linear.apply(ln1, qkv_lin.weight, shadows.get(qkv_lin.weight), qkv_bias, "qkv")
        o = c.ops.attn_core(c, qkv.view(B, N, 3 * d), rope, mask, attn.num_heads)
        o = o.reshape(M, d)
        if o.dtype != torch.bfloat16:
            o = o.to(torch.bfloat16)
        o = o.contiguous()
        x1, ln2 = _ProjRes.apply(o, xr, attn.proj.weight, shadows.get(attn.proj.weight), attn.proj.bias,
                                 blk.ls1.gamma, blk.norm2.weight, blk.norm2.bias, blk.norm2.eps)
        h, z = _FC1.apply(ln2, blk.mlp.fc1.weight, shadows.get(blk.mlp.fc1.weight), blk.mlp.fc1.bias, bundle)
        x2 = _FC2Res.apply(h, z, x1, blk.mlp.fc2.weight, shadows.get(blk.mlp.fc2.weight), blk.mlp.fc2.bias,
                           blk.ls2.gamma, bundle)
        return x2.view(B, N, d)

    block.shadows = shadows
    return block
