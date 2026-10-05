"""Round17 MXFP8 training block: round13's fused block with the Linear GEMMs of a chosen subset of (blocks, linears) in block-scaled
MXFP8 (E4M3 + UE8M0 1x32, fp32 accumulation).  Everything else (LN, residual stream, LayerScale, GELU, attention, bias / gamma
gradients) is round13's code unchanged, so the semantics outside the GEMM operands are identical to TB1+R13.

Per MX8 Linear (``Y = X W^T``; X [M,K], W [N,K], M = B*N_tok tokens):

    forward : X is quantized ONCE into a row copy (K-major, consumed by this GEMM) and a column copy (blocks along tokens, kept for the
              wgrad instead of the bf16 X); W comes from ``Mx8Weights`` (row + column copies, refreshed once per step at block 0).
    backward: dY is quantized once into row (dgrad) and column (wgrad) copies; dX = dY W (N-major W copy), dW = dY^T X (fp32, straight
              into the Parameter gradient; the contraction runs over the zero-padded token axis Mpad = ceil(M/128)*128, pad rows are zero).

Quantization is NOT fused into the producers here (separate ``quant_dual`` launches); the producers' epilogues are round13's.
"""
from __future__ import annotations

import torch

from round13_gemm_fusion_sm120.impl import block_kernels as bk
from round13_gemm_fusion_sm120.impl import gemm_ops as ops13
from round13_gemm_fusion_sm120.impl.fused_block import (  # round13 building blocks, reused verbatim
    Shadows, _Bundle, _FC1, _FC2Res, _Linear, _LN1, _ProjRes, _need, eligible,
)

from . import gemm_mx8 as G
from . import mx8

LINEARS = ("qkv", "proj", "fc1", "fc2")


# ------------------------------------------------------------------------------------------------ policy
class Policy:
    """Which Linears of which blocks run in MXFP8.  Parsed from the ``+``-joined variant tokens::

        qkv proj fc1 fc2   add one Linear      fc (=fc1+fc2)   attn (=qkv+proj)   all (=all four)
        nofirst nolast nojoint    keep block 0 / the last block / the joint blocks (last three) in bf16
        blocks=a-b         MXFP8 only in blocks a..b (inclusive)
    """

    def __init__(self, parts):
        lin, self.nofirst, self.nolast, self.nojoint, self.rng = set(), False, False, False, None
        self.qkv_kinds = "fdw"  # which of the qkv GEMMs (f = fwd, d = dgrad, w = wgrad) run in MXFP8 when qkv is in the policy
        for p in parts:
            if p.startswith("qkv_"):  # qkv_f / qkv_fd / qkv_dw ...: qkv with only the listed GEMM kinds in MXFP8
                kinds = p[4:]
                assert kinds and set(kinds) <= set("fdw"), p
                self.qkv_kinds = "".join(k for k in "fdw" if k in kinds)
                lin.add("qkv")
            elif p in LINEARS:
                lin.add(p)
            elif p in ("fc", "mlp"):
                lin |= {"fc1", "fc2"}
            elif p == "attn":
                lin |= {"qkv", "proj"}
            elif p == "all":
                lin |= set(LINEARS)
            elif p == "nofirst":
                self.nofirst = True
            elif p == "nolast":
                self.nolast = True
            elif p == "nojoint":
                self.nojoint = True
            elif p.startswith("blocks="):
                a, _, b = p[7:].partition("-")
                self.rng = (int(a), int(b or a))
            else:
                raise ValueError(f"unknown round17 token {p!r}")
        self.lin = lin or set(LINEARS)

    def linears(self, i: int, nblocks: int) -> frozenset:
        if (self.nofirst and i == 0) or (self.nolast and i == nblocks - 1) or (self.nojoint and i >= nblocks - 3):
            return frozenset()
        if self.rng is not None and not (self.rng[0] <= i <= self.rng[1]):
            return frozenset()
        return frozenset(self.lin)

    def describe(self) -> str:
        return f"lin={sorted(self.lin)} qkv_kinds={self.qkv_kinds} nofirst={self.nofirst} nolast={self.nolast} nojoint={self.nojoint} blocks={self.rng}"


# ------------------------------------------------------------------------------------------------ weights
class Mx8Weights:
    """Per-step MXFP8 copies (row + column) of the block Linear weights, quantized from the fp32 Parameters at block 0."""

    def __init__(self, ctx, policy: Policy):
        self.policy = policy
        blocks = ctx.model.backbone.blocks
        self.nblocks = len(blocks)
        self.params = {}  # (block, linear) -> Parameter
        for i, blk in enumerate(blocks):
            for name, lin in zip(LINEARS, (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2)):
                if name in policy.linears(i, self.nblocks):
                    self.params[(i, name)] = lin.weight
        self.cur: dict = {}

    @torch.no_grad()
    def refresh(self):
        self.cur = {k: mx8.quantize(p.detach()) for k, p in self.params.items()}

    def get(self, i, name):
        return self.cur.get((i, name))


# ------------------------------------------------------------------------------------------------ Functions
class _MXLinear(torch.autograd.Function):
    """qkv: ``y = x @ W^T + b`` (bf16 out), fp32 wgrad, fused bias gradient column sum.  ``kinds`` selects which of the three GEMMs
    (f = fwd, d = dgrad, w = wgrad) run in MXFP8; the others use round13's bf16 GEMMs on the bf16 shadow weight."""

    @staticmethod
    def forward(ctx, x, weight, wq, shadow, bias, kinds):
        f, d, w = "f" in kinds, "d" in kinds, "w" in kinds
        with torch.autocast("cuda", enabled=False):
            xq = mx8.quantize(x, row=f, col=w) if (f or w) else None
            y = G.fwd_bias(xq, wq, bias) if f else ops13.linear_bias(x, shadow, bias, "qkv")
        if xq is not None:
            xq.qr = xq.sfr = None  # the row copy is dead after the forward GEMM; keep the column copy for the wgrad
        ctx.xq, ctx.wq, ctx.kinds = xq, wq, kinds
        ctx.x = None if w else x
        ctx.shadow = shadow
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        kinds = ctx.kinds
        f, d, w = "f" in kinds, "d" in kinds, "w" in kinds
        dyq = mx8.quantize(dy, row=d, col=w) if (d or w) else None
        dx = (G.dgrad(dyq, ctx.wq) if d else ops13.dgrad(dy, ctx.shadow, "qkv")) if _need(ctx, 0) else None
        dw = (G.wgrad(dyq, ctx.xq) if w else ops13.wgrad(dy, ctx.x, "qkv")) if _need(ctx, 1) else None
        db = bk.colsum_bf16(dy) if (ctx.has_bias and _need(ctx, 4)) else None
        ctx.xq = ctx.wq = ctx.x = ctx.shadow = None
        return dx, dw, None, None, db, None


class _MXProjRes(torch.autograd.Function):
    """``x1 = xr + gamma1 * (o @ Wp^T + bp)`` (epilogue) followed by ``ln2 = LN2(x1)`` (round13 LN kernel)."""

    @staticmethod
    def forward(ctx, o, xr, weight, wq, bias, gamma, ln_w, ln_b, eps):
        with torch.autocast("cuda", enabled=False):
            oq = mx8.quantize(o)
            x1, y1 = G.fwd_res(oq, wq, bias, gamma, xr)
            ln2, mean, rstd = bk.ln_fwd(x1, ln_w, ln_b, eps)
        oq.qr = oq.sfr = None
        ctx.oq, ctx.wq = oq, wq
        ctx.save_for_backward(y1, x1, mean, rstd, gamma, ln_w)
        ctx.set_materialize_grads(False)
        return x1, ln2

    @staticmethod
    def backward(ctx, g_x1, d_ln2):
        y1, x1, mean, rstd, gamma, ln_w = ctx.saved_tensors
        if g_x1 is None:
            g_x1 = torch.zeros_like(x1)
        if d_ln2 is None:
            d_ln2 = torch.zeros(x1.shape, device=x1.device, dtype=torch.bfloat16)
        dx1, g, sums = bk.ln_bwd_res(x1, mean, rstd, ln_w, d_ln2.contiguous(), g_x1.contiguous(), y=y1, gamma=gamma)
        gq = mx8.quantize(g)
        do = G.dgrad(gq, ctx.wq) if _need(ctx, 0) else None
        dw = G.wgrad(gq, ctx.oq) if _need(ctx, 2) else None
        ctx.oq = ctx.wq = None
        # grads: o, xr, weight, wq, bias, gamma, ln_w, ln_b, eps
        return do, dx1, dw, None, sums[3], sums[2], sums[0], sums[1], None


class _MXFC1(torch.autograd.Function):
    """``z = ln2 @ W1^T + b1``, ``h = GELU_erf(z)`` (epilogue); same protocol as round13 ``_FC1`` (z is a non-differentiable carrier;
    the gradient arriving for h is already the pre-activation gradient dz)."""

    @staticmethod
    def forward(ctx, ln2, weight, wq, bias, bundle):
        with torch.autocast("cuda", enabled=False):
            xq = mx8.quantize(ln2)
            z, h = G.fwd_gelu(xq, wq, bias)
        xq.qr = xq.sfr = None
        ctx.xq, ctx.wq, ctx.bundle = xq, wq, bundle
        ctx.mark_non_differentiable(z)
        ctx.set_materialize_grads(False)
        return h, z

    @staticmethod
    def backward(ctx, dz, _unused):
        dzq = mx8.quantize(dz.contiguous())
        dln2 = G.dgrad(dzq, ctx.wq) if _need(ctx, 0) else None
        dw = G.wgrad(dzq, ctx.xq) if _need(ctx, 1) else None
        db = ctx.bundle.db1
        ctx.bundle.db1 = None
        ctx.xq = ctx.wq = None
        return dln2, dw, None, db, None


class _MXFC2Res(torch.autograd.Function):
    """``x2 = x1 + gamma2 * (h @ W2^T + b2)`` (epilogue); backward applies GELU' in the dgrad epilogue (round13 protocol)."""

    @staticmethod
    def forward(ctx, h, z, x1, weight, wq, bias, gamma, bundle):
        with torch.autocast("cuda", enabled=False):
            hq = mx8.quantize(h)
            x2, y2 = G.fwd_res(hq, wq, bias, gamma, x1)
        hq.qr = hq.sfr = None
        ctx.hq, ctx.wq, ctx.bundle = hq, wq, bundle
        ctx.save_for_backward(z, y2, gamma)
        return x2

    @staticmethod
    def backward(ctx, g2):
        z, y2, gamma = ctx.saved_tensors
        g2 = g2.contiguous()
        g, sums = bk.gamma_prep(g2, y2, gamma)  # dy2 = gamma2*g2 (bf16); sums = [dGamma2, db2]
        gq = mx8.quantize(g)
        dz, db1 = G.dgrad_dgelu(gq, ctx.wq, z)
        ctx.bundle.db1 = db1
        dw = G.wgrad(gq, ctx.hq) if _need(ctx, 3) else None
        ctx.hq = ctx.wq = None
        # grads: h, z, x1, weight, wq, bias, gamma, bundle
        return dz, None, g2, dw, None, sums[1], sums[0], None


# ------------------------------------------------------------------------------------------------ block
def make_block(ctx, previous, policy: Policy):
    shadows = Shadows(ctx)
    weights = Mx8Weights(ctx, policy)
    nblocks = weights.nblocks
    # bf16 shadows are only needed by the Linears that stay bf16 (an MXFP8 Linear is fed from ``weights``): refresh just those
    keep = [shadows.items[4 * bi + j] for bi in range(nblocks) for j, ln in enumerate(LINEARS)
            if ln not in policy.linears(bi, nblocks) or (ln == "qkv" and policy.qkv_kinds != "fdw")]

    @torch.no_grad()
    def refresh_shadows():
        if shadows.external or not keep:
            return
        torch._foreach_copy_([ctx.shadow[n] for n, _ in keep], [p.detach() for _, p in keep])

    def block(c, i, blk, x, rope, mask):
        reason = eligible(blk, x)
        if reason is not None:
            return previous(c, i, blk, x, rope, mask)
        if i == 0:
            refresh_shadows()
            weights.refresh()
        B, N, d = x.shape
        M = B * N
        attn = blk.attn
        xc = (x if x.dtype == torch.float32 else x.float()).reshape(M, d).contiguous()
        bundle = _Bundle()
        mxs = policy.linears(i, nblocks)

        qkv_lin = attn.qkv
        qkv_bias = qkv_lin.bias
        if qkv_bias is not None and hasattr(qkv_lin, "bias_mask"):  # LinearKMaskedBias (DINOv3 mask_k_bias)
            qkv_bias = qkv_bias * qkv_lin.bias_mask.to(qkv_bias.dtype)

        xr, ln1 = _LN1.apply(xc, blk.norm1.weight, blk.norm1.bias, blk.norm1.eps)
        if "qkv" in mxs:
            qkv = _MXLinear.apply(ln1, qkv_lin.weight, weights.get(i, "qkv"), shadows.get(qkv_lin.weight), qkv_bias, policy.qkv_kinds)
        else:
            qkv = _Linear.apply(ln1, qkv_lin.weight, shadows.get(qkv_lin.weight), qkv_bias, "qkv")
        o = c.ops.attn_core(c, qkv.view(B, N, 3 * d), rope, mask, attn.num_heads)
        o = o.reshape(M, d)
        if o.dtype != torch.bfloat16:
            o = o.to(torch.bfloat16)
        o = o.contiguous()
        if "proj" in mxs:
            x1, ln2 = _MXProjRes.apply(o, xr, attn.proj.weight, weights.get(i, "proj"), attn.proj.bias,
                                       blk.ls1.gamma, blk.norm2.weight, blk.norm2.bias, blk.norm2.eps)
        else:
            x1, ln2 = _ProjRes.apply(o, xr, attn.proj.weight, shadows.get(attn.proj.weight), attn.proj.bias,
                                     blk.ls1.gamma, blk.norm2.weight, blk.norm2.bias, blk.norm2.eps)
        if "fc1" in mxs:
            h, z = _MXFC1.apply(ln2, blk.mlp.fc1.weight, weights.get(i, "fc1"), blk.mlp.fc1.bias, bundle)
        else:
            h, z = _FC1.apply(ln2, blk.mlp.fc1.weight, shadows.get(blk.mlp.fc1.weight), blk.mlp.fc1.bias, bundle)
        if "fc2" in mxs:
            x2 = _MXFC2Res.apply(h, z, x1, blk.mlp.fc2.weight, weights.get(i, "fc2"), blk.mlp.fc2.bias, blk.ls2.gamma, bundle)
        else:
            x2 = _FC2Res.apply(h, z, x1, blk.mlp.fc2.weight, shadows.get(blk.mlp.fc2.weight), blk.mlp.fc2.bias, blk.ls2.gamma, bundle)
        return x2.view(B, N, d)

    block.shadows = shadows
    block.mx8_weights = weights
    block.policy = policy
    return block
