"""Round20b FP4 head: ``ops.predict`` override.  Adapted from round14's ``impl/head.py`` (copied structure, r14 is untouched).

Same fused EoMT head as r14 (ConvT k2s2 as GEMM -> fused GELU / dwconv3x3 / LN2D kernels -> einsum with the mask embeddings); the only
change is that each of the two ConvT GEMMs picks the format of its f / d / w kind from the policy (``head.f / head.d / head.w``):
bf16 (r14's QuACK / cuBLAS GEMMs), mx8 (r17 quantizer + QuACK MXFP8), nv4 / mx4 (torch simulated quantization + the real QuACK FP4 GEMM, with
SR on dY and RHT along the pixel axis for the wgrad as options).  The weights are quantized per call from the bf16 ``Wm`` (as r14's mx8 path).
With an empty policy every GEMM is r14's, op for op.
"""
from __future__ import annotations

import torch

from round14_head_sm120.impl import head as h14
from round17_mxfp8_sm120.impl import gemm_mx8 as G8  # noqa: F401  (documented dependency)

from . import fp4_block as FB

ht = h14.ht


class Ops14:
    """bf16 ConvT backends = round14's (QuACK fwd / dgrad, cuBLAS fp32-out wgrad, per h14.POLICY)."""

    def fwd_bias(self, x, w, b):
        return h14.mm_fwd(x, w)

    def dgrad(self, dz, w):
        return h14.mm_dgrad(dz, w)

    def wgrad(self, dz, x):
        return h14.mm_wgrad(dz, x)


_OPS14 = Ops14()


class _ConvT4:
    """One ConvT(k2s2)-as-GEMM layer: Z = X Wm^T, dX = dZ Wm, dWm = dZ^T X with per-kind formats (see ``fp4_block.LayerQ``)."""

    def __init__(self, lq: FB.LayerQ, x, wm):
        self.lq, self.x, self.wm = lq, x, wm
        self.xq = self.wq = None

    def fwd(self):
        lq = self.lq
        self.wq = lq.prep_weight(self.wm) if lq.needs_weight() else None
        self.xq = lq.quant_x(self.x, None)
        z = lq.fwd_bias(self.xq, self.wq, None, self.wm)
        lq.after_fwd(self.xq)
        if lq.spec.fw == "bf16":
            self.xq.x_bf = self.x
        return z

    def bwd(self, dz):
        """-> (dWm fp32 [4C, C], dX bf16 [M, C])."""
        lq = self.lq
        dq = lq.quant_dy(dz, None)
        dwm = lq.wgrad(dq, self.xq)
        dx = lq.dgrad(dq, self.wq, self.wm)
        self.xq = self.wq = None
        return dwm, dx


class _Upscale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tok, e, w1, b1, d1, g1, be1, w2, b2, d2, g2, be2, eps1, eps2, grid, handle, nvtx=False):
        if nvtx:
            torch.cuda.nvtx.range_push("r20_up_fwd")
        ctx.nvtx = nvtx
        B, T, C = tok.shape
        h1 = grid
        h2 = 2 * grid
        call = handle.head_call
        handle.head_call += 1
        spec = handle.policy.spec("head")
        tok2d = tok.reshape(B * T, C)
        wm1, wm2 = h14._convt_to_gemm(w1), h14._convt_to_gemm(w2)
        dwt1, dwt2 = h14._dw_taps(d1), h14._dw_taps(d2)
        L1 = _ConvT4(FB.LayerQ(handle.sim, f"h{call}.c1", spec, _OPS14), tok2d, wm1)
        z1 = L1.fwd()
        y1, m1, r1 = ht.fwd(z1, b1, dwt1, g1, be1, B, h1, h1, eps1)
        L2 = _ConvT4(FB.LayerQ(handle.sim, f"h{call}.c2", spec, _OPS14), y1, wm2)
        z2 = L2.fwd()
        y2, m2, r2 = ht.fwd(z2, b2, dwt2, g2, be2, B, h2, h2, eps2)
        hw = (2 * h2) * (2 * h2)
        logits = h14.einsum_fwd(e, y2.view(B, hw, C))
        ctx.layers = (L1, L2)
        ctx.save_for_backward(tok, e, dwt1, dwt2, b1, b2, g1, g2, z1, m1, r1, z2, y2, m2, r2)
        ctx.dims = (B, C, h1, h2, hw, w1.dtype)
        if nvtx:
            torch.cuda.nvtx.range_pop()
        return logits.view(B, e.shape[1], 2 * h2, 2 * h2)

    @staticmethod
    def backward(ctx, glogits):
        if ctx.nvtx:
            torch.cuda.nvtx.range_push("r20_up_bwd")
        tok, e, dwt1, dwt2, b1, b2, g1, g2, z1, m1, r1, z2, y2, m2, r2 = ctx.saved_tensors
        L1, L2 = ctx.layers
        ctx.layers = None
        B, C, h1, h2, hw, _ = ctx.dims
        dl = glogits.reshape(B, e.shape[1], hw).to(torch.bfloat16)
        dy2, de = h14.einsum_bwd(dl, e, y2.view(B, hw, C))
        du2, dg2, dbe2 = ht.bwd1(z2, b2, dwt2, g2, m2, r2, dy2.view(-1, C), B, h2, h2)
        dz2, ddw2, db2 = ht.bwd2(z2, b2, dwt2, du2, B, h2, h2)
        dwm2, dy1 = L2.bwd(dz2)
        du1, dg1, dbe1 = ht.bwd1(z1, b1, dwt1, g1, m1, r1, dy1, B, h1, h1)
        dz1, ddw1, db1 = ht.bwd2(z1, b1, dwt1, du1, B, h1, h1)
        dwm1, dtok = L1.bwd(dz1)
        dtok = dtok.view(tok.shape)
        dw2c = h14._gemm_grad_to_convt(dwm2, C, C)
        dw1c = h14._gemm_grad_to_convt(dwm1, C, C)
        if ctx.nvtx:
            torch.cuda.nvtx.range_pop()
        return (dtok, de, dw1c, db1, ddw1.t().reshape(C, 1, 3, 3), dg1, dbe1,
                dw2c, db2, ddw2.t().reshape(C, 1, 3, 3), dg2, dbe2, None, None, None, None, None)


def upscale(model, tok, e, handle, nvtx=False):
    s1, s2 = model.upscale[0], model.upscale[1]
    return _Upscale.apply(
        tok, e,
        s1.conv1.weight, s1.conv1.bias, s1.conv2.weight, s1.norm.weight, s1.norm.bias,
        s2.conv1.weight, s2.conv1.bias, s2.conv2.weight, s2.norm.weight, s2.norm.bias,
        float(s1.norm.eps), float(s2.norm.eps), 32, handle, nvtx,
    )


eligible = h14.eligible


def predict(ctx, handle, x, grid_size):
    model = ctx.train_model.model
    xn = model.backbone.norm(x)
    q = xn[:, : model.num_queries]
    class_logits = model.class_head(q)
    e = model.mask_head(q)
    prefix = model.num_queries + 1 + model.backbone.n_storage_tokens
    tok = xn[:, prefix:].to(torch.bfloat16).contiguous()
    mask_logits = upscale(model, tok, e.to(torch.bfloat16), handle, bool(getattr(ctx, "nvtx", False)))
    return mask_logits, class_logits.float()
