"""Round20b "half-real" FP4 training block: r17's fused block (``fused_block_mx8.py`` is the template) with the format of every Linear GEMM kind (f / d / w)
chosen independently from bf16 | mx8 | nv4 | mx4 by a runtime-switchable policy string (``fp4_policy``).

* mx8 / bf16 kinds call exactly the r17 / r13 kernels (an empty policy therefore reproduces FINAL's block op for op);
* nv4 / mx4 kinds: operands are quantized by the torch simulation ``fp4_sim`` (``fp4_ref`` semantics, dedicated SR generator, RHT, delayed amax)
  and consumed by the REAL QuACK FP4 GEMMs (``fp4_gemm``: nv4 epilogues with device-scalar ts, mx4 with round13's), wgrad output fp32;
* ``prefix``: the FP4 wgrad contracts only over the patch tokens (last 1024 per image); the prefix tokens (plain block 5, joint block 105 per image)
  are added back by a small exact bf16 GEMM (fp32 output);
* X is quantized once in the forward (row copy for the fwd GEMM, transposed copy for the wgrad), dY once in the backward (row copy for dgrad,
  transposed copy for wgrad); weights once per step at block 0 (``Handle.begin_step``), like r17's ``Mx8Weights.refresh``.

Nothing here touches the torch default generators: SR noise comes from ``Sim.gen``.  Parameters / state_dict are untouched.
"""
from __future__ import annotations

import torch

from round13_gemm_fusion_sm120.impl import block_kernels as bk
from round13_gemm_fusion_sm120.impl import gemm_ops as ops13
from round13_gemm_fusion_sm120.impl.fused_block import Shadows, _Bundle, _need, _LN1, eligible
from round17_mxfp8_sm120.impl import gemm_mx8 as G8
from round17_mxfp8_sm120.impl import mx8

from . import fp4_gemm as G4
from . import fp4_policy as P
from . import fp4_ref as R
from . import fp4_sim as S

PATCH_TOKENS = 1024  # patch tokens per image (32 x 32 grid)


# ------------------------------------------------------------------------------------------------ containers
class XQ:
    """Forward-quantized activation copies (only the ones the policy needs)."""
    __slots__ = ("x8", "xf", "xw", "x_bf", "x_pre")

    def __init__(self):
        self.x8 = self.xf = self.xw = self.x_bf = self.x_pre = None


class DQ:
    """Gradient copies (dgrad row copy, wgrad transposed copy)."""
    __slots__ = ("d8", "dr", "dc", "dy_bf", "dy_pre")

    def __init__(self):
        self.d8 = self.dr = self.dc = self.dy_bf = self.dy_pre = None


class WQ:
    """Per-step weight copies: mx8 container (row / col), fp4 f copy (``q`` along K) and d copy (``qt`` along N)."""
    __slots__ = ("w8", "wf", "wd")

    def __init__(self):
        self.w8 = self.wf = self.wd = None


class Ops13:
    """bf16 backends of the block Linears = round13's gemm_ops (name selects the per-shape backend policy)."""

    def __init__(self, name):
        self.name = name

    def fwd_bias(self, x, w, b):
        return ops13.linear_bias(x, w, b, self.name)

    def fwd_res(self, x, w, b, gamma, resid):
        return ops13.linear_res(x, w, b, gamma, resid, self.name)

    def fwd_gelu(self, x, w, b):
        return ops13.linear_gelu(x, w, b)

    def dgrad(self, dy, w):
        return ops13.dgrad(dy, w, self.name)

    def dgrad_dgelu(self, dy, w, z):
        return ops13.dgrad_dgelu(dy, w, z)

    def wgrad(self, dy, x):
        return ops13.wgrad(dy, x, self.name)


# ------------------------------------------------------------------------------------------------ one layer
class LayerQ:
    """Quantization + GEMM dispatch of one Linear (block) or ConvT-as-GEMM (head) layer for a resolved ``LinSpec``.

    ``bf`` provides the bf16 GEMMs (fwd_bias / fwd_res / fwd_gelu / dgrad / dgrad_dgelu / wgrad).  ``site`` prefixes the delayed-amax call-site keys."""

    def __init__(self, sim: S.Sim, site: str, spec: P.LinSpec, bf=None):
        self.sim, self.site, self.spec = sim, site, spec
        self.bf = bf if bf is not None else Ops13(spec.name)

    # ---- weights ----
    def needs_weight(self) -> bool:
        return any(f != "bf16" for f in (self.spec.ff, self.spec.fd))

    def prep_weight(self, w: torch.Tensor) -> WQ:
        """w [N, K] (fp32 Parameter data, or the bf16 ConvT matrix)."""
        s, wq = self.spec, WQ()
        f8, d8 = s.ff == "mx8", s.fd == "mx8"
        if f8 or d8:
            wq.w8 = mx8.quantize(w, row=f8, col=d8)
        fpf, fpd = s.ff in P.FP4, s.fd in P.FP4
        if fpf and fpd and s.ff == s.fd and s.w2d_f == s.w2d_d:
            t = S.q_weight(self.sim, self.site + ".W", w, s.ff, two_d=s.w2d_f, rht_row=s.rht_f, rht_col=s.rht_d)
            wq.wf = wq.wd = t
        else:
            if fpf:
                wq.wf = S.q_weight(self.sim, self.site + ".Wf", w, s.ff, two_d=s.w2d_f, rht_row=s.rht_f, want_col=False)
            if fpd:
                wq.wd = S.q_weight(self.sim, self.site + ".Wd", w, s.fd, two_d=s.w2d_d, rht_col=s.rht_d, want_row=False)
        return wq

    # ---- activations (forward) ----
    def quant_x(self, x: torch.Tensor, ntok=None) -> XQ:
        """x [M, K] bf16 contiguous.  ``ntok`` = tokens per image (for ``prefix``)."""
        s, sim, xq = self.spec, self.sim, XQ()
        f8, w8 = s.ff == "mx8", s.fw == "mx8"
        if f8 or w8:
            xq.x8 = mx8.quantize(x, row=f8, col=w8)
        if s.ff in P.FP4:
            xq.xf = S.q_row(sim, self.site + ".xr", x, s.ff, rht=s.rht_f)
        if s.fw in P.FP4:
            if s.prefix and ntok is not None and ntok > PATCH_TOKENS:
                B, pre = x.shape[0] // ntok, ntok - PATCH_TOKENS
                xv = x.view(B, ntok, x.shape[1])
                xq.x_pre = xv[:, :pre].reshape(B * pre, -1).contiguous()
                xq.xw = S.q_col(sim, self.site + ".xc", xv[:, pre:].reshape(B * PATCH_TOKENS, -1), s.fw, rht=s.rht_w, rpad=R.pad128(B * PATCH_TOKENS))
            else:
                xq.xw = S.q_col(sim, self.site + ".xc", x, s.fw, rht=s.rht_w)
        if s.ff == "bf16" or s.fw == "bf16":
            xq.x_bf = x
        return xq

    def after_fwd(self, xq: XQ) -> None:
        """Free what the backward does not need (the row copies; the bf16 x unless the wgrad is bf16)."""
        if xq.x8 is not None:
            xq.x8.qr = xq.x8.sfr = None
        xq.xf = None
        if self.spec.fw != "bf16":
            xq.x_bf = None

    def fwd_bias(self, xq, wq, bias, shadow):
        f = self.spec.ff
        if f == "bf16":
            return self.bf.fwd_bias(xq.x_bf, shadow, bias)
        if f == "mx8":
            return G8.fwd_bias(xq.x8, wq.w8, bias)
        return G4.fwd_bias(xq.xf, wq.wf, bias)

    def fwd_res(self, xq, wq, bias, gamma, resid, shadow):
        f = self.spec.ff
        if f == "bf16":
            return self.bf.fwd_res(xq.x_bf, shadow, bias, gamma, resid)
        if f == "mx8":
            return G8.fwd_res(xq.x8, wq.w8, bias, gamma, resid)
        return G4.fwd_res(xq.xf, wq.wf, bias, gamma, resid)

    def fwd_gelu(self, xq, wq, bias, shadow):
        f = self.spec.ff
        if f == "bf16":
            return self.bf.fwd_gelu(xq.x_bf, shadow, bias)
        if f == "mx8":
            return G8.fwd_gelu(xq.x8, wq.w8, bias)
        return G4.fwd_gelu(xq.xf, wq.wf, bias)

    # ---- gradients (backward) ----
    def quant_dy(self, dy: torch.Tensor, ntok=None) -> DQ:
        s, sim, dq = self.spec, self.sim, DQ()
        d8, w8 = s.fd == "mx8", s.fw == "mx8"
        if d8 or w8:
            dq.d8 = mx8.quantize(dy, row=d8, col=w8)
        if s.fd in P.FP4:
            dq.dr = S.q_row(sim, self.site + ".dr", dy, s.fd, rht=s.rht_d, sr=s.sr_d)
        if s.fw in P.FP4:
            if s.prefix and ntok is not None and ntok > PATCH_TOKENS:
                B, pre = dy.shape[0] // ntok, ntok - PATCH_TOKENS
                dv = dy.view(B, ntok, dy.shape[1])
                dq.dy_pre = dv[:, :pre].reshape(B * pre, -1).contiguous()
                dq.dc = S.q_col(sim, self.site + ".dc", dv[:, pre:].reshape(B * PATCH_TOKENS, -1), s.fw, rht=s.rht_w, sr=s.sr_w, rpad=R.pad128(B * PATCH_TOKENS))
            else:
                dq.dc = S.q_col(sim, self.site + ".dc", dy, s.fw, rht=s.rht_w, sr=s.sr_w)
        if s.fd == "bf16" or s.fw == "bf16":
            dq.dy_bf = dy
        return dq

    def dgrad(self, dq, wq, shadow):
        f = self.spec.fd
        if f == "bf16":
            return self.bf.dgrad(dq.dy_bf, shadow)
        if f == "mx8":
            return G8.dgrad(dq.d8, wq.w8)
        return G4.dgrad(dq.dr, wq.wd)

    def dgrad_dgelu(self, dq, wq, shadow, z):
        f = self.spec.fd
        if f == "bf16":
            return self.bf.dgrad_dgelu(dq.dy_bf, shadow, z)
        if f == "mx8":
            return G8.dgrad_dgelu(dq.d8, wq.w8, z)
        return G4.dgrad_dgelu(dq.dr, wq.wd, z)

    def wgrad(self, dq, xq):
        f = self.spec.fw
        if f == "bf16":
            return self.bf.wgrad(dq.dy_bf, xq.x_bf)
        if f == "mx8":
            return G8.wgrad(dq.d8, xq.x8)
        dw = G4.wgrad(dq.dc, xq.xw)
        if dq.dy_pre is not None:  # prefix tokens: exact bf16 contraction, fp32 output, added to the FP4 result
            dw += torch.mm(dq.dy_pre.t(), xq.x_pre, out_dtype=torch.float32)
        return dw


# ------------------------------------------------------------------------------------------------ Functions
class _QLinear(torch.autograd.Function):
    """qkv: ``y = x @ W^T + b`` (bf16 out), fp32 wgrad, fused bias-gradient column sum."""

    @staticmethod
    def forward(ctx, x, weight, lq, wq, shadow, bias, ntok):
        with torch.autocast("cuda", enabled=False):
            xq = lq.quant_x(x, ntok)
            y = lq.fwd_bias(xq, wq, bias, shadow)
        lq.after_fwd(xq)
        ctx.lq, ctx.xq, ctx.wq, ctx.shadow, ctx.ntok = lq, xq, wq, shadow, ntok
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        lq = ctx.lq
        dq = lq.quant_dy(dy, ctx.ntok)
        dx = lq.dgrad(dq, ctx.wq, ctx.shadow) if _need(ctx, 0) else None
        dw = lq.wgrad(dq, ctx.xq) if _need(ctx, 1) else None
        db = bk.colsum_bf16(dy) if (ctx.has_bias and _need(ctx, 5)) else None
        ctx.xq = ctx.wq = ctx.shadow = None
        return dx, dw, None, None, None, db, None


class _QProjRes(torch.autograd.Function):
    """``x1 = xr + gamma1 * (o @ Wp^T + bp)`` (epilogue) followed by ``ln2 = LN2(x1)``."""

    @staticmethod
    def forward(ctx, o, xr, weight, lq, wq, shadow, bias, gamma, ln_w, ln_b, eps, ntok):
        with torch.autocast("cuda", enabled=False):
            oq = lq.quant_x(o, ntok)
            x1, y1 = lq.fwd_res(oq, wq, bias, gamma, xr, shadow)
            ln2, mean, rstd = bk.ln_fwd(x1, ln_w, ln_b, eps)
        lq.after_fwd(oq)
        ctx.lq, ctx.oq, ctx.wq, ctx.shadow, ctx.ntok = lq, oq, wq, shadow, ntok
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
        lq = ctx.lq
        dq = lq.quant_dy(g, ctx.ntok)
        do = lq.dgrad(dq, ctx.wq, ctx.shadow) if _need(ctx, 0) else None
        dw = lq.wgrad(dq, ctx.oq) if _need(ctx, 2) else None
        ctx.oq = ctx.wq = ctx.shadow = None
        # grads: o, xr, weight, lq, wq, shadow, bias, gamma, ln_w, ln_b, eps, ntok
        return do, dx1, dw, None, None, None, sums[3], sums[2], sums[0], sums[1], None, None


class _QFC1(torch.autograd.Function):
    """``z = ln2 @ W1^T + b1``, ``h = GELU_erf(z)`` (epilogue); same protocol as round13 ``_FC1`` (z is a non-differentiable carrier)."""

    @staticmethod
    def forward(ctx, ln2, weight, lq, wq, shadow, bias, bundle, ntok):
        with torch.autocast("cuda", enabled=False):
            xq = lq.quant_x(ln2, ntok)
            z, h = lq.fwd_gelu(xq, wq, bias, shadow)
        lq.after_fwd(xq)
        ctx.lq, ctx.xq, ctx.wq, ctx.shadow, ctx.bundle, ctx.ntok = lq, xq, wq, shadow, bundle, ntok
        ctx.mark_non_differentiable(z)
        ctx.set_materialize_grads(False)
        return h, z

    @staticmethod
    def backward(ctx, dz, _unused):
        lq = ctx.lq
        dq = lq.quant_dy(dz.contiguous(), ctx.ntok)
        dln2 = lq.dgrad(dq, ctx.wq, ctx.shadow) if _need(ctx, 0) else None
        dw = lq.wgrad(dq, ctx.xq) if _need(ctx, 1) else None
        db = ctx.bundle.db1
        ctx.bundle.db1 = None
        ctx.xq = ctx.wq = ctx.shadow = None
        return dln2, dw, None, None, None, db, None, None


class _QFC2Res(torch.autograd.Function):
    """``x2 = x1 + gamma2 * (h @ W2^T + b2)`` (epilogue); backward applies GELU' in the dgrad epilogue (round13 protocol)."""

    @staticmethod
    def forward(ctx, h, z, x1, weight, lq, wq, shadow, bias, gamma, bundle, ntok):
        with torch.autocast("cuda", enabled=False):
            hq = lq.quant_x(h, ntok)
            x2, y2 = lq.fwd_res(hq, wq, bias, gamma, x1, shadow)
        lq.after_fwd(hq)
        ctx.lq, ctx.hq, ctx.wq, ctx.shadow, ctx.bundle, ctx.ntok = lq, hq, wq, shadow, bundle, ntok
        ctx.save_for_backward(z, y2, gamma)
        return x2

    @staticmethod
    def backward(ctx, g2):
        z, y2, gamma = ctx.saved_tensors
        g2 = g2.contiguous()
        g, sums = bk.gamma_prep(g2, y2, gamma)  # dy2 = gamma2*g2 (bf16); sums = [dGamma2, db2]
        lq = ctx.lq
        dq = lq.quant_dy(g, ctx.ntok)
        dz, db1 = lq.dgrad_dgelu(dq, ctx.wq, ctx.shadow, z)
        ctx.bundle.db1 = db1
        dw = lq.wgrad(dq, ctx.hq) if _need(ctx, 3) else None
        ctx.hq = ctx.wq = ctx.shadow = None
        # grads: h, z, x1, weight, lq, wq, shadow, bias, gamma, bundle, ntok
        return dz, None, g2, dw, None, None, None, sums[1], sums[0], None, None


# ------------------------------------------------------------------------------------------------ handle (runtime policy)
class Handle:
    """Owns the policy, the simulation state and the per-step weight copies.  ``ctx.meta['round20_fp4_probe']``."""

    def __init__(self, ctx, variant: str = ""):
        self.ctx = ctx
        self.shadows = Shadows(ctx)
        blocks = ctx.model.backbone.blocks
        self.nblocks = len(blocks)
        self.params = {}
        for i, blk in enumerate(blocks):
            for name, lin in zip(P.LINS, (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2)):
                self.params[(i, name)] = lin.weight
        self.sim = S.Sim(device=next(ctx.model.parameters()).device)
        self.head_call = 0
        self.set_policy(variant)

    def set_policy(self, text: str) -> None:
        """Switch the policy at run time (builds nothing on the GPU; clears the delayed-amax table and re-seeds the SR generator)."""
        self.policy = P.parse(text)
        self.sim.reset(seed=777, amax=self.policy.amax)
        self.layers: dict = {}
        self.weights: dict = {}
        self.head_call = 0
        self.ctx.meta["round20_fp4_probe_policy"] = self.policy.describe()

    def layer(self, i: int, name: str) -> LayerQ:
        key = (i, name)
        lq = self.layers.get(key)
        if lq is None:
            lq = self.layers[key] = LayerQ(self.sim, f"b{i}.{name}", self.policy.spec(name, i, self.nblocks))
        return lq

    @torch.no_grad()
    def begin_step(self) -> None:
        """Block 0: refresh bf16 shadows (unless an optimizer patch owns them), re-quantize the weights of every layer that has a quantized f / d kind."""
        self.head_call = 0
        self.shadows.refresh()
        self.weights = {k: self.layer(*k).prep_weight(p.detach()) for k, p in self.params.items() if self.layer(*k).needs_weight()}

    def wq(self, i, name):
        return self.weights.get((i, name))


# ------------------------------------------------------------------------------------------------ block
def make_block(ctx, previous, handle: Handle):
    shadows = handle.shadows

    def block(c, i, blk, x, rope, mask):
        reason = eligible(blk, x)
        if reason is not None:
            return previous(c, i, blk, x, rope, mask)
        if i == 0:
            handle.begin_step()
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
        qkv = _QLinear.apply(ln1, qkv_lin.weight, handle.layer(i, "qkv"), handle.wq(i, "qkv"), shadows.get(qkv_lin.weight), qkv_bias, N)
        o = c.ops.attn_core(c, qkv.view(B, N, 3 * d), rope, mask, attn.num_heads)
        o = o.reshape(M, d)
        if o.dtype != torch.bfloat16:
            o = o.to(torch.bfloat16)
        o = o.contiguous()
        x1, ln2 = _QProjRes.apply(o, xr, attn.proj.weight, handle.layer(i, "proj"), handle.wq(i, "proj"), shadows.get(attn.proj.weight), attn.proj.bias,
                                  blk.ls1.gamma, blk.norm2.weight, blk.norm2.bias, blk.norm2.eps, N)
        h, z = _QFC1.apply(ln2, blk.mlp.fc1.weight, handle.layer(i, "fc1"), handle.wq(i, "fc1"), shadows.get(blk.mlp.fc1.weight), blk.mlp.fc1.bias, bundle, N)
        x2 = _QFC2Res.apply(h, z, x1, blk.mlp.fc2.weight, handle.layer(i, "fc2"), handle.wq(i, "fc2"), shadows.get(blk.mlp.fc2.weight), blk.mlp.fc2.bias,
                            blk.ls2.gamma, bundle, N)
        return x2.view(B, N, d)

    block.handle = handle
    block.shadows = shadows
    return block
