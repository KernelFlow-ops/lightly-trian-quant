"""Round22 REAL-kernel FP4 backbone block: round20b's "half-real" block (``round20_fp4_probe/impl/fp4_block.py``) with the torch simulation quantizer replaced by
round21's fused quantizers (``fp4q``).  The structure (r17 block, per-GEMM-kind formats from the R20 policy grammar) is unchanged.

Supported kinds:  f in {bf16, mx8}  (forward stays FINAL's: qkv bf16, proj / fc1 / fc2 mx8, r17 kernels),  d, w in {mx8, nv4}  (nv4 = the new real path).
Not supported (raise at ``set_policy``): f = nv4 / mx4, mx4 anywhere, rht on f / d (the weight quantizer has no RHT), ``prefix`` (CONTRACTS 13.2: nofirst instead).

Data flow of one nv4-d/w Linear (Y = X W^T, M tokens):
  forward   X -> ``mx8.quantize(row=True, col=False)`` (r17, feeds the forward GEMM)   [qkv: none, its forward GEMM is bf16]
              -> ``fp4q.quant_act(row=False, col=True)``: nv4 transposed copy ``qt`` (blocks along the zero padded token axis), kept for the wgrad
            W (block 0, once per step) -> ``mx8.quantize(row=True, col=False)`` (forward)  +  ``fp4q.quant_weight`` (2D 16x16 scale; only ``qt`` = W^T is used, dgrad)
  backward  dY -> ``fp4q.quant_act(row=True, col=True, sr_row=True, sr_col=True)`` ONE read: ``q`` (dgrad A) and ``qt`` (wgrad A)
            dgrad = QuACK nv4 ``dY.q . W.qt^T`` (fc2: ``dgelu_ts`` epilogue = GELU' + db1; others plain), wgrad = QuACK nv4 ``dY.qt . X.qt^T`` fp32 (split-K for small outputs)
Everything outside the GEMM operands (LN, residual, LayerScale, GELU, attention, bias / gamma gradients) is round13 / round17 code unchanged.

Per-tensor amax: ``delayed`` (prev step x headroom, first call exact; ``FP4State`` rolled once per step by the ``opt_step`` wrapper installed in ``patch.py``) or ``exact``.
Headroom (CONTRACTS 13.5): X and weights 2, dY 8 when the round21 quantizer has the ``headroom`` argument (``dyhr=`` / ``xhr=`` / ``whr=`` policy tokens override).
SR noise: Philox in the kernel (seed = device counter, +1 per roll, fixed ``sr_offset`` per call site); torch's default generators are never touched.
"""
from __future__ import annotations

import inspect
import re

import torch

from round13_gemm_fusion_sm120.impl import block_kernels as bk
from round13_gemm_fusion_sm120.impl import gemm_ops as ops13
from round13_gemm_fusion_sm120.impl.fused_block import Shadows, _Bundle, _need, _LN1, eligible
from round17_mxfp8_sm120.impl import gemm_mx8 as G8
from round17_mxfp8_sm120.impl import mx8
from round20_fp4_probe.impl import fp4_policy as P
from round21_fp4_quant_head.impl import fp4q

from . import fp4_gemm22 as G4

KIND = {"x": 0, "dy": 1, "w": 2}
EXTRA = {"dyhr": "hr_dy", "xhr": "hr_x", "whr": "hr_w"}


# ------------------------------------------------------------------------------------------------ containers
class XQ:
    """Forward-quantized activation copies: mx8 (row for the forward GEMM, col if the wgrad is mx8) and the nv4 transposed copy for the wgrad."""
    __slots__ = ("x8", "x4", "x_bf")

    def __init__(self):
        self.x8 = self.x4 = self.x_bf = None


class DQ:
    """Gradient copies: mx8 (d8) and nv4 (d4: ``q`` for dgrad, ``qt`` for wgrad)."""
    __slots__ = ("d8", "d4", "dy_bf")

    def __init__(self):
        self.d8 = self.d4 = self.dy_bf = None


class WQ:
    """Per-step weight copies: mx8 container (``w8``: row for the forward, col if the dgrad is mx8) and the nv4 W^T copy (``wd``)."""
    __slots__ = ("w8", "wd")

    def __init__(self):
        self.w8 = self.wd = None


class Ops13:
    """bf16 backends of the block Linears = round13's gemm_ops (name selects the per-shape backend policy)."""

    def __init__(self, name):
        self.name = name

    def fwd_bias(self, x, w, b):
        return ops13.linear_bias(x, w, b, self.name)

    def dgrad(self, dy, w):
        return ops13.dgrad(dy, w, self.name)

    def wgrad(self, dy, x):
        return ops13.wgrad(dy, x, self.name)


# ------------------------------------------------------------------------------------------------ one layer
class LayerQ:
    """Quantization + GEMM dispatch of one block Linear for a resolved ``LinSpec``.  ``site`` ids (x, dy, w) index the ``FP4State`` amax table."""

    def __init__(self, h: "Handle", i: int, name: str, spec: P.LinSpec):
        self.h, self.i, self.name, self.spec = h, i, name, spec
        base = (i * len(P.LINS) + P.LINS.index(name)) * 3
        self.s_x, self.s_dy, self.s_w = base + KIND["x"], base + KIND["dy"], base + KIND["w"]
        self.bf = Ops13(name)

    def _kw(self, site: int, hr: float) -> dict:
        h = self.h
        h.check_site(site)
        kw = dict(state=h.state, site=site, amax=h.policy.amax)
        if h.has_headroom and h.policy.amax == "delayed":
            kw["headroom"] = hr
        return kw

    # ---- weights ----
    def needs_weight(self) -> bool:
        s = self.spec
        return s.ff == "mx8" or s.fd in ("mx8", "nv4")

    def prep_weight(self, w: torch.Tensor) -> WQ:
        s, wq = self.spec, WQ()
        f8, d8 = s.ff == "mx8", s.fd == "mx8"
        if f8 or d8:
            wq.w8 = mx8.quantize(w, row=f8, col=d8)
        if s.fd == "nv4":
            t = fp4q.quant_weight(w, "nv4", **self._kw(self.s_w, self.h.hr_w))
            t.q = t.sf = None  # the forward stays mx8 / bf16: only the W^T copy (qt, 2D scales, tst) is consumed (dgrad)
            wq.wd = t
        return wq

    # ---- activations (forward) ----
    def quant_x(self, x: torch.Tensor) -> XQ:
        s, xq = self.spec, XQ()
        f8, w8 = s.ff == "mx8", s.fw == "mx8"
        if f8 or w8:
            xq.x8 = mx8.quantize(x, row=f8, col=w8)
        if s.fw == "nv4":
            xq.x4 = fp4q.quant_act(x, "nv4", row=False, col=True, rht_col=s.rht_w, **self._kw(self.s_x, self.h.hr_x))
        if s.ff == "bf16" or s.fw == "bf16":
            xq.x_bf = x
        return xq

    def after_fwd(self, xq: XQ) -> None:
        """Free what the backward does not need (the mx8 row copy; the bf16 x unless the wgrad is bf16)."""
        if xq.x8 is not None:
            xq.x8.qr = xq.x8.sfr = None
        if self.spec.fw != "bf16":
            xq.x_bf = None

    def fwd_bias(self, xq, wq, bias, shadow):
        f = self.spec.ff
        if f == "bf16":
            return self.bf.fwd_bias(xq.x_bf, shadow, bias)
        return G8.fwd_bias(xq.x8, wq.w8, bias)

    def fwd_res(self, xq, wq, bias, gamma, resid, shadow):
        assert self.spec.ff == "mx8"
        return G8.fwd_res(xq.x8, wq.w8, bias, gamma, resid)

    def fwd_gelu(self, xq, wq, bias, shadow):
        assert self.spec.ff == "mx8"
        return G8.fwd_gelu(xq.x8, wq.w8, bias)

    # ---- gradients (backward) ----
    def quant_dy(self, dy: torch.Tensor) -> DQ:
        s, dq = self.spec, DQ()
        d8, w8, d4, w4 = s.fd == "mx8", s.fw == "mx8", s.fd == "nv4", s.fw == "nv4"
        if d8 or w8:
            dq.d8 = mx8.quantize(dy, row=d8, col=w8)
        if d4 or w4:
            dq.d4 = fp4q.quant_act(dy, "nv4", row=d4, col=w4, rht_col=s.rht_w, sr_row=s.sr_d, sr_col=s.sr_w, sr_offset=self.s_dy, **self._kw(self.s_dy, self.h.hr_dy))
        if s.fd == "bf16" or s.fw == "bf16":
            dq.dy_bf = dy
        return dq

    def dgrad(self, dq, wq, shadow):
        f = self.spec.fd
        if f == "bf16":
            return self.bf.dgrad(dq.dy_bf, shadow)
        if f == "mx8":
            return G8.dgrad(dq.d8, wq.w8)
        return G4.dgrad(dq.d4, wq.wd)

    def dgrad_dgelu(self, dq, wq, shadow, z):
        f = self.spec.fd
        assert f in ("mx8", "nv4"), f
        if f == "mx8":
            return G8.dgrad_dgelu(dq.d8, wq.w8, z)
        return G4.dgrad_dgelu(dq.d4, wq.wd, z)

    def wgrad(self, dq, xq):
        f = self.spec.fw
        if f == "bf16":
            return self.bf.wgrad(dq.dy_bf, xq.x_bf)
        if f == "mx8":
            return G8.wgrad(dq.d8, xq.x8)
        return G4.wgrad(dq.d4, xq.x4)


# ------------------------------------------------------------------------------------------------ Functions
class _QLinear(torch.autograd.Function):
    """qkv: ``y = x @ W^T + b`` (bf16 out), fp32 wgrad, fused bias-gradient column sum."""

    @staticmethod
    def forward(ctx, x, weight, lq, wq, shadow, bias):
        with torch.autocast("cuda", enabled=False):
            xq = lq.quant_x(x)
            y = lq.fwd_bias(xq, wq, bias, shadow)
        lq.after_fwd(xq)
        ctx.lq, ctx.xq, ctx.wq, ctx.shadow = lq, xq, wq, shadow
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        lq = ctx.lq
        dq = lq.quant_dy(dy)
        dx = lq.dgrad(dq, ctx.wq, ctx.shadow) if _need(ctx, 0) else None
        dw = lq.wgrad(dq, ctx.xq) if _need(ctx, 1) else None
        db = bk.colsum_bf16(dy) if (ctx.has_bias and _need(ctx, 5)) else None
        ctx.xq = ctx.wq = ctx.shadow = None
        return dx, dw, None, None, None, db


class _QProjRes(torch.autograd.Function):
    """``x1 = xr + gamma1 * (o @ Wp^T + bp)`` (epilogue) followed by ``ln2 = LN2(x1)``."""

    @staticmethod
    def forward(ctx, o, xr, weight, lq, wq, shadow, bias, gamma, ln_w, ln_b, eps):
        with torch.autocast("cuda", enabled=False):
            oq = lq.quant_x(o)
            x1, y1 = lq.fwd_res(oq, wq, bias, gamma, xr, shadow)
            ln2, mean, rstd = bk.ln_fwd(x1, ln_w, ln_b, eps)
        lq.after_fwd(oq)
        ctx.lq, ctx.oq, ctx.wq, ctx.shadow = lq, oq, wq, shadow
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
        dq = lq.quant_dy(g)
        do = lq.dgrad(dq, ctx.wq, ctx.shadow) if _need(ctx, 0) else None
        dw = lq.wgrad(dq, ctx.oq) if _need(ctx, 2) else None
        ctx.oq = ctx.wq = ctx.shadow = None
        # grads: o, xr, weight, lq, wq, shadow, bias, gamma, ln_w, ln_b, eps
        return do, dx1, dw, None, None, None, sums[3], sums[2], sums[0], sums[1], None


class _QFC1(torch.autograd.Function):
    """``z = ln2 @ W1^T + b1``, ``h = GELU_erf(z)`` (epilogue); same protocol as round13 ``_FC1`` (z is a non-differentiable carrier)."""

    @staticmethod
    def forward(ctx, ln2, weight, lq, wq, shadow, bias, bundle):
        with torch.autocast("cuda", enabled=False):
            xq = lq.quant_x(ln2)
            z, h = lq.fwd_gelu(xq, wq, bias, shadow)
        lq.after_fwd(xq)
        ctx.lq, ctx.xq, ctx.wq, ctx.shadow, ctx.bundle = lq, xq, wq, shadow, bundle
        ctx.mark_non_differentiable(z)
        ctx.set_materialize_grads(False)
        return h, z

    @staticmethod
    def backward(ctx, dz, _unused):
        lq = ctx.lq
        dq = lq.quant_dy(dz.contiguous())
        dln2 = lq.dgrad(dq, ctx.wq, ctx.shadow) if _need(ctx, 0) else None
        dw = lq.wgrad(dq, ctx.xq) if _need(ctx, 1) else None
        db = ctx.bundle.db1
        ctx.bundle.db1 = None
        ctx.xq = ctx.wq = ctx.shadow = None
        return dln2, dw, None, None, None, db, None


class _QFC2Res(torch.autograd.Function):
    """``x2 = x1 + gamma2 * (h @ W2^T + b2)`` (epilogue); backward applies GELU' in the dgrad epilogue (round13 protocol)."""

    @staticmethod
    def forward(ctx, h, z, x1, weight, lq, wq, shadow, bias, gamma, bundle):
        with torch.autocast("cuda", enabled=False):
            hq = lq.quant_x(h)
            x2, y2 = lq.fwd_res(hq, wq, bias, gamma, x1, shadow)
        lq.after_fwd(hq)
        ctx.lq, ctx.hq, ctx.wq, ctx.shadow, ctx.bundle = lq, hq, wq, shadow, bundle
        ctx.save_for_backward(z, y2, gamma)
        return x2

    @staticmethod
    def backward(ctx, g2):
        z, y2, gamma = ctx.saved_tensors
        g2 = g2.contiguous()
        g, sums = bk.gamma_prep(g2, y2, gamma)  # dy2 = gamma2*g2 (bf16); sums = [dGamma2, db2]
        lq = ctx.lq
        dq = lq.quant_dy(g)
        dz, db1 = lq.dgrad_dgelu(dq, ctx.wq, ctx.shadow, z)
        ctx.bundle.db1 = db1
        dw = lq.wgrad(dq, ctx.hq) if _need(ctx, 3) else None
        ctx.hq = ctx.wq = ctx.shadow = None
        # grads: h, z, x1, weight, lq, wq, shadow, bias, gamma, bundle
        return dz, None, g2, dw, None, None, None, sums[1], sums[0], None


# ------------------------------------------------------------------------------------------------ handle (runtime policy)
def split_variant(text: str):
    """Pull the round22-only tokens (``dyhr=8`` ``xhr=2`` ``whr=2``) out of a policy string; the rest is the round20 grammar."""
    extras, rest = {}, []
    for tok in [t.strip() for t in re.split(r"[,+]", text) if t.strip()]:
        key, _, val = tok.partition("=")
        if key in EXTRA and val:
            extras[EXTRA[key]] = float(val)
        else:
            rest.append(tok)
    return extras, "+".join(rest)


class Handle:
    """Owns the policy, the ``FP4State`` and the per-step weight copies.  ``ctx.meta['round22_fp4_backbone']``."""

    def __init__(self, ctx, variant: str = ""):
        self.ctx = ctx
        self.shadows = Shadows(ctx)
        blocks = ctx.model.backbone.blocks
        self.nblocks = len(blocks)
        self.params = {}
        for i, blk in enumerate(blocks):
            for name, lin in zip(P.LINS, (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2)):
                self.params[(i, name)] = lin.weight
        self.device = next(ctx.model.parameters()).device
        self.n_sites = 3 * len(P.LINS) * self.nblocks
        self.site_names = [f"b{i}.{n}.{k}" for i in range(self.nblocks) for n in P.LINS for k in KIND]
        self.has_headroom = "headroom" in inspect.signature(fp4q.quant_act).parameters
        self.state = None
        self.set_policy(variant)

    def set_policy(self, text: str) -> None:
        """Switch the policy at run time: builds a fresh ``FP4State`` (amax table cleared, seed reset), clears the per-layer / weight caches."""
        extras, rest = split_variant(text)
        self.policy = P.parse(rest)
        self.hr_x, self.hr_w = extras.get("hr_x", 2.0), extras.get("hr_w", 2.0)
        self.hr_dy = extras.get("hr_dy", 8.0 if self.has_headroom else 2.0)
        pol = self.policy
        if pol.prefix:
            raise NotImplementedError("round22: prefix is not implemented (CONTRACTS 13.2: use nofirst)")
        for name in P.LINS:
            for i in range(self.nblocks):
                s = pol.spec(name, i, self.nblocks)
                if s.ff not in ("bf16", "mx8") or s.fd not in ("mx8", "nv4") or s.fw not in ("mx8", "nv4") or s.rht_f or s.rht_d:
                    raise NotImplementedError(f"round22 supports f in (bf16, mx8), d / w in (mx8, nv4) without rht on f / d; got {s}")
        self.state = fp4q.FP4State(self.n_sites, device=self.device)
        self.layers: dict = {}
        self.weights: dict = {}
        self.ctx.meta["round22_fp4_backbone_policy"] = self.describe()

    def describe(self) -> str:
        return f"{self.policy.describe()} [amax={self.policy.amax} hr_x={self.hr_x} hr_dy={self.hr_dy} hr_w={self.hr_w} headroom_arg={self.has_headroom}]"

    def layer(self, i: int, name: str) -> LayerQ:
        key = (i, name)
        lq = self.layers.get(key)
        if lq is None:
            lq = self.layers[key] = LayerQ(self, i, name, self.policy.spec(name, i, self.nblocks))
        return lq

    def active_sites(self) -> list:
        """Site ids the policy actually uses (nofirst blocks have none): x (nv4 wgrad), dy (nv4 dgrad or wgrad), w (nv4 dgrad weight)."""
        out = []
        for i in range(self.nblocks):
            for n in P.LINS:
                lq = self.layer(i, n)
                s = lq.spec
                if s.fw == "nv4":
                    out.append(lq.s_x)
                if s.fd == "nv4" or s.fw == "nv4":
                    out.append(lq.s_dy)
                if s.fd == "nv4":
                    out.append(lq.s_w)
        return out

    def check_site(self, site: int) -> None:
        """A delayed-amax site must be initialised eagerly (exact pass) BEFORE any CUDA graph capture: the first call inside a capture would bake the init pass into the graph."""
        if self.policy.amax == "delayed" and not self.state.inited[site] and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"round22: first call of delayed-amax site {self.site_names[site]} during CUDA graph capture (warm the step eagerly first)")

    @torch.no_grad()
    def refresh_shadows(self) -> None:
        """bf16 shadows only for the Linears whose forward is bf16 (qkv); the optimizer patch maintains them when ``shadows.external``."""
        if self.shadows.external:
            return
        need = [self.shadows.items[len(P.LINS) * i + j] for i in range(self.nblocks) for j, n in enumerate(P.LINS) if self.layer(i, n).spec.ff == "bf16"]
        if need:
            torch._foreach_copy_([self.ctx.shadow[n] for n, _ in need], [p.detach() for _, p in need])

    @torch.no_grad()
    def begin_step(self) -> None:
        """Block 0: refresh the needed bf16 shadows, re-quantize the weights (mx8 copies as r17, nv4 W^T copy of every nv4-dgrad layer)."""
        self.refresh_shadows()
        self.weights = {k: self.layer(*k).prep_weight(p.detach()) for k, p in self.params.items() if self.layer(*k).needs_weight()}

    def wq(self, i, name):
        return self.weights.get((i, name))

    def stats(self) -> dict:
        """Compact host-side saturation summary of round21's ``FP4State.stats`` (6b 13.5): per kind (x = X col copy, dy = dY row / col copy, w = weight), the roll comparisons
        ``calls``, ``exceed`` (current amax > headroom x the amax its scale was built from: the nv4 block scale saturated at 448 that step) and the max cur/prev ratio, plus the
        worst sites.  One device sync; not for use inside a graph."""
        fn = getattr(self.state, "stats", None)
        if fn is None:
            return {"available": False}
        raw = fn()
        kinds = tuple(KIND)
        per = {k: {"calls": 0, "exceed": 0, "max_ratio": 0.0, "n_inf": 0, "sites_exceeding": 0} for k in kinds}
        worst = []
        for c in ("row", "col"):
            d = raw[c]
            for s in range(raw["n_sites"]):
                calls, ex, mr = d["calls"][s], d["exceed"][s], d["max_ratio"][s]
                if calls == 0:
                    continue
                k = per[kinds[s % 3]]
                k["calls"] += calls
                k["exceed"] += ex
                k["sites_exceeding"] += int(ex > 0)
                if mr == float("inf"):
                    k["n_inf"] += 1
                else:
                    k["max_ratio"] = max(k["max_ratio"], mr)
                if ex > 0:
                    worst.append([f"{self.site_names[s]}.{c}", calls, ex, None if mr == float("inf") else round(mr, 3)])
        worst.sort(key=lambda r: -r[2])
        return {"available": True, "step": raw["step"], "total_calls": raw["total_calls"], "total_exceed": raw["total_exceed"], "exceed_rate": raw["exceed_rate"],
                "headroom": {"x": self.hr_x, "dy": self.hr_dy, "w": self.hr_w}, "per_kind": per, "worst_sites": worst[:12]}


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
        qkv = _QLinear.apply(ln1, qkv_lin.weight, handle.layer(i, "qkv"), handle.wq(i, "qkv"), shadows.get(qkv_lin.weight), qkv_bias)
        o = c.ops.attn_core(c, qkv.view(B, N, 3 * d), rope, mask, attn.num_heads)
        o = o.reshape(M, d)
        if o.dtype != torch.bfloat16:
            o = o.to(torch.bfloat16)
        o = o.contiguous()
        x1, ln2 = _QProjRes.apply(o, xr, attn.proj.weight, handle.layer(i, "proj"), handle.wq(i, "proj"), shadows.get(attn.proj.weight), attn.proj.bias,
                                  blk.ls1.gamma, blk.norm2.weight, blk.norm2.bias, blk.norm2.eps)
        h, z = _QFC1.apply(ln2, blk.mlp.fc1.weight, handle.layer(i, "fc1"), handle.wq(i, "fc1"), shadows.get(blk.mlp.fc1.weight), blk.mlp.fc1.bias, bundle)
        x2 = _QFC2Res.apply(h, z, x1, blk.mlp.fc2.weight, handle.layer(i, "fc2"), handle.wq(i, "fc2"), shadows.get(blk.mlp.fc2.weight), blk.mlp.fc2.bias,
                            blk.ls2.gamma, bundle)
        return x2.view(B, N, d)

    block.handle = handle
    block.shadows = shadows
    block.previous = previous
    return block
