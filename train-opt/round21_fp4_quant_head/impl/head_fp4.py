"""Round21b REAL-kernel FP4 head: ``ops.predict`` override.  Structure copied from round14's ``impl/head.py`` and round20b's ``impl/fp4_head.py`` (both untouched).

Same fused EoMT head as r14 (ConvT k2s2 as GEMM -> fused GELU / dwconv3x3 / LN2D kernels -> einsum with the mask embeddings).  Each of the two ConvT GEMMs picks the
format of its f / d / w kind from the policy string of round20b (``fp4_policy``; ``head.f / head.d / head.w``):

    bf16   r14's QuACK fwd / dgrad, cuBLAS fp32-out wgrad
    mx8    r17 quantizer + QuACK MXFP8 GEMMs (r14 ``mx8`` path)
    nv4    round21a fused quantizer (``fp4q``) + REAL QuACK NVFP4 GEMMs (device-scalar ts)

Policies used by round21b (CONTRACTS 6b 13.2)::

    H3  head.f=mx8,head.dw=nv4+sr   (the strategy)         H1  head.dw=nv4+sr (forward stays bf16)         H2  head.dw=mx8 (r14's mx8_bwd path)

Data flow of one ConvT (4 predicts per step: 3 joint-block predicts + final):
    forward : X is read by the mx8 row quantizer (forward GEMM) and by the nv4 column quantizer (transposed copy ``qt`` along the token axis, kept for the wgrad);
              W is quantized ONCE per step (first predict of the forward): mx8 row copy + nv4 ``q`` / ``qt`` (2D 16x16 scales), shared by the other predicts.
    backward: dZ is read ONCE by ``fp4q.quant_act(row=True, col=True, sr_row=True, sr_col=True, headroom=8)`` -> ``q`` (dgrad A operand) and ``qt`` (wgrad A operand).
              dgrad = dZ.q x W.qt, wgrad = dZ.qt x X.qt (fp32 out = the Parameter gradient), alpha = ts_a * ts_b as a device scalar.
Unchanged r14 parts: the fused GELU / dwconv / LN2D kernels, the einsum, mask_head, class_head, backbone.norm.

Delayed amax state (``fp4q.FP4State``): one site per (predict call, ConvT, X-column copy | dZ copies) and one per ConvT weight; ``Handle.end_step`` (wrapped around
``ctx.ops.opt_step`` by the patch) rolls it once per step.  Everything is capturable by a CUDA graph (no host sync, static shapes, buffers from the graph pool).
"""
from __future__ import annotations

import torch

from round14_head_sm120.impl import head as h14
from round17_mxfp8_sm120.impl import gemm_mx8 as G8
from round17_mxfp8_sm120.impl import mx8
from round20_fp4_probe.impl import fp4_cands as C
from round20_fp4_probe.impl import fp4_epi
from round20_fp4_probe.impl import fp4_policy as P

from . import fp4q

ht = h14.ht
H3 = "head.f=mx8,head.dw=nv4+sr"
H1 = "head.dw=nv4+sr"
H2 = "head.dw=mx8"
GEMM_IMPL = "epi"   # nv4 plain GEMM: "epi" = fp4_epi plain_ts epilogue (device scalar ts), "operand" = BlockScaledOperand with per_tensor_scale


# ------------------------------------------------------------------------------------------------ nv4 GEMM (TN, K-major operands, blocked scales)
def _f4(q: torch.Tensor) -> torch.Tensor:
    return q.view(torch.float4_e2m1fn_x2)


def nv4_gemm(qa, sfa, tsa, qb, sfb, tsb, out, impl=None):
    """``out[M, N] = A[M, K] . B[N, K]^T`` with A = (qa [M, K/2], sfa, tsa), B = (qb [N, K/2], sfb, tsb); out bf16 or fp32; alpha = tsa * tsb on the device."""
    impl = impl or GEMM_IMPL
    if impl == "operand":
        C.quack_fp4(C.operand((qa, sfa, tsa), "nv4"), C.operand((qb, sfb, tsb), "nv4"), out)
    else:
        fp4_epi.nv4_epilogues()["plain"](_f4(qa), _f4(qb).t(), out={"D": out}, ts=(tsa * tsb).reshape(1), **C.epi_kw(sfa, sfb, "nv4"))
    return out


# ------------------------------------------------------------------------------------------------ containers
class WQ:
    """Per-step copies of one ConvT weight ``wm [4C, C]``: bf16 matrix, mx8 (row for fwd / col for dgrad), nv4 (q for fwd, qt = W^T for dgrad)."""
    __slots__ = ("wm", "w8", "w4")

    def __init__(self):
        self.wm = self.w8 = self.w4 = None


class XQ:
    __slots__ = ("x8", "x4")

    def __init__(self):
        self.x8 = self.x4 = None


# ------------------------------------------------------------------------------------------------ handle
class Handle:
    """Owns the policy, the FP4 state (delayed amax / SR seed / saturation monitor) and the per-step weight copies.  ``ctx.meta['round21_fp4_quant_head']``."""

    N_CONVT = 2

    def __init__(self, ctx, variant: str = H3, *, dy_headroom: float = 8.0, x_headroom: float = 2.0, w_headroom: float = 2.0, impl: str | None = None):
        self.n_calls = int(ctx.model.num_joint_blocks) + 1       # predicts per forward: one before each joint block + the final one
        self.dy_headroom, self.x_headroom, self.w_headroom = float(dy_headroom), float(x_headroom), float(w_headroom)
        self.impl = impl
        self.device = next(ctx.model.parameters()).device if hasattr(ctx.model, "parameters") else torch.device("cuda")
        self.state = fp4q.FP4State(self.n_calls * 2 * self.N_CONVT + self.N_CONVT, device=self.device)
        self.calls = 0
        self.wq: list = []
        self.set_policy(variant)

    # ---- sites ----
    def x_site(self, call: int, j: int) -> int:
        return 4 * call + 2 * j

    def dz_site(self, call: int, j: int) -> int:
        return 4 * call + 2 * j + 1

    def w_site(self, j: int) -> int:
        return 4 * self.n_calls + j

    # ---- policy / state ----
    def set_policy(self, text: str) -> None:
        """Switch the policy at run time (clears the delayed-amax table, the SR counter and the per-step weight copies)."""
        self.policy = P.parse(text)
        self.spec = self.policy.spec("head")
        s = self.spec
        for f in (s.ff, s.fd, s.fw):
            if f not in ("bf16", "mx8", "nv4"):
                raise NotImplementedError(f"round21b head: format {f!r} is not implemented (bf16 / mx8 / nv4)")
        if s.rht_f or s.rht_d or not s.w2d_f or not s.w2d_d:
            raise NotImplementedError("round21b head: rht on f/d and w1d are not implemented (rht on w is)")
        self.amax = self.policy.amax
        self.reset_state(full=True)

    def reset_state(self, full: bool = True) -> None:
        """full: forget the delayed-amax history (the next call of every site re-initialises with an exact amax).  not full: keep ``inited`` and the previous amax,
        clear the current-step slot and restart the SR counter (used by graph_equiv after the eager warm-up)."""
        st = self.state
        if full:
            st.reset()
        else:
            st.amax[1].zero_()
            st.step.zero_()
            st.reset_stats()
        st.seed.fill_(20251003)
        self.calls = 0
        self.wq = []

    def end_step(self) -> None:
        """After the optimizer (patch wrapper): roll the amax state once per step and restart the predict counter."""
        self.state.roll()
        self.calls = 0

    def next_call(self) -> int:
        c = self.calls % self.n_calls
        self.calls += 1
        return c

    # ---- per-step weight copies ----
    def refresh_weights(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        """Called by the first predict of every forward (so that capture / eager warm-up / plain eager steps all re-quantize exactly once per step)."""
        s = self.spec
        out = []
        for j, w in enumerate((w1, w2)):
            wm = h14._convt_to_gemm(w)
            wq = WQ()
            if s.ff == "bf16" or s.fd == "bf16":
                wq.wm = wm
            f8, d8 = s.ff == "mx8", s.fd == "mx8"
            if f8 or d8:
                wq.w8 = mx8.quantize(wm, row=f8, col=d8)
            if s.ff == "nv4" or s.fd == "nv4":
                wq.w4 = fp4q.quant_weight(wm, "nv4", amax=self.amax, state=self.state, site=self.w_site(j), headroom=self.w_headroom)
            out.append(wq)
        self.wq = out


# ------------------------------------------------------------------------------------------------ one ConvT layer
class _ConvT4:
    """One ConvT(k2s2)-as-GEMM layer: Z = X Wm^T, dX = dZ Wm, dWm = dZ^T X with per-kind formats."""

    def __init__(self, h: Handle, j: int, call: int, x: torch.Tensor):
        self.h, self.j, self.call, self.x = h, j, call, x
        self.wq: WQ = h.wq[j]
        self.xq = XQ()

    def fwd(self) -> torch.Tensor:
        h, s, wq, xq, x = self.h, self.h.spec, self.wq, self.xq, self.x
        f8, w8 = s.ff == "mx8", s.fw == "mx8"
        if f8 or w8:
            xq.x8 = mx8.quantize(x, row=f8, col=w8)
        f4, w4 = s.ff == "nv4", s.fw == "nv4"
        if f4 or w4:
            xq.x4 = fp4q.quant_act(x, "nv4", row=f4, col=w4, rht_col=bool(w4 and s.rht_w), amax=h.amax, state=h.state, site=h.x_site(self.call, self.j), headroom=h.x_headroom)
        if s.ff == "bf16":
            z = h14.mm_fwd(x, wq.wm)
        elif f8:
            z = G8.fwd_bias(xq.x8, wq.w8, None)
        else:
            z = torch.empty(x.shape[0], wq.w4.rows, device=x.device, dtype=torch.bfloat16)
            nv4_gemm(xq.x4.q, xq.x4.sf, xq.x4.ts, wq.w4.q, wq.w4.sf, wq.w4.ts, z, h.impl)
            xq.x4.q = xq.x4.sf = None
        if f8:  # the row copy is spent; keep (only) the column copy for the mx8 wgrad
            xq.x8.qr = xq.x8.sfr = None
            if not w8:
                xq.x8 = None
        if s.fw != "bf16":
            self.x = None   # the wgrad works from the quantized transposed copy only (frees the bf16 convT input, e.g. y1 [4M, C])
        return z

    def bwd(self, dz: torch.Tensor):
        """-> (dWm fp32 [4C, C], dX bf16 [M, C])."""
        h, s, wq, xq = self.h, self.h.spec, self.wq, self.xq
        d8, w8 = s.fd == "mx8", s.fw == "mx8"
        d4, w4 = s.fd == "nv4", s.fw == "nv4"
        dq8 = mx8.quantize(dz, row=d8, col=w8) if (d8 or w8) else None
        dq4 = None
        if d4 or w4:
            site = h.dz_site(self.call, self.j)
            dq4 = fp4q.quant_act(dz, "nv4", row=d4, col=w4, sr_row=bool(d4 and s.sr_d), sr_col=bool(w4 and s.sr_w), rht_col=bool(w4 and s.rht_w), amax=h.amax, state=h.state,
                                 site=site, sr_offset=site, headroom=h.dy_headroom)
        # dgrad: dX [M, C] = dZ [M, 4C] Wm
        if s.fd == "bf16":
            dx = h14.mm_dgrad(dz, wq.wm)
        elif d8:
            dx = G8.dgrad(dq8, wq.w8)
        else:
            dx = torch.empty(dz.shape[0], wq.w4.cols, device=dz.device, dtype=torch.bfloat16)
            nv4_gemm(dq4.q, dq4.sf, dq4.ts, wq.w4.qt, wq.w4.sft, wq.w4.tst, dx, h.impl)
        # wgrad: dWm [4C, C] fp32 = dZ^T X over the tokens
        if s.fw == "bf16":
            dwm = h14.mm_wgrad(dz, self.x)
        elif w8:
            dwm = G8.wgrad(dq8, xq.x8)
        else:
            assert dq4.kt == xq.x4.kt, (dq4.kt, xq.x4.kt)
            dwm = torch.empty(dz.shape[1], xq.x4.cols, device=dz.device, dtype=torch.float32)
            nv4_gemm(dq4.qt, dq4.sft, dq4.tst, xq.x4.qt, xq.x4.sft, xq.x4.tst, dwm, h.impl)
        self.xq = self.wq = self.x = None
        return dwm, dx


# ------------------------------------------------------------------------------------------------ the fused upscale Function
class _Upscale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tok, e, w1, b1, d1, g1, be1, w2, b2, d2, g2, be2, eps1, eps2, grid, handle, nvtx=False):
        if nvtx:
            torch.cuda.nvtx.range_push("r21_up_fwd")
        ctx.nvtx = nvtx
        B, T, C = tok.shape
        h1 = grid
        h2 = 2 * grid
        with torch.autocast("cuda", enabled=False):
            call = handle.next_call()
            if call == 0:
                handle.refresh_weights(w1, w2)
            tok2d = tok.reshape(B * T, C)
            dwt1, dwt2 = h14._dw_taps(d1), h14._dw_taps(d2)
            L1 = _ConvT4(handle, 0, call, tok2d)
            z1 = L1.fwd()
            y1, m1, r1 = ht.fwd(z1, b1, dwt1, g1, be1, B, h1, h1, eps1)
            L2 = _ConvT4(handle, 1, call, y1)
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
            torch.cuda.nvtx.range_push("r21_up_bwd")
        tok, e, dwt1, dwt2, b1, b2, g1, g2, z1, m1, r1, z2, y2, m2, r2 = ctx.saved_tensors
        L1, L2 = ctx.layers
        ctx.layers = None
        B, C, h1, h2, hw, _ = ctx.dims
        with torch.autocast("cuda", enabled=False):
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
