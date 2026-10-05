"""round14 fused head kernels (Triton): GELU -> depthwise 3x3 -> LayerNorm2D, forward and backward, for the EoMT ScaleBlock.

Layouts (all channels-last, C contiguous):
  Z   [B*h*w, 4C] bf16   raw ConvTranspose(k=2,s=2) GEMM output WITHOUT bias, "pixel-shuffle on read":
                         output pixel (b,P,Q) (P<2h, Q<2w) lives at row (b*h + P//2)*w + Q//2, columns ((P%2)*2 + Q%2)*C .. +C
  Y   [B*H*W, C]  bf16   LayerNorm2D output, H=2h, W=2w, normal NHWC tokens (input of the next GEMM / the einsum)

Every kernel streams one image row (b, P) along Q with a 3x3 register window, so each input element is read once per
output row that needs it (3x, L2-resident) and GELU/bias are applied to the loaded values only.
  fwd : u = dwconv3x3(gelu(Z+bias)) ; y = LN_C(u) * gamma + beta ; saves mean / rstd per pixel
  bwd1: recomputes u (stencil over gelu(Z+bias)), LayerNorm backward with dY -> dU (bf16), per-program dgamma/dbeta partials
  bwd2: stencil over dU -> dh ; dZ = dh * gelu'(Z+bias) written in the Z layout ; per-program dW(9 taps) and dbias partials
Reductions are fp32 and deterministic (per-program partials reduced by torch .sum(0)).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_SQRT1_2 = 0.7071067811865476
_INV_SQRT_2PI = 0.3989422804014327


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + libdevice.erf(x * 0.7071067811865476))


@triton.jit
def _zoff(b, P, Q, h: tl.constexpr, w: tl.constexpr, C: tl.constexpr):
    return ((b * h + (P >> 1)) * w + (Q >> 1)) * (4 * C) + (((P & 1) << 1) | (Q & 1)) * C


@triton.jit
def _ld_g(Z, bias, b, P, Q, c, cm, h: tl.constexpr, w: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr):
    ok = (P >= 0) & (P < H) & (Q >= 0) & (Q < W)
    off = _zoff(b, P, Q, h, w, C)
    z = tl.load(Z + off + c, mask=cm & ok, other=0.0).to(tl.float32)
    return tl.where(cm & ok, _gelu(z + bias), 0.0)


@triton.jit
def _ld_du(DU, b, P, Q, c, cm, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr):
    ok = (P >= 0) & (P < H) & (Q >= 0) & (Q < W)
    return tl.load(DU + ((b * H + P) * W + Q) * C + c, mask=cm & ok, other=0.0).to(tl.float32)


@triton.jit
def _ldw(DWT, t: tl.constexpr, c, cm, C: tl.constexpr):
    return tl.load(DWT + t * C + c, mask=cm, other=0.0)


@triton.jit
def _stencil(w0, w1, w2, w3, w4, w5, w6, w7, w8, a0, a1, a2, b0, b1, b2, c0, c1, c2):
    # taps (ky,kx) -> w[ky*3+kx]; window columns a=left(Q-1) b=mid(Q) c=right(Q+1), rows 0..2 = P-1..P+1
    return (w0 * a0 + w1 * b0 + w2 * c0 + w3 * a1 + w4 * b1 + w5 * c1 + w6 * a2 + w7 * b2 + w8 * c2)


@triton.jit
def _fwd_kernel(Z, BIAS, DWT, GAM, BET, Y, MEAN, RSTD, eps,
                h: tl.constexpr, w: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, PW: tl.constexpr):
    H: tl.constexpr = 2 * h
    W: tl.constexpr = 2 * w
    pq = tl.program_id(0)
    r = tl.program_id(1)
    b = r // H
    P = r % H
    c = tl.arange(0, CP)
    cm = c < C
    bias = tl.load(BIAS + c, mask=cm, other=0.0)
    gam = tl.load(GAM + c, mask=cm, other=0.0)
    bet = tl.load(BET + c, mask=cm, other=0.0)
    w0 = _ldw(DWT, 0, c, cm, C); w1 = _ldw(DWT, 1, c, cm, C); w2 = _ldw(DWT, 2, c, cm, C)
    w3 = _ldw(DWT, 3, c, cm, C); w4 = _ldw(DWT, 4, c, cm, C); w5 = _ldw(DWT, 5, c, cm, C)
    w6 = _ldw(DWT, 6, c, cm, C); w7 = _ldw(DWT, 7, c, cm, C); w8 = _ldw(DWT, 8, c, cm, C)
    q0 = pq * PW
    a0 = _ld_g(Z, bias, b, P - 1, q0 - 1, c, cm, h, w, H, W, C)
    a1 = _ld_g(Z, bias, b, P, q0 - 1, c, cm, h, w, H, W, C)
    a2 = _ld_g(Z, bias, b, P + 1, q0 - 1, c, cm, h, w, H, W, C)
    b0 = _ld_g(Z, bias, b, P - 1, q0, c, cm, h, w, H, W, C)
    b1 = _ld_g(Z, bias, b, P, q0, c, cm, h, w, H, W, C)
    b2 = _ld_g(Z, bias, b, P + 1, q0, c, cm, h, w, H, W, C)
    for q in range(q0, q0 + PW):
        c0 = _ld_g(Z, bias, b, P - 1, q + 1, c, cm, h, w, H, W, C)
        c1 = _ld_g(Z, bias, b, P, q + 1, c, cm, h, w, H, W, C)
        c2 = _ld_g(Z, bias, b, P + 1, q + 1, c, cm, h, w, H, W, C)
        u = _stencil(w0, w1, w2, w3, w4, w5, w6, w7, w8, a0, a1, a2, b0, b1, b2, c0, c1, c2)
        mean = tl.sum(u, 0) / C
        d = tl.where(cm, u - mean, 0.0)
        rstd = libdevice.rsqrt(tl.sum(d * d, 0) / C + eps)
        y = d * rstd * gam + bet
        pix = (b * H + P) * W + q
        tl.store(Y + pix * C + c, y.to(tl.bfloat16), mask=cm)
        tl.store(MEAN + pix, mean)
        tl.store(RSTD + pix, rstd)
        a0 = b0; a1 = b1; a2 = b2
        b0 = c0; b1 = c1; b2 = c2


@triton.jit
def _bwd1_kernel(Z, BIAS, DWT, GAM, MEAN, RSTD, DY, DU, PART,
                 h: tl.constexpr, w: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, PW: tl.constexpr):
    H: tl.constexpr = 2 * h
    W: tl.constexpr = 2 * w
    pq = tl.program_id(0)
    r = tl.program_id(1)
    b = r // H
    P = r % H
    c = tl.arange(0, CP)
    cm = c < C
    bias = tl.load(BIAS + c, mask=cm, other=0.0)
    gam = tl.load(GAM + c, mask=cm, other=0.0)
    w0 = _ldw(DWT, 0, c, cm, C); w1 = _ldw(DWT, 1, c, cm, C); w2 = _ldw(DWT, 2, c, cm, C)
    w3 = _ldw(DWT, 3, c, cm, C); w4 = _ldw(DWT, 4, c, cm, C); w5 = _ldw(DWT, 5, c, cm, C)
    w6 = _ldw(DWT, 6, c, cm, C); w7 = _ldw(DWT, 7, c, cm, C); w8 = _ldw(DWT, 8, c, cm, C)
    q0 = pq * PW
    a0 = _ld_g(Z, bias, b, P - 1, q0 - 1, c, cm, h, w, H, W, C)
    a1 = _ld_g(Z, bias, b, P, q0 - 1, c, cm, h, w, H, W, C)
    a2 = _ld_g(Z, bias, b, P + 1, q0 - 1, c, cm, h, w, H, W, C)
    b0 = _ld_g(Z, bias, b, P - 1, q0, c, cm, h, w, H, W, C)
    b1 = _ld_g(Z, bias, b, P, q0, c, cm, h, w, H, W, C)
    b2 = _ld_g(Z, bias, b, P + 1, q0, c, cm, h, w, H, W, C)
    acc_g = tl.zeros([CP], dtype=tl.float32)
    acc_b = tl.zeros([CP], dtype=tl.float32)
    for q in range(q0, q0 + PW):
        c0 = _ld_g(Z, bias, b, P - 1, q + 1, c, cm, h, w, H, W, C)
        c1 = _ld_g(Z, bias, b, P, q + 1, c, cm, h, w, H, W, C)
        c2 = _ld_g(Z, bias, b, P + 1, q + 1, c, cm, h, w, H, W, C)
        u = _stencil(w0, w1, w2, w3, w4, w5, w6, w7, w8, a0, a1, a2, b0, b1, b2, c0, c1, c2)
        pix = (b * H + P) * W + q
        mean = tl.load(MEAN + pix)
        rstd = tl.load(RSTD + pix)
        xh = tl.where(cm, (u - mean) * rstd, 0.0)
        dy = tl.load(DY + pix * C + c, mask=cm, other=0.0).to(tl.float32)
        a = dy * gam
        s1 = tl.sum(a, 0) / C
        s2 = tl.sum(a * xh, 0) / C
        du = rstd * (a - s1 - xh * s2)
        tl.store(DU + pix * C + c, du.to(tl.bfloat16), mask=cm)
        acc_g += dy * xh
        acc_b += dy
        a0 = b0; a1 = b1; a2 = b2
        b0 = c0; b1 = c1; b2 = c2
    pid = r * tl.num_programs(0) + pq
    tl.store(PART + (pid * 2) * C + c, acc_g, mask=cm)
    tl.store(PART + (pid * 2 + 1) * C + c, acc_b, mask=cm)


@triton.jit
def _bwd2_kernel(Z, BIAS, DWT, DU, DZ, PW_DW, PW_DB,
                 h: tl.constexpr, w: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, PW: tl.constexpr):
    H: tl.constexpr = 2 * h
    W: tl.constexpr = 2 * w
    pq = tl.program_id(0)
    r = tl.program_id(1)
    b = r // H
    P = r % H
    c = tl.arange(0, CP)
    cm = c < C
    bias = tl.load(BIAS + c, mask=cm, other=0.0)
    w0 = _ldw(DWT, 0, c, cm, C); w1 = _ldw(DWT, 1, c, cm, C); w2 = _ldw(DWT, 2, c, cm, C)
    w3 = _ldw(DWT, 3, c, cm, C); w4 = _ldw(DWT, 4, c, cm, C); w5 = _ldw(DWT, 5, c, cm, C)
    w6 = _ldw(DWT, 6, c, cm, C); w7 = _ldw(DWT, 7, c, cm, C); w8 = _ldw(DWT, 8, c, cm, C)
    q0 = pq * PW
    # window of dU: rows r=0,1,2 <-> P-1,P,P+1 ; columns a,b,c <-> Q-1,Q,Q+1
    a0 = _ld_du(DU, b, P - 1, q0 - 1, c, cm, H, W, C)
    a1 = _ld_du(DU, b, P, q0 - 1, c, cm, H, W, C)
    a2 = _ld_du(DU, b, P + 1, q0 - 1, c, cm, H, W, C)
    b0 = _ld_du(DU, b, P - 1, q0, c, cm, H, W, C)
    b1 = _ld_du(DU, b, P, q0, c, cm, H, W, C)
    b2 = _ld_du(DU, b, P + 1, q0, c, cm, H, W, C)
    d0 = tl.zeros([CP], dtype=tl.float32); d1 = tl.zeros([CP], dtype=tl.float32); d2 = tl.zeros([CP], dtype=tl.float32)
    d3 = tl.zeros([CP], dtype=tl.float32); d4 = tl.zeros([CP], dtype=tl.float32); d5 = tl.zeros([CP], dtype=tl.float32)
    d6 = tl.zeros([CP], dtype=tl.float32); d7 = tl.zeros([CP], dtype=tl.float32); d8 = tl.zeros([CP], dtype=tl.float32)
    acc_b = tl.zeros([CP], dtype=tl.float32)
    for q in range(q0, q0 + PW):
        c0 = _ld_du(DU, b, P - 1, q + 1, c, cm, H, W, C)
        c1 = _ld_du(DU, b, P, q + 1, c, cm, H, W, C)
        c2 = _ld_du(DU, b, P + 1, q + 1, c, cm, H, W, C)
        # dh = sum_{ky,kx} w[ky,kx] * dU[P+1-ky, Q+1-kx]  -> window row (2-ky), column (2-kx)
        dh = _stencil(w8, w7, w6, w5, w4, w3, w2, w1, w0, a0, a1, a2, b0, b1, b2, c0, c1, c2)
        off = _zoff(b, P, q, h, w, C)
        z = tl.load(Z + off + c, mask=cm, other=0.0).to(tl.float32) + bias
        cdf = 0.5 * (1.0 + libdevice.erf(z * 0.7071067811865476))
        g = z * cdf
        dgelu = cdf + z * 0.3989422804014327 * tl.exp(-0.5 * z * z)
        dz = dh * dgelu
        tl.store(DZ + off + c, dz.to(tl.bfloat16), mask=cm)
        acc_b += dz
        # dW[ky,kx] += g * dU[P+1-ky, Q+1-kx]
        d0 += g * c2; d1 += g * b2; d2 += g * a2
        d3 += g * c1; d4 += g * b1; d5 += g * a1
        d6 += g * c0; d7 += g * b0; d8 += g * a0
        a0 = b0; a1 = b1; a2 = b2
        b0 = c0; b1 = c1; b2 = c2
    pid = r * tl.num_programs(0) + pq
    base = PW_DW + (pid * 9) * C
    tl.store(base + 0 * C + c, d0, mask=cm); tl.store(base + 1 * C + c, d1, mask=cm); tl.store(base + 2 * C + c, d2, mask=cm)
    tl.store(base + 3 * C + c, d3, mask=cm); tl.store(base + 4 * C + c, d4, mask=cm); tl.store(base + 5 * C + c, d5, mask=cm)
    tl.store(base + 6 * C + c, d6, mask=cm); tl.store(base + 7 * C + c, d7, mask=cm); tl.store(base + 8 * C + c, d8, mask=cm)
    tl.store(PW_DB + pid * C + c, acc_b, mask=cm)


# ------------------------------------------------------------------------------------------------ v2: prefetched loads, fused reductions
@triton.jit
def _add2(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@triton.jit
def _ld_raw(Z, b, P, Q, c, cm, h: tl.constexpr, w: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr):
    ok = (P >= 0) & (P < H) & (Q >= 0) & (Q < W)
    off = _zoff(b, P, Q, h, w, C)
    return tl.load(Z + off + c, mask=cm & ok, other=0.0)


@triton.jit
def _g_of(raw, bias, P, Q, cm, H: tl.constexpr, W: tl.constexpr):
    ok = (P >= 0) & (P < H) & (Q >= 0) & (Q < W)
    return tl.where(cm & ok, _gelu(raw.to(tl.float32) + bias), 0.0)


@triton.jit
def _fwd_kernel2(Z, BIAS, DWT, GAM, BET, Y, MEAN, RSTD, eps,
                 h: tl.constexpr, w: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, PW: tl.constexpr):
    H: tl.constexpr = 2 * h
    W: tl.constexpr = 2 * w
    pq = tl.program_id(0)
    r = tl.program_id(1)
    b = r // H
    P = r % H
    c = tl.arange(0, CP)
    cm = c < C
    bias = tl.load(BIAS + c, mask=cm, other=0.0)
    gam = tl.load(GAM + c, mask=cm, other=0.0)
    bet = tl.load(BET + c, mask=cm, other=0.0)
    w0 = _ldw(DWT, 0, c, cm, C); w1 = _ldw(DWT, 1, c, cm, C); w2 = _ldw(DWT, 2, c, cm, C)
    w3 = _ldw(DWT, 3, c, cm, C); w4 = _ldw(DWT, 4, c, cm, C); w5 = _ldw(DWT, 5, c, cm, C)
    w6 = _ldw(DWT, 6, c, cm, C); w7 = _ldw(DWT, 7, c, cm, C); w8 = _ldw(DWT, 8, c, cm, C)
    q0 = pq * PW
    a0 = _g_of(_ld_raw(Z, b, P - 1, q0 - 1, c, cm, h, w, H, W, C), bias, P - 1, q0 - 1, cm, H, W)
    a1 = _g_of(_ld_raw(Z, b, P, q0 - 1, c, cm, h, w, H, W, C), bias, P, q0 - 1, cm, H, W)
    a2 = _g_of(_ld_raw(Z, b, P + 1, q0 - 1, c, cm, h, w, H, W, C), bias, P + 1, q0 - 1, cm, H, W)
    b0 = _g_of(_ld_raw(Z, b, P - 1, q0, c, cm, h, w, H, W, C), bias, P - 1, q0, cm, H, W)
    b1 = _g_of(_ld_raw(Z, b, P, q0, c, cm, h, w, H, W, C), bias, P, q0, cm, H, W)
    b2 = _g_of(_ld_raw(Z, b, P + 1, q0, c, cm, h, w, H, W, C), bias, P + 1, q0, cm, H, W)
    r0 = _ld_raw(Z, b, P - 1, q0 + 1, c, cm, h, w, H, W, C)
    r1 = _ld_raw(Z, b, P, q0 + 1, c, cm, h, w, H, W, C)
    r2 = _ld_raw(Z, b, P + 1, q0 + 1, c, cm, h, w, H, W, C)
    pivot = eps * 0.0
    for q in range(q0, q0 + PW):
        n0 = _ld_raw(Z, b, P - 1, q + 2, c, cm, h, w, H, W, C)
        n1 = _ld_raw(Z, b, P, q + 2, c, cm, h, w, H, W, C)
        n2 = _ld_raw(Z, b, P + 1, q + 2, c, cm, h, w, H, W, C)
        c0 = _g_of(r0, bias, P - 1, q + 1, cm, H, W)
        c1 = _g_of(r1, bias, P, q + 1, cm, H, W)
        c2 = _g_of(r2, bias, P + 1, q + 1, cm, H, W)
        u = _stencil(w0, w1, w2, w3, w4, w5, w6, w7, w8, a0, a1, a2, b0, b1, b2, c0, c1, c2)
        ud = tl.where(cm, u - pivot, 0.0)
        s1, s2 = tl.reduce((ud, ud * ud), 0, _add2)
        m1 = s1 / C
        mean = pivot + m1
        var = tl.maximum(s2 / C - m1 * m1, 0.0)
        rstd = libdevice.rsqrt(var + eps)
        y = tl.where(cm, (ud - m1) * rstd * gam + bet, 0.0)
        pix = (b * H + P) * W + q
        tl.store(Y + pix * C + c, y.to(tl.bfloat16), mask=cm)
        tl.store(MEAN + pix, mean)
        tl.store(RSTD + pix, rstd)
        pivot = mean
        a0 = b0; a1 = b1; a2 = b2
        b0 = c0; b1 = c1; b2 = c2
        r0 = n0; r1 = n1; r2 = n2


@triton.jit
def _bwd1_kernel2(Z, BIAS, DWT, GAM, MEAN, RSTD, DY, DU, PART,
                  h: tl.constexpr, w: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, PW: tl.constexpr):
    H: tl.constexpr = 2 * h
    W: tl.constexpr = 2 * w
    pq = tl.program_id(0)
    r = tl.program_id(1)
    b = r // H
    P = r % H
    c = tl.arange(0, CP)
    cm = c < C
    bias = tl.load(BIAS + c, mask=cm, other=0.0)
    gam = tl.load(GAM + c, mask=cm, other=0.0)
    w0 = _ldw(DWT, 0, c, cm, C); w1 = _ldw(DWT, 1, c, cm, C); w2 = _ldw(DWT, 2, c, cm, C)
    w3 = _ldw(DWT, 3, c, cm, C); w4 = _ldw(DWT, 4, c, cm, C); w5 = _ldw(DWT, 5, c, cm, C)
    w6 = _ldw(DWT, 6, c, cm, C); w7 = _ldw(DWT, 7, c, cm, C); w8 = _ldw(DWT, 8, c, cm, C)
    q0 = pq * PW
    a0 = _g_of(_ld_raw(Z, b, P - 1, q0 - 1, c, cm, h, w, H, W, C), bias, P - 1, q0 - 1, cm, H, W)
    a1 = _g_of(_ld_raw(Z, b, P, q0 - 1, c, cm, h, w, H, W, C), bias, P, q0 - 1, cm, H, W)
    a2 = _g_of(_ld_raw(Z, b, P + 1, q0 - 1, c, cm, h, w, H, W, C), bias, P + 1, q0 - 1, cm, H, W)
    b0 = _g_of(_ld_raw(Z, b, P - 1, q0, c, cm, h, w, H, W, C), bias, P - 1, q0, cm, H, W)
    b1 = _g_of(_ld_raw(Z, b, P, q0, c, cm, h, w, H, W, C), bias, P, q0, cm, H, W)
    b2 = _g_of(_ld_raw(Z, b, P + 1, q0, c, cm, h, w, H, W, C), bias, P + 1, q0, cm, H, W)
    r0 = _ld_raw(Z, b, P - 1, q0 + 1, c, cm, h, w, H, W, C)
    r1 = _ld_raw(Z, b, P, q0 + 1, c, cm, h, w, H, W, C)
    r2 = _ld_raw(Z, b, P + 1, q0 + 1, c, cm, h, w, H, W, C)
    pix0 = (b * H + P) * W + q0
    dyn = tl.load(DY + pix0 * C + c, mask=cm, other=0.0)
    acc_g = tl.zeros([CP], dtype=tl.float32)
    acc_b = tl.zeros([CP], dtype=tl.float32)
    for q in range(q0, q0 + PW):
        n0 = _ld_raw(Z, b, P - 1, q + 2, c, cm, h, w, H, W, C)
        n1 = _ld_raw(Z, b, P, q + 2, c, cm, h, w, H, W, C)
        n2 = _ld_raw(Z, b, P + 1, q + 2, c, cm, h, w, H, W, C)
        pix = (b * H + P) * W + q
        dyn_next = tl.load(DY + (pix + 1) * C + c, mask=cm & (q + 1 < q0 + PW), other=0.0)
        c0 = _g_of(r0, bias, P - 1, q + 1, cm, H, W)
        c1 = _g_of(r1, bias, P, q + 1, cm, H, W)
        c2 = _g_of(r2, bias, P + 1, q + 1, cm, H, W)
        u = _stencil(w0, w1, w2, w3, w4, w5, w6, w7, w8, a0, a1, a2, b0, b1, b2, c0, c1, c2)
        mean = tl.load(MEAN + pix)
        rstd = tl.load(RSTD + pix)
        xh = tl.where(cm, (u - mean) * rstd, 0.0)
        dy = dyn.to(tl.float32)
        a = dy * gam
        s1, s2 = tl.reduce((a, a * xh), 0, _add2)
        s1 = s1 / C
        s2 = s2 / C
        du = rstd * (a - s1 - xh * s2)
        tl.store(DU + pix * C + c, du.to(tl.bfloat16), mask=cm)
        acc_g += dy * xh
        acc_b += dy
        a0 = b0; a1 = b1; a2 = b2
        b0 = c0; b1 = c1; b2 = c2
        r0 = n0; r1 = n1; r2 = n2
        dyn = dyn_next
    pid = r * tl.num_programs(0) + pq
    tl.store(PART + (pid * 2) * C + c, acc_g, mask=cm)
    tl.store(PART + (pid * 2 + 1) * C + c, acc_b, mask=cm)


@triton.jit
def _ld_du_raw(DU, b, P, Q, c, cm, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr):
    ok = (P >= 0) & (P < H) & (Q >= 0) & (Q < W)
    return tl.load(DU + ((b * H + P) * W + Q) * C + c, mask=cm & ok, other=0.0)


@triton.jit
def _bwd2_kernel2(Z, BIAS, DWT, DU, DZ, PW_DW, PW_DB,
                  h: tl.constexpr, w: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, PW: tl.constexpr):
    H: tl.constexpr = 2 * h
    W: tl.constexpr = 2 * w
    pq = tl.program_id(0)
    r = tl.program_id(1)
    b = r // H
    P = r % H
    c = tl.arange(0, CP)
    cm = c < C
    bias = tl.load(BIAS + c, mask=cm, other=0.0)
    w0 = _ldw(DWT, 0, c, cm, C); w1 = _ldw(DWT, 1, c, cm, C); w2 = _ldw(DWT, 2, c, cm, C)
    w3 = _ldw(DWT, 3, c, cm, C); w4 = _ldw(DWT, 4, c, cm, C); w5 = _ldw(DWT, 5, c, cm, C)
    w6 = _ldw(DWT, 6, c, cm, C); w7 = _ldw(DWT, 7, c, cm, C); w8 = _ldw(DWT, 8, c, cm, C)
    q0 = pq * PW
    a0 = _ld_du(DU, b, P - 1, q0 - 1, c, cm, H, W, C)
    a1 = _ld_du(DU, b, P, q0 - 1, c, cm, H, W, C)
    a2 = _ld_du(DU, b, P + 1, q0 - 1, c, cm, H, W, C)
    b0 = _ld_du(DU, b, P - 1, q0, c, cm, H, W, C)
    b1 = _ld_du(DU, b, P, q0, c, cm, H, W, C)
    b2 = _ld_du(DU, b, P + 1, q0, c, cm, H, W, C)
    r0 = _ld_du_raw(DU, b, P - 1, q0 + 1, c, cm, H, W, C)
    r1 = _ld_du_raw(DU, b, P, q0 + 1, c, cm, H, W, C)
    r2 = _ld_du_raw(DU, b, P + 1, q0 + 1, c, cm, H, W, C)
    off0 = _zoff(b, P, q0, h, w, C)
    zn = tl.load(Z + off0 + c, mask=cm, other=0.0)
    d0 = tl.zeros([CP], dtype=tl.float32); d1 = tl.zeros([CP], dtype=tl.float32); d2 = tl.zeros([CP], dtype=tl.float32)
    d3 = tl.zeros([CP], dtype=tl.float32); d4 = tl.zeros([CP], dtype=tl.float32); d5 = tl.zeros([CP], dtype=tl.float32)
    d6 = tl.zeros([CP], dtype=tl.float32); d7 = tl.zeros([CP], dtype=tl.float32); d8 = tl.zeros([CP], dtype=tl.float32)
    acc_b = tl.zeros([CP], dtype=tl.float32)
    for q in range(q0, q0 + PW):
        n0 = _ld_du_raw(DU, b, P - 1, q + 2, c, cm, H, W, C)
        n1 = _ld_du_raw(DU, b, P, q + 2, c, cm, H, W, C)
        n2 = _ld_du_raw(DU, b, P + 1, q + 2, c, cm, H, W, C)
        offn = _zoff(b, P, q + 1, h, w, C)
        zn_next = tl.load(Z + offn + c, mask=cm & (q + 1 < q0 + PW), other=0.0)
        c0 = r0.to(tl.float32)
        c1 = r1.to(tl.float32)
        c2 = r2.to(tl.float32)
        dh = _stencil(w8, w7, w6, w5, w4, w3, w2, w1, w0, a0, a1, a2, b0, b1, b2, c0, c1, c2)
        off = _zoff(b, P, q, h, w, C)
        z = zn.to(tl.float32) + bias
        cdf = 0.5 * (1.0 + libdevice.erf(z * 0.7071067811865476))
        g = z * cdf
        dgelu = cdf + z * 0.3989422804014327 * tl.exp(-0.5 * z * z)
        dz = dh * dgelu
        tl.store(DZ + off + c, dz.to(tl.bfloat16), mask=cm)
        acc_b += dz
        d0 += g * c2; d1 += g * b2; d2 += g * a2
        d3 += g * c1; d4 += g * b1; d5 += g * a1
        d6 += g * c0; d7 += g * b0; d8 += g * a0
        a0 = b0; a1 = b1; a2 = b2
        b0 = c0; b1 = c1; b2 = c2
        r0 = n0; r1 = n1; r2 = n2
        zn = zn_next
    pid = r * tl.num_programs(0) + pq
    base = PW_DW + (pid * 9) * C
    tl.store(base + 0 * C + c, d0, mask=cm); tl.store(base + 1 * C + c, d1, mask=cm); tl.store(base + 2 * C + c, d2, mask=cm)
    tl.store(base + 3 * C + c, d3, mask=cm); tl.store(base + 4 * C + c, d4, mask=cm); tl.store(base + 5 * C + c, d5, mask=cm)
    tl.store(base + 6 * C + c, d6, mask=cm); tl.store(base + 7 * C + c, d7, mask=cm); tl.store(base + 8 * C + c, d8, mask=cm)
    tl.store(PW_DB + pid * C + c, acc_b, mask=cm)


# ------------------------------------------------------------------------------------------------ python wrappers
V = 2  # kernel generation (1 = plain, 2 = prefetched loads + fused reductions)


def _cfg(C, num_warps=None):
    cp = triton.next_power_of_2(C)
    return cp, (num_warps or (8 if cp >= 1024 else 4))


def fwd(Z, bias, dwt, gam, bet, B, h, w, eps, PW=None, num_warps=None):
    """Z [B*h*w,4C] bf16; bias [C] fp32; dwt [9,C] fp32; -> (Y [B*H*W,C] bf16, mean [B*H*W], rstd [B*H*W])."""
    C = bias.numel()
    H, W = 2 * h, 2 * w
    PW = PW or W
    cp, nw = _cfg(C, num_warps)
    Y = torch.empty(B * H * W, C, device=Z.device, dtype=torch.bfloat16)
    mean = torch.empty(B * H * W, device=Z.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    (_fwd_kernel2 if V == 2 else _fwd_kernel)[(W // PW, B * H)](Z, bias, dwt, gam, bet, Y, mean, rstd, eps, h, w, C, cp, PW, num_warps=nw)
    return Y, mean, rstd


def bwd1(Z, bias, dwt, gam, mean, rstd, dy, B, h, w, PW=None, num_warps=None):
    """-> (dU [B*H*W,C] bf16, dgamma [C] fp32, dbeta [C] fp32)."""
    C = bias.numel()
    H, W = 2 * h, 2 * w
    PW = PW or W
    cp, nw = _cfg(C, num_warps)
    du = torch.empty(B * H * W, C, device=Z.device, dtype=torch.bfloat16)
    npr = (W // PW) * B * H
    part = torch.empty(npr, 2, C, device=Z.device, dtype=torch.float32)
    (_bwd1_kernel2 if V == 2 else _bwd1_kernel)[(W // PW, B * H)](Z, bias, dwt, gam, mean, rstd, dy, du, part, h, w, C, cp, PW, num_warps=nw)
    s = part.sum(0)
    return du, s[0], s[1]


def bwd2(Z, bias, dwt, du, B, h, w, PW=None, num_warps=None):
    """-> (dZ [B*h*w,4C] bf16, ddw [9,C] fp32, dbias [C] fp32)."""
    C = bias.numel()
    H, W = 2 * h, 2 * w
    PW = PW or W
    cp, nw = _cfg(C, num_warps)
    dz = torch.empty_like(Z)
    npr = (W // PW) * B * H
    pdw = torch.empty(npr, 9, C, device=Z.device, dtype=torch.float32)
    pdb = torch.empty(npr, C, device=Z.device, dtype=torch.float32)
    (_bwd2_kernel2 if V == 2 else _bwd2_kernel)[(W // PW, B * H)](Z, bias, dwt, du, dz, pdw, pdb, h, w, C, cp, PW, num_warps=nw)
    return dz, pdw.sum(0), pdb.sum(0)
