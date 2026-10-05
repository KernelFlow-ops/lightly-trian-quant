"""Deterministic packed-QKV FlashAttention with RoPE and compact EoMT masks.

QKV is [B,N,3,H,64], output [B,N,H*64]. Backward uses two streaming
passes (dQ and dK/dV), never atomic accumulation. All softmax/reductions
are fp32. RoPE tables are the original per-layer random tables, not cached.
"""
from __future__ import annotations

import json
from pathlib import Path
import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
_CFG_PATH=Path(__file__).with_name('tuned_cfg.json')
_TUNED=json.loads(_CFG_PATH.read_text()).get('configs',{}) if _CFG_PATH.exists() else {}

@triton.jit
def _accurate_dot(a,b):
    if a.dtype==tl.float32 and b.dtype==tl.float32:
        return tl.dot(a,b,input_precision='tf32x3')
    return tl.dot(a,b)


@triton.jit
def _load_rot(QKV, SIN, COS, b, h, rows, cols, N: tl.constexpr,
              H: tl.constexpr, WHICH: tl.constexpr, PREFIX: tl.constexpr,
              ROPE: tl.constexpr):
    off = ((b * N + rows[:, None]) * 3 * H + WHICH * H + h) * 64 + cols[None, :]
    raw = tl.load(QKV + off, rows[:, None] < N, 0)
    if ROPE:
        other_cols = (cols + 32) % 64
        off2 = ((b * N + rows[:, None]) * 3 * H + WHICH * H + h) * 64 + other_cols[None, :]
        other = tl.load(QKV + off2, rows[:, None] < N, 0).to(tl.float32)
        r = rows[:, None] - PREFIX
        active = (r >= 0) & (rows[:, None] < N)
        sin = tl.load(SIN + r * 64 + cols[None, :], active, 0)
        cos = tl.load(COS + r * 64 + cols[None, :], active, 1)
        sign = tl.where(cols[None, :] < 32, -1., 1.)
        raw = (raw.to(tl.float32) * cos + other * sign * sin).to(raw.dtype)
    return raw


@triton.jit
def _allowed(MASK, b, rows, keys, N: tl.constexpr, Q: tl.constexpr,
             PREFIX: tl.constexpr, MASKED: tl.constexpr):
    valid = keys[None, :] < N
    if MASKED:
        relevant = (rows[:, None] < Q) & (keys[None, :] >= PREFIX) & valid
        allowed = tl.load(MASK + (b * Q + rows[:, None]) * (N - PREFIX)
                          + keys[None, :] - PREFIX, relevant, 1)
        valid = valid & allowed
    return valid


@triton.jit
def _forward(QKV, SIN, COS, MASK, OUT, LSE, N: tl.constexpr, H: tl.constexpr,
             PREFIX: tl.constexpr, Q: tl.constexpr, ROPE: tl.constexpr,
             MASKED: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    d = tl.arange(0, 64)
    q = _load_rot(QKV, SIN, COS, b, h, m, d, N, H, 0, PREFIX, ROPE)
    maxv = tl.full((BM,), -float('inf'), tl.float32)
    denom = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, 64), 0., tl.float32)
    for start in range(tl.cdiv(N, BN)):
        n = start * BN + tl.arange(0, BN)
        k = _load_rot(QKV, SIN, COS, b, h, n, d, N, H, 1, PREFIX, ROPE)
        v = _load_rot(QKV, SIN, COS, b, h, n, d, N, H, 2, PREFIX, False)
        score = _accurate_dot(q, tl.trans(k)) * (0.125 * LOG2E)
        score = tl.where(_allowed(MASK, b, m, n, N, Q, PREFIX, MASKED), score, -float('inf'))
        newmax = tl.maximum(maxv, tl.max(score, 1))
        alpha = tl.exp2(maxv - newmax)
        p = tl.exp2(score - newmax[:, None])
        acc = acc * alpha[:, None] + _accurate_dot(p.to(v.dtype), v)
        denom = denom * alpha + tl.sum(p, 1)
        maxv = newmax
    out_off = ((b * N + m[:, None]) * H + h) * 64 + d[None, :]
    tl.store(OUT + out_off, acc / denom[:, None], m[:, None] < N)
    tl.store(LSE + bh * N + m, maxv + tl.log2(denom), m < N)


@triton.jit
def _delta(OUT, DO, DELTA, N: tl.constexpr, H: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    d = tl.arange(0, 64)
    off = ((b * N + rows[:, None]) * H + h) * 64 + d[None, :]
    o = tl.load(OUT + off, rows[:, None] < N, 0).to(tl.float32)
    dy = tl.load(DO + off, rows[:, None] < N, 0).to(tl.float32)
    tl.store(DELTA + bh * N + rows, tl.sum(o * dy, 1), rows < N)


@triton.jit
def _store_drot(DQKV, GRAD, SIN, COS, b, h, rows, d, N: tl.constexpr,
                H: tl.constexpr, WHICH: tl.constexpr, PREFIX: tl.constexpr,
                ROPE: tl.constexpr):
    # Match original cast-backward: SDPA gradient rounds to bf16 before
    # the fp32 inverse rotation; the qkv cast then rounds back to bf16.
    g = GRAD.to(DQKV.dtype.element_ty).to(tl.float32)
    if ROPE:
        other = tl.gather(g, tl.broadcast_to(((d + 32) % 64)[None, :], g.shape), 1)
        active = (rows[:, None] >= PREFIX) & (rows[:, None] < N)
        r = rows[:, None] - PREFIX
        sin = tl.load(SIN + r * 64 + d[None, :], active, 0)
        cos = tl.load(COS + r * 64 + d[None, :], active, 1)
        sign = tl.where(d[None, :] < 32, 1., -1.)
        g = g * cos + other * sign * sin
    off = ((b * N + rows[:, None]) * 3 * H + WHICH * H + h) * 64 + d[None, :]
    tl.store(DQKV + off, g, rows[:, None] < N)


@triton.jit
def _backward_q(QKV, SIN, COS, MASK, DO, LSE, DELTA, DQKV,
                N: tl.constexpr, H: tl.constexpr, PREFIX: tl.constexpr,
                Q: tl.constexpr, ROPE: tl.constexpr, MASKED: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    d = tl.arange(0, 64)
    q = _load_rot(QKV, SIN, COS, b, h, m, d, N, H, 0, PREFIX, ROPE)
    off = ((b * N + m[:, None]) * H + h) * 64 + d[None, :]
    dy = tl.load(DO + off, m[:, None] < N, 0)
    lse = tl.load(LSE + bh * N + m, m < N, 0)
    delta = tl.load(DELTA + bh * N + m, m < N, 0)
    dq = tl.full((BM, 64), 0., tl.float32)
    for start in range(tl.cdiv(N, BN)):
        n = start * BN + tl.arange(0, BN)
        k = _load_rot(QKV, SIN, COS, b, h, n, d, N, H, 1, PREFIX, ROPE)
        v = _load_rot(QKV, SIN, COS, b, h, n, d, N, H, 2, PREFIX, False)
        s = _accurate_dot(q, tl.trans(k)) * (0.125 * LOG2E)
        s = tl.where(_allowed(MASK, b, m, n, N, Q, PREFIX, MASKED), s, -float('inf'))
        p = tl.exp2(s - lse[:, None])
        dp = _accurate_dot(dy, tl.trans(v))
        ds = p * (dp - delta[:, None]) * 0.125
        dq += _accurate_dot(ds.to(k.dtype), k)
    _store_drot(DQKV, dq, SIN, COS, b, h, m, d, N, H, 0, PREFIX, ROPE)


@triton.jit
def _backward_kv(QKV, SIN, COS, MASK, DO, LSE, DELTA, DQKV,
                 N: tl.constexpr, H: tl.constexpr, PREFIX: tl.constexpr,
                 Q: tl.constexpr, ROPE: tl.constexpr, MASKED: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    d = tl.arange(0, 64)
    k = _load_rot(QKV, SIN, COS, b, h, n, d, N, H, 1, PREFIX, ROPE)
    v = _load_rot(QKV, SIN, COS, b, h, n, d, N, H, 2, PREFIX, False)
    dk = tl.full((BN, 64), 0., tl.float32)
    dv = tl.full((BN, 64), 0., tl.float32)
    for start in range(tl.cdiv(N, BM)):
        m = start * BM + tl.arange(0, BM)
        q = _load_rot(QKV, SIN, COS, b, h, m, d, N, H, 0, PREFIX, ROPE)
        off = ((b * N + m[:, None]) * H + h) * 64 + d[None, :]
        dy = tl.load(DO + off, m[:, None] < N, 0)
        lse = tl.load(LSE + bh * N + m, m < N, 0)
        delta = tl.load(DELTA + bh * N + m, m < N, 0)
        score = _accurate_dot(q, tl.trans(k)) * (0.125 * LOG2E)
        valid = _allowed(MASK, b, m, n, N, Q, PREFIX, MASKED) & (m[:, None] < N)
        score = tl.where(valid, score, -float('inf'))
        p = tl.exp2(score - lse[:, None])
        dp = _accurate_dot(dy, tl.trans(v))
        ds = p * (dp - delta[:, None]) * 0.125
        dk += _accurate_dot(tl.trans(ds.to(q.dtype)), q)
        dv += _accurate_dot(tl.trans(p.to(dy.dtype)), dy)
    _store_drot(DQKV, dk, SIN, COS, b, h, n, d, N, H, 1, PREFIX, ROPE)
    _store_drot(DQKV, dv, SIN, COS, b, h, n, d, N, H, 2, PREFIX, False)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, sin, cos, compact_mask, prefix, config):
        b, n, _, h, d = qkv.shape
        assert qkv.is_contiguous() and d == 64 and qkv.dtype in (torch.bfloat16,torch.float32)
        assert sin is None or (sin.dtype == torch.float32 and sin.is_contiguous())
        rope, masked = sin is not None, compact_mask is not None
        dummy = qkv if sin is None else sin
        st, ct = dummy, (dummy if cos is None else cos)
        mt = qkv if compact_mask is None else compact_mask
        q = 0 if not masked else compact_mask.shape[1]
        bm, bn = config
        out = torch.empty((b, n, h, d), device=qkv.device, dtype=qkv.dtype)
        lse = torch.empty((b*h, n), device=qkv.device, dtype=torch.float32)
        _forward[(triton.cdiv(n, bm), b*h)](qkv, st, ct, mt, out, lse, n, h,
              prefix, q, rope, masked, bm, bn, num_warps=4, num_stages=2, enable_fp_fusion=False)
        ctx.save_for_backward(qkv, st, ct, mt, out, lse)
        ctx.args = (n, h, prefix, q, rope, masked, bm, bn)
        ctx.mark_non_differentiable(lse)
        return out.reshape(b, n, h*d),lse

    @staticmethod
    def backward(ctx, grad, grad_lse):
        qkv, st, ct, mt, out, lse = ctx.saved_tensors
        n, h, prefix, q, rope, masked, bm, bn = ctx.args
        bh = qkv.shape[0]*h
        grad = grad.contiguous().reshape_as(out)
        delta = torch.empty_like(lse)
        dqkv = torch.empty_like(qkv)
        _delta[(triton.cdiv(n, 32), bh)](out, grad, delta, n, h, 32, num_warps=4)
        _backward_q[(triton.cdiv(n, bm), bh)](qkv, st, ct, mt, grad, lse, delta, dqkv,
                n, h, prefix, q, rope, masked, bm, bn, num_warps=4, num_stages=2, enable_fp_fusion=False)
        _backward_kv[(triton.cdiv(n, bn), bh)](qkv, st, ct, mt, grad, lse, delta, dqkv,
                n, h, prefix, q, rope, masked, bm, bn, num_warps=4, num_stages=2, enable_fp_fusion=False)
        return dqkv, None, None, None, None, None


def attention(qkv, rope=None, compact_mask=None, prefix=None, config=None,return_lse=False):
    if prefix is None:
        prefix = 0 if rope is None else qkv.shape[1] - rope[0].shape[0]
    sin, cos = (None, None) if rope is None else rope
    if config is None:
        cfg={'bm':32,'bn':32} if qkv.dtype==torch.float32 else _TUNED.get(f'{qkv.shape[3]}|{qkv.shape[1]}',{'bm':32,'bn':64})
        config=(cfg['bm'],cfg['bn'])
    out,lse=_Attention.apply(qkv,sin,cos,compact_mask,prefix,config)
    return (out,lse) if return_lse else out


def reference(qkv, rope=None, compact_mask=None, prefix=None):
    b, n, _, h, d = qkv.shape
    q, k, v = (z.transpose(1, 2) for z in qkv.unbind(2))
    if rope is not None:
        sin, cos = rope
        prefix = n - sin.shape[0]
        def rotate(x):
            xf = x[:, :, prefix:].to(sin.dtype)
            rot = torch.cat((-xf[..., 32:], xf[..., :32]), -1)
            return torch.cat((x[:, :, :prefix].to(sin.dtype), xf*cos + rot*sin), -2).to(x.dtype)
        q, k = rotate(q), rotate(k)
    mask = None
    if compact_mask is not None:
        mask = torch.ones((b, 1, n, n), device=qkv.device, dtype=torch.bool)
        mask[:, 0, :compact_mask.shape[1], prefix:] = compact_mask
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1,2).reshape(b,n,h*d)
