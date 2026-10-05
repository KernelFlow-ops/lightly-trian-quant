"""TileLang FlashAttention forward for sm_120 (RTX 5090): non-causal, no mask, head_dim=64, bf16.

Layouts (all contiguous):
  "bhnd": q,k,v,out = [B, H, N, D]
  "qkv" : qkv = [B, N, 3, H, D] (the raw output of the QKV GEMM), out = [B, N, H, D] (== [B, N, H*D],
          i.e. exactly what the proj GEMM consumes) -> no transposes / copies around the kernel.

`rope=True` fuses the RoPE rotation of q and k (fp32 math, result rounded to bf16 like the reference)
for token rows >= `prefix` (row n rotates with table row n-prefix); the tables are [HW, D] fp32 and the
kernel relies on the DINOv3 structure sin[:, :D/2] == sin[:, D/2:] (angles are duplicated).

Softmax: online, scale*log2(e) folded into the exp2 argument, fp32 accumulate.
Tails: N is not a multiple of BLOCK_M/BLOCK_N. Q rows >= N are dropped on store (TileLang predicates
the copy); key columns >= N of the last K tile are masked to -inf before the softmax.
"""

import math
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

LOG2E = 1.44269504


def _flash_prim(B, H, N, D, bm, bn, stages, threads, layout, rope, prefix, hw):
    dtype, accum = "bfloat16", "float"
    scale = (1.0 / math.sqrt(D)) * LOG2E
    half = D // 2
    n_kblk = (N + bn - 1) // bn

    if layout == "bhnd":
        qkv_shape = [B, H, N, D]
        out_shape = [B, H, N, D]
    elif layout == "3bnhd":  # K2's physical layout [3, B, N, H, D]
        qkv_shape = [3, B, N, H, D]
        out_shape = [B, N, H, D]
    else:
        qkv_shape = [B, N, 3, H, D]
        out_shape = [B, N, H, D]

    @T.macro
    def load_q(Q, Q_s, bz, by, bx):
        if layout == "bhnd":
            T.copy(Q[bz, by, bx * bm:(bx + 1) * bm, :], Q_s)
        else:
            if layout == "3bnhd":
                T.copy(Q[0, bz, bx * bm:(bx + 1) * bm, by, :], Q_s)
            else:
                T.copy(Q[bz, bx * bm:(bx + 1) * bm, 0, by, :], Q_s)

    @T.macro
    def load_k(K, K_s, bz, by, k):
        if layout == "bhnd":
            T.copy(K[bz, by, k * bn:(k + 1) * bn, :], K_s)
        else:
            if layout == "3bnhd":
                T.copy(K[1, bz, k * bn:(k + 1) * bn, by, :], K_s)
            else:
                T.copy(K[bz, k * bn:(k + 1) * bn, 1, by, :], K_s)

    @T.macro
    def load_v(V, V_s, bz, by, k):
        if layout == "bhnd":
            T.copy(V[bz, by, k * bn:(k + 1) * bn, :], V_s)
        else:
            if layout == "3bnhd":
                T.copy(V[2, bz, k * bn:(k + 1) * bn, by, :], V_s)
            else:
                T.copy(V[bz, k * bn:(k + 1) * bn, 2, by, :], V_s)

    @T.macro
    def store_o(O_s, Out, bz, by, bx):
        if layout == "bhnd":
            T.copy(O_s, Out[bz, by, bx * bm:(bx + 1) * bm, :])
        else:
            T.copy(O_s, Out[bz, bx * bm:(bx + 1) * bm, by, :])

    @T.macro
    def rot_tile(Raw, Dst, row0, SIN, COS):
        # Dst = bf16( x*cos + rotate_half(x)*sin ) in fp32; rows with token index < prefix pass through.
        # sin/cos are duplicated across the two halves, so each work item handles the pair (j, j+half).
        for i, j in T.Parallel(Raw.shape[0], half):
            tok = row0 + i - prefix
            t = T.min(T.max(tok, 0), hw - 1)
            c = COS[t, j]
            s = SIN[t, j]
            x1 = T.cast(Raw[i, j], accum)
            x2 = T.cast(Raw[i, j + half], accum)
            Dst[i, j] = T.cast(T.if_then_else(tok >= 0, x1 * c - x2 * s, x1), dtype)
            Dst[i, j + half] = T.cast(T.if_then_else(tok >= 0, x2 * c + x1 * s, x2), dtype)

    @T.prim_func
    def main(Q: T.Tensor(qkv_shape, dtype), K: T.Tensor(qkv_shape, dtype),
             V: T.Tensor(qkv_shape, dtype), SIN: T.Tensor([hw, D], accum),
             COS: T.Tensor([hw, D], accum), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(T.ceildiv(N, bm), H, B, threads=threads) as (bx, by, bz):
            Q_s = T.alloc_shared([bm, D], dtype)
            K_s = T.alloc_shared([bn, D], dtype)
            V_s = T.alloc_shared([bn, D], dtype)
            O_s = T.alloc_shared([bm, D], dtype)
            if rope:
                Q_raw = T.alloc_shared([bm, D], dtype)
                K_raw = T.alloc_shared([bn, D], dtype)
            acc_s = T.alloc_fragment([bm, bn], accum)
            acc_c = T.alloc_fragment([bm, bn], dtype)
            acc_o = T.alloc_fragment([bm, D], accum)
            m_cur = T.alloc_fragment([bm], accum)
            m_prev = T.alloc_fragment([bm], accum)
            s_scale = T.alloc_fragment([bm], accum)
            s_sum = T.alloc_fragment([bm], accum)
            logsum = T.alloc_fragment([bm], accum)

            if rope:
                load_q(Q, Q_raw, bz, by, bx)
                rot_tile(Q_raw, Q_s, bx * bm, SIN, COS)
            else:
                load_q(Q, Q_s, bz, by, bx)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(m_cur, -T.infinity(accum))
            for k in T.Pipelined(n_kblk, num_stages=stages):
                if rope:
                    load_k(K, K_raw, bz, by, k)
                    rot_tile(K_raw, K_s, k * bn, SIN, COS)
                else:
                    load_k(K, K_s, bz, by, k)
                T.clear(acc_s)
                T.gemm(Q_s, K_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if N % bn != 0:
                    for i, j in T.Parallel(bm, bn):
                        acc_s[i, j] = T.if_then_else(k * bn + j < N, acc_s[i, j], -T.infinity(accum))
                T.copy(m_cur, m_prev)
                # running max: m_cur keeps the previous max and reduce_max(clear=False) folds the tile into it
                # (re-initialising it to -inf per tile, as the TileLang sample does, overflows exp2(m_prev-m_cur)
                # once a later tile has a much smaller max: real DINOv3 registers give NaN).
                T.reduce_max(acc_s, m_cur, dim=1, clear=False)
                for i in T.Parallel(bm):
                    s_scale[i] = T.exp2(m_prev[i] * scale - m_cur[i] * scale)
                for i, j in T.Parallel(bm, bn):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m_cur[i] * scale)
                T.reduce_sum(acc_s, s_sum, dim=1)
                for i in T.Parallel(bm):
                    logsum[i] = logsum[i] * s_scale[i] + s_sum[i]
                T.copy(acc_s, acc_c)
                for i, j in T.Parallel(bm, D):
                    acc_o[i, j] *= s_scale[i]
                load_v(V, V_s, bz, by, k)
                T.gemm(acc_c, V_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(bm, D):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_s)
            store_o(O_s, Out, bz, by, bx)

    return main


@lru_cache(maxsize=None)
def build(B, H, N, D=64, bm=64, bn=64, stages=2, threads=128, layout="bhnd", rope=False, prefix=0, hw=1024):
    """Compile (once per config) and return the kernel callable; out is allocated by TileLang."""
    prim = _flash_prim(B, H, N, D, bm, bn, stages, threads, layout, rope, prefix, hw)
    return tilelang.compile(prim, out_idx=[-1], target="cuda")


def source(*args, **kw) -> str:
    return build(*args, **kw).get_kernel_source()
