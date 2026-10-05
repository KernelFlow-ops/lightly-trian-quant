"""TileLang GEMM kernels with fused epilogues (K2, K3, K4) -- Round 5 version (adds swizzle / warp-policy knobs).
sm_120: mma.sync path; TMA + warp-specialised or cp.async pipeline.

Extra knobs vs round3: swz (T.use_swizzle panel size, 0 = off), pol (T.gemm warp policy: square/row/col).
"""

import functools

import tilelang
import tilelang.language as T

BF16, F32 = "bfloat16", "float32"
INV_SQRT2 = 0.7071067811865476
PC = tilelang.PassConfigKey


def _compile(main, out_idx, ws=True, tma=True):
    """ws/tma=False -> Ampere-style cp.async + ldmatrix pipeline instead of TMA + warp specialization."""
    pc = {}
    if not ws:
        pc[PC.TL_DISABLE_WARP_SPECIALIZED] = True
    if not tma:
        pc[PC.TL_DISABLE_TMA_LOWER] = True
    return tilelang.compile(main, out_idx=out_idx, target="cuda", pass_configs=pc)


def _policy(name):
    return {"square": T.GemmWarpPolicy.Square, "row": T.GemmWarpPolicy.FullRow,
            "col": T.GemmWarpPolicy.FullCol}[name]


@T.macro
def _mainloop(A, W, a_s, w_s, c_l, by, bx, bm, bn, bk, K, stages, pol):
    T.clear(c_l)
    for k in T.Pipelined(T.ceildiv(K, bk), num_stages=stages):
        T.copy(A[by * bm, k * bk], a_s)
        T.copy(W[bx * bn, k * bk], w_s)
        T.gemm(a_s, w_s, c_l, transpose_B=True, policy=pol)


@functools.lru_cache(maxsize=None)
def gemm_bias_gelu_kernel(M, N, K, bm=128, bn=128, bk=32, stages=3, threads=128, ws=True, tma=True,
                          swz=0, pol="square"):
    P = _policy(pol)

    @T.prim_func
    def main(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16), Bias: T.Tensor((N,), BF16),
             C: T.Tensor((M, N), BF16)):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, bk), BF16)
            w_s = T.alloc_shared((bn, bk), BF16)
            c_l = T.alloc_fragment((bm, bn), F32)
            if swz > 0:
                T.use_swizzle(swz)
            _mainloop(A, W, a_s, w_s, c_l, by, bx, bm, bn, bk, K, stages, P)
            for i, j in T.Parallel(bm, bn):
                x = c_l[i, j] + T.cast(Bias[bx * bn + j], F32)
                c_l[i, j] = x * 0.5 * (1.0 + T.erf(x * INV_SQRT2))
            T.copy(c_l, C[by * bm, bx * bn])

    return _compile(main, [3], ws, tma)


@functools.lru_cache(maxsize=None)
def gemm_bias_residual_kernel(M, N, K, bm=128, bn=128, bk=32, stages=3, threads=128, ws=True, tma=True,
                              res_dtype=BF16, swz=0, pol="square"):
    """C = R + Gamma * (A@W^T + bias).  R / C dtype = res_dtype (bf16 or fp32 residual stream)."""
    P = _policy(pol)

    @T.prim_func
    def main(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16), Bias: T.Tensor((N,), BF16),
             Gamma: T.Tensor((N,), F32), R: T.Tensor((M, N), res_dtype), C: T.Tensor((M, N), res_dtype)):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, bk), BF16)
            w_s = T.alloc_shared((bn, bk), BF16)
            c_l = T.alloc_fragment((bm, bn), F32)
            if swz > 0:
                T.use_swizzle(swz)
            _mainloop(A, W, a_s, w_s, c_l, by, bx, bm, bn, bk, K, stages, P)
            for i, j in T.Parallel(bm, bn):
                gm = by * bm + i
                gn = bx * bn + j
                c_l[i, j] = (c_l[i, j] + T.cast(Bias[gn], F32)) * Gamma[gn] + T.if_then_else(
                    gm < M, T.cast(R[gm, gn], F32), 0.0)
            T.copy(c_l, C[by * bm, bx * bn])

    return _compile(main, [5], ws, tma)


@functools.lru_cache(maxsize=None)
def qkv_rope_kernel(B, Ntok, d, H, prefix, bm=128, bn=128, bk=32, stages=3, threads=128, ws=True, tma=True,
                    round_bf16=False, layout="bnhd", swz=0, pol="square"):
    """RoPE (fp32) on q,k rows with token index >= prefix; bf16 output (see round3 for the layout notes)."""
    stage_dt = BF16 if round_bf16 else F32
    M, N, K, hd = B * Ntok, 3 * d, d, 64
    HW = Ntok - prefix
    P = _policy(pol)
    assert d % bn == 0 and bn % 64 == 0

    @T.prim_func
    def main(A: T.Tensor((M, K), BF16), W: T.Tensor((N, K), BF16), Bias: T.Tensor((N,), BF16),
             Sin: T.Tensor((HW, hd), F32), Cos: T.Tensor((HW, hd), F32),
             Out: T.Tensor((3, B, Ntok, H, hd) if layout == "bnhd" else (3, B, H, Ntok, hd), BF16)):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, bk), BF16)
            w_s = T.alloc_shared((bn, bk), BF16)
            c_l = T.alloc_fragment((bm, bn), F32)
            c_s = T.alloc_shared((bm, bn), stage_dt)
            if swz > 0:
                T.use_swizzle(swz)
            _mainloop(A, W, a_s, w_s, c_l, by, bx, bm, bn, bk, K, stages, P)
            for i, j in T.Parallel(bm, bn):
                c_l[i, j] = c_l[i, j] + T.cast(Bias[bx * bn + j], F32)
            T.copy(c_l, c_s)
            for i, j in T.Parallel(bm, bn):
                gm = by * bm + i
                gn = bx * bn + j
                b_i = gm // Ntok
                n_i = gm % Ntok
                p = n_i - prefix
                which = gn // d
                h = (gn % d) // hd
                c = gn % hd
                x = T.cast(c_s[i, j], F32)
                xp = T.cast(c_s[i, j ^ 32], F32)
                sgn = T.if_then_else(c < 32, -1.0, 1.0)
                do_rope = (which < 2) and (p >= 0)
                pc = T.max(p, 0)
                r = x * Cos[pc, c] + sgn * xp * Sin[pc, c]
                val = T.if_then_else(do_rope, r, x)
                if gm < M:
                    if layout == "bnhd":
                        Out[which, b_i, n_i, h, c] = T.cast(val, BF16)
                    else:
                        Out[which, b_i, h, n_i, c] = T.cast(val, BF16)

    return _compile(main, [5], ws, tma)
