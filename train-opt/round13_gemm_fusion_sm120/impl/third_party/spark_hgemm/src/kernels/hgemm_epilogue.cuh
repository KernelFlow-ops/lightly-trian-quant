// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// The fused epilogue of the bf16 GEMM, shared by variant 4's tile (hgemm_tile.cuh), variant 6's
// finishing piece (hgemm_tma_sk.cu) and the decode kernel (hgemm_decode.cu). Private to
// src/kernels.
//
// Every path ends the same way: a lane holds fp32 sums for a few output elements and rounds
// them to bf16. This header is that step with the options of HgemmEpilogue
// (include/spark/kernels.h) applied in fp32 first, in this order:
//
//   x = acc + bias[col]                    (bias: one bf16 per column of B)
//   y = act(x)                             (none, silu, gelu with the tanh form, relu)
//   y = silu(x[2j]) * x[2j+1]              (swiglu: instead of act, see below)
//   C = bf16(y + residual[row][col])       (residual: bf16 [M][N]; residual == C is C += A B)
//
// SwiGLU form. The mma.sync accumulator gives each lane columns (c, c+1) of an n8 tile, an
// adjacent pair. So a weight whose column 2j is gate_j and column 2j+1 is up_j puts gate and
// up of the same output element into one lane's register pair, whatever the block or warp
// tile, and the epilogue multiplies them without touching shared memory. The output then has
// N / 2 columns. The lane pairs (2q, 2q+1) hold output columns j and j + 1 of the same rows,
// so one shuffle per fragment turns two 2-byte stores into one 4-byte store per lane: the
// even lane stores row `row`, the odd lane stores row `row + 8`.
//
// Shape of the code. The kernels that use this run eight or nine warps per SM, so the
// epilogue cannot hide latency behind other warps; it has to be short and issue its loads
// early. Three things follow. The activation and the SwiGLU flag are template parameters of
// the warp-tile loop and `dispatch` picks the instantiation once per tile, so the executed
// path is one compact loop and not a switch per element inside sixteen unrolled fragments
// (that version cost 33 us on a 4096^3 GEMM for relu alone: instruction fetch on a 100 KB
// straight-line epilogue). The loads come first: every bias and residual value of the warp
// tile is loaded into registers (`Pre`) before any math, in variant 6 before the k-loop of
// the piece that will store, so a tile pays one DRAM round trip for its residual, hidden
// under the mma work, and not one per fragment. And the activations use the SFU
// exponential and the approximate divide, not the correctly rounded reciprocal, whose
// slow-path subroutine call per element serialized the loop (57 us for gelu at 4096^3).
//
// Bounds: rows past M are skipped one by one (the callers zero-fill them on the way in).
// Alignment: col is even and N is a multiple of 64, so every bias, residual and C access is a
// whole __nv_bfloat162; the SwiGLU output column of an even lane is even too.
#pragma once

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark::hgemm_epi {

// Device-side copy of HgemmEpilogue: the same fields, no defaults, trivially copyable as a
// kernel parameter.
struct Ep {
    const __nv_bfloat16* bias;
    const __nv_bfloat16* residual;
    int act;
    int swiglu;
};

inline Ep make(const HgemmEpilogue& e) {
    return {e.bias, e.residual, e.act, e.swiglu ? 1 : 0};
}

inline bool is_plain(const HgemmEpilogue& e) {
    return e.bias == nullptr && e.residual == nullptr && e.act == HGEMM_ACT_NONE && !e.swiglu;
}

// Host: the columns of C for a B of N columns.
inline int out_cols(const HgemmEpilogue& e, int N) {
    return e.swiglu ? N / 2 : N;
}

// x * sigmoid(x) = x / (1 + exp(-x)), with the SFU exponential and the approximate divide
// (MUFU.RCP and a multiply, 2 ulp): exp(-x) overflows to +inf for x << 0, the quotient is
// then -0, the right limit. Not __frcp_rn or a plain division: those are correctly rounded
// through a slow-path subroutine, a call per element that serializes the epilogue.
__device__ __forceinline__ float silu_f(float x) {
    return __fdividef(x, 1.0f + __expf(-x));
}

// gelu in the tanh form, as torch's F.gelu(approximate="tanh"): 0.5 x (1 + tanh(u)) with
// u = sqrt(2/pi) (x + 0.044715 x^3). Since 1 + tanh(u) = 2 sigmoid(2u), that is
// x * sigmoid(2u), the same handful of instructions as silu.
__device__ __forceinline__ float gelu_tanh_f(float x) {
    const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return __fdividef(x, 1.0f + __expf(-2.0f * u));
}

template <int ACT>
__device__ __forceinline__ float act_f(float x) {
    if constexpr (ACT == HGEMM_ACT_SILU) return silu_f(x);
    if constexpr (ACT == HGEMM_ACT_GELU) return gelu_tanh_f(x);
    if constexpr (ACT == HGEMM_ACT_RELU) return fmaxf(x, 0.0f);
    return x;
}

// The activation and the SwiGLU flag as a type, so the warp-tile loop is instantiated per
// mode and `dispatch` picks one per tile.
template <int ACT, bool SWIGLU>
struct Mode {
    static constexpr int act = ACT;
    static constexpr bool swiglu = SWIGLU;
};

// Calls f(Mode<...>{}) for the mode `ep` asks for. One uniform branch per tile.
template <class F>
__device__ __forceinline__ void dispatch(const Ep& ep, F&& f) {
    if (ep.swiglu) {
        f(Mode<HGEMM_ACT_SILU, true>{});
        return;
    }
    switch (ep.act) {
        case HGEMM_ACT_SILU:
            f(Mode<HGEMM_ACT_SILU, false>{});
            break;
        case HGEMM_ACT_GELU:
            f(Mode<HGEMM_ACT_GELU, false>{});
            break;
        case HGEMM_ACT_RELU:
            f(Mode<HGEMM_ACT_RELU, false>{});
            break;
        default:
            f(Mode<HGEMM_ACT_NONE, false>{});
            break;
    }
}

__device__ __forceinline__ float2 load2(const __nv_bfloat16* p) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}

__device__ __forceinline__ void store2(__nv_bfloat16* p, float a, float b) {
    *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

// The residual pair (row, col..col+1) as raw bf16x2 bits, or 0 (= +0 twice) when there is
// none or the row is past M. A plain load: in the accumulate form the residual is C, which
// this thread stores to at that address after the load has returned, and nothing else reads
// it again.
__device__ __forceinline__ unsigned load_res(const Ep& ep, int ldc, int row, int col, bool ok) {
    if (ep.residual == nullptr || !ok) return 0u;
    return *reinterpret_cast<const unsigned*>(ep.residual + static_cast<size_t>(row) * ldc + col);
}

__device__ __forceinline__ float2 unpack(unsigned bits) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
}

// The warp-tile geometry every kernel shares: fragment (mi, nj) of a lane's MT x NT m16n8
// accumulators is (row0 + 16 mi, col0 + 8 nj .. + 1) in v[0..1] and the row 8 below in
// v[2..3], col0 even. N is the width of B; C has N columns, N / 2 with SwiGLU. In the SwiGLU
// form the lane pair (2q, 2q+1) stores output columns (j0, j0 + 1): the even lane the upper
// row of each fragment, the odd lane the lower one.
struct Geo {
    int M, N, row0, col0;
    __device__ __forceinline__ bool odd() const { return (threadIdx.x & 1) != 0; }
    __device__ __forceinline__ int r0() const { return row0 + (odd() ? 8 : 0); }
    __device__ __forceinline__ int j0() const { return (col0 >> 1) - (odd() ? 1 : 0); }
};

// The epilogue's inputs for one warp tile, loaded before they are needed: the bias of the
// lane's column pairs and the residual of every fragment (both rows; the SwiGLU form uses
// [0] only). 8 + 2 MT NT registers, 40 for the 128x128 tile. Variant 6 loads them before a
// finishing piece's k-loop, so the residual's DRAM round trip hides under the mma work;
// the other kernels load them right before the store.
template <int MT, int NT>
struct Pre {
    float2 bias[NT];
    unsigned res[MT][NT][2];
};

template <int MT, int NT>
__device__ __forceinline__ void prefetch(Pre<MT, NT>& pf, const Ep& ep, const Geo& g) {
#pragma unroll
    for (int nj = 0; nj < NT; ++nj)
        pf.bias[nj] = ep.bias ? load2(ep.bias + g.col0 + nj * 8) : make_float2(0.f, 0.f);
    if (ep.swiglu) {
        const int ldc = g.N >> 1, r0 = g.r0(), j0 = g.j0();
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                pf.res[mi][nj][0] =
                    load_res(ep, ldc, r0 + mi * 16, j0 + nj * 4, r0 + mi * 16 < g.M);
                pf.res[mi][nj][1] = 0u;
            }
    } else {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row = g.row0 + mi * 16;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                pf.res[mi][nj][0] = load_res(ep, g.N, row, g.col0 + nj * 8, row < g.M);
                pf.res[mi][nj][1] = load_res(ep, g.N, row + 8, g.col0 + nj * 8, row + 8 < g.M);
            }
        }
    }
}

// The math and the stores of one warp tile for one mode, from the accumulators and the
// prefetched inputs. Every lane of the warp must call it (the SwiGLU form shuffles), so the
// callers' loops are warp-uniform.
template <class M_, int MT, int NT>
__device__ __forceinline__ void store_warp_tile_mode(const float (&acc)[MT][NT][4],
                                                     const Pre<MT, NT>& pf,
                                                     __nv_bfloat16* __restrict__ C, const Geo& g) {
    constexpr int ACT = M_::act;
    if constexpr (M_::swiglu) {
        const bool odd = g.odd();
        const int ldc = g.N >> 1, r0 = g.r0(), j0 = g.j0();
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float* v = acc[mi][nj];
                const float2 b = pf.bias[nj];
                const float y0 = silu_f(v[0] + b.x) * (v[1] + b.y);  // row0 + 16 mi
                const float y1 = silu_f(v[2] + b.x) * (v[3] + b.y);  // 8 below
                // Even lane q holds column j, odd lane q + 1 holds j + 1, of both rows. The
                // even lane takes the odd lane's upper-row value and stores (j, j + 1) of
                // that row; the odd lane takes the even lane's lower-row value and stores the
                // same pair of the lower row.
                const float recv = __shfl_xor_sync(kFullMask, odd ? y0 : y1, 1);
                const int r = r0 + mi * 16;
                if (r < g.M) {
                    const float2 rs = unpack(pf.res[mi][nj][0]);
                    store2(C + static_cast<size_t>(r) * ldc + j0 + nj * 4, (odd ? recv : y0) + rs.x,
                           (odd ? y1 : recv) + rs.y);
                }
            }
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row = g.row0 + mi * 16;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float* v = acc[mi][nj];
                const float2 b = pf.bias[nj];
                __nv_bfloat16* p = C + static_cast<size_t>(row) * g.N + g.col0 + nj * 8;
                if (row < g.M) {
                    const float2 rs = unpack(pf.res[mi][nj][0]);
                    store2(p, act_f<ACT>(v[0] + b.x) + rs.x, act_f<ACT>(v[1] + b.y) + rs.y);
                }
                if (row + 8 < g.M) {
                    const float2 rs = unpack(pf.res[mi][nj][1]);
                    store2(p + static_cast<size_t>(8) * g.N, act_f<ACT>(v[2] + b.x) + rs.x,
                           act_f<ACT>(v[3] + b.y) + rs.y);
                }
            }
        }
    }
}

// The fused store from prefetched inputs: one uniform branch picks the mode's loop.
template <int MT, int NT>
__device__ __forceinline__ void store_warp_tile_fused(const float (&acc)[MT][NT][4],
                                                      const Pre<MT, NT>& pf,
                                                      __nv_bfloat16* __restrict__ C, const Geo& g,
                                                      const Ep& ep) {
    dispatch(ep, [&](auto mode) { store_warp_tile_mode<decltype(mode), MT, NT>(acc, pf, C, g); });
}

// The warp-tile store every kernel calls: the plain rounding when FUSED is false (the code
// the kernels had before), otherwise the loads and then the mode `ep` asks for.
template <bool FUSED, int MT, int NT>
__device__ __forceinline__ void store_warp_tile(const float (&acc)[MT][NT][4],
                                                __nv_bfloat16* __restrict__ C, int M, int N,
                                                int row0, int col0, const Ep& ep) {
    if constexpr (!FUSED) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const int row = row0 + mi * 16;
                __nv_bfloat16* p = C + static_cast<size_t>(row) * N + col0 + nj * 8;
                if (row < M) store2(p, acc[mi][nj][0], acc[mi][nj][1]);
                if (row + 8 < M)
                    store2(p + static_cast<size_t>(8) * N, acc[mi][nj][2], acc[mi][nj][3]);
            }
        }
    } else {
        const Geo g{M, N, row0, col0};
        Pre<MT, NT> pf;
        prefetch(pf, ep, g);
        store_warp_tile_fused(acc, pf, C, g, ep);
    }
}

// Four consecutive columns c..c+3 (c % 4 == 0) of one row r < M, from an fp32 workspace: the
// split-K finisher of the decode kernel. N is the width of B. The caller runs its loop inside
// `dispatch` so the mode is fixed per launch.
template <class M_>
__device__ __forceinline__ void store_vec4_mode(float4 v, __nv_bfloat16* __restrict__ C, int N,
                                                int r, int c, const Ep& ep) {
    if (ep.bias) {
        const float2 b0 = load2(ep.bias + c);
        const float2 b1 = load2(ep.bias + c + 2);
        v.x += b0.x;
        v.y += b0.y;
        v.z += b1.x;
        v.w += b1.y;
    }
    if constexpr (M_::swiglu) {
        const int ldc = N >> 1;
        const float2 rs = unpack(load_res(ep, ldc, r, c >> 1, true));
        store2(C + static_cast<size_t>(r) * ldc + (c >> 1), silu_f(v.x) * v.y + rs.x,
               silu_f(v.z) * v.w + rs.y);
    } else {
        constexpr int ACT = M_::act;
        const float2 r0 = unpack(load_res(ep, N, r, c, true));
        const float2 r1 = unpack(load_res(ep, N, r, c + 2, true));
        __nv_bfloat16* p = C + static_cast<size_t>(r) * N + c;
        store2(p, act_f<ACT>(v.x) + r0.x, act_f<ACT>(v.y) + r0.y);
        store2(p + 2, act_f<ACT>(v.z) + r1.x, act_f<ACT>(v.w) + r1.y);
    }
}

// The plain form of the above: four bf16 in one 8-byte store.
__device__ __forceinline__ void store_vec4_plain(float4 v, __nv_bfloat16* __restrict__ C, int N,
                                                 int r, int c) {
    const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 packed;
    packed.x = *reinterpret_cast<const unsigned*>(&lo);
    packed.y = *reinterpret_cast<const unsigned*>(&hi);
    *reinterpret_cast<uint2*>(C + static_cast<size_t>(r) * N + c) = packed;
}

}  // namespace spark::hgemm_epi
