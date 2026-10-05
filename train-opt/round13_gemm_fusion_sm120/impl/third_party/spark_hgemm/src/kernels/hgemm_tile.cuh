// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// The mma.sync + ldmatrix block tile shared by hgemm variant 3 (hgemm.cu) and variant 4
// (hgemm_streamk.cu). Private to src/kernels: the two kernels differ only in how they hand
// out (tile, K-range) work to blocks, so the tile itself lives here once.
//
// Block tile BM x BN x BK, 8 warps as 2 (M) x 4 (N), warp tile (BM/2) x (BN/4), XOR-swizzled
// shared memory, STAGES-deep cp.async pipeline. Rows past M are zero-filled on the way in
// (cp.async with a source size of 0) and skipped on the way out, so any M % 16 == 0 works;
// N % BN == 0 and K % BK == 0 are the caller's job.
#pragma once

#include "hgemm_epilogue.cuh"
#include "spark/common.cuh"

namespace spark::hgemm_tile {

constexpr int THREADS = 256;
constexpr int WARPS_M = 2, WARPS_N = 4;  // 8 warps as 2 (M) x 4 (N), whatever the tile

// Physical 16-byte chunk for logical (row, chunk) of an A tile whose rows are BK bf16 long.
// Rows of 64 B (BK=32): chunk ^= (row/2)%4; rows of 128 B (BK=64): chunk ^= row%8. Either
// way the 8 rows an ldmatrix touches land in 8 different bank groups.
template <int BK>
__device__ __forceinline__ int swz_a(int row, int chunk) {
    if constexpr (BK == 32)
        return chunk ^ ((row >> 1) & 3);
    else
        return chunk ^ (row & 7);
}
// B rows are >= 128 B (BN >= 64 bf16): XOR the low three bits of the chunk index with row%8.
__device__ __forceinline__ int swz_b(int row, int chunk) {
    return chunk ^ (row & 7);
}

template <int BM, int BN, int BK, int STAGES>
constexpr int smem_bytes() {
    return STAGES * (BM * BK + BK * BN) * static_cast<int>(sizeof(__nv_bfloat16));
}

// Compile-time layout of one tile configuration.
template <int BM_, int BN_, int BK_, int STAGES_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_;
    static_assert(BK == 32 || BK == 64, "swizzle assumes 64 B or 128 B rows of A");
    static_assert(BM == 64 || BM == 128, "warp tile is BM/2 tall: 32 or 64");
    static_assert(BN == 64 || BN == 128 || BN == 256, "warp tile is BN/4 wide: 16, 32 or 64");
    static constexpr int WM = BM / WARPS_M;  // 32 or 64
    static constexpr int WN = BN / WARPS_N;  // 16, 32 or 64
    static constexpr int MT = WM / 16;       // m16 tiles per warp
    static constexpr int NT = WN / 8;        // n8 tiles per warp (even: ldmatrix.x4 loads two)
    static_assert(NT % 2 == 0, "");
    static constexpr int B_CPR = BN / 8;                    // 16-byte chunks per smem row of B
    static constexpr int A_CPR = BK / 8;                    // chunks per A row
    static constexpr int A_ITERS = (BM * A_CPR) / THREADS;  // chunks per thread per stage
    static constexpr int B_ITERS = (BK * B_CPR) / THREADS;
    static_assert(A_ITERS * THREADS == BM * A_CPR && B_ITERS * THREADS == BK * B_CPR, "");
    static constexpr int A_STAGE = BM * BK;  // elements
    static constexpr int B_STAGE = BK * BN;
    static constexpr int SMEM = smem_bytes<BM, BN, BK, STAGES>();
    using Acc = float[MT][NT][4];
};

// Accumulates A[bm.., kt_begin*BK .. (kt_begin+nkt)*BK) x B[same K, bn..] into `acc`, which
// it zeroes first. `smem` is the STAGES-deep tile buffer (Cfg::SMEM bytes). The pipeline is
// drained on return; the caller must __syncthreads() before reusing `smem` for another call.
template <class C>
__device__ __forceinline__ void mainloop(const __nv_bfloat16* __restrict__ A,
                                         const __nv_bfloat16* __restrict__ B, int N, int K, int bm,
                                         int bn, int m_valid, int kt_begin, int nkt,
                                         unsigned char* smem, typename C::Acc& acc) {
    constexpr int BN = C::BN, BK = C::BK, STAGES = C::STAGES;
    constexpr int WM = C::WM, WN = C::WN, MT = C::MT, NT = C::NT;
    constexpr int A_CPR = C::A_CPR, B_CPR = C::B_CPR, A_ITERS = C::A_ITERS, B_ITERS = C::B_ITERS;
    constexpr int A_STAGE = C::A_STAGE, B_STAGE = C::B_STAGE;

    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Bs = As + STAGES * A_STAGE;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;

    const __nv_bfloat16* Ab = A + static_cast<size_t>(bm) * K + static_cast<size_t>(kt_begin) * BK;
    const __nv_bfloat16* Bb = B + static_cast<size_t>(kt_begin) * BK * N + bn;

    auto load_stage = [&](int stage, int k0) {
        __nv_bfloat16* as = As + stage * A_STAGE;
        __nv_bfloat16* bs = Bs + stage * B_STAGE;
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int c = tid + i * THREADS;
            const int row = c / A_CPR;
            const int ch = c % A_CPR;
            const bool ok = row < m_valid;
            cp_async_16_zfill(as + row * BK + swz_a<BK>(row, ch) * 8,
                              Ab + static_cast<size_t>(ok ? row : 0) * K + k0 + ch * 8, ok);
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
            const int c = tid + i * THREADS;
            const int row = c / B_CPR;
            const int ch = c % B_CPR;
            cp_async_16(bs + row * BN + swz_b(row, ch) * 8,
                        Bb + static_cast<size_t>(k0 + row) * N + ch * 8);
        }
    };

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    // Prologue: the first STAGES-1 tiles are in flight before any compute starts.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nkt) load_stage(s, s * BK);
        cp_async_commit();
    }

    // Per-lane ldmatrix row/chunk selectors (constant across the K loop).
    const int a_row_in_tile = lane & 15;   // row within a 16-row m tile
    const int a_kchunk = lane >> 4;        // 0/1: k 0-7 or 8-15 of the current k16 step
    const int b_krow_in_step = lane & 15;  // k row within the k16 step
    const int b_nchunk = lane >> 4;        // 0/1: which n8 tile of the pair

    for (int kt = 0; kt < nkt; ++kt) {
        cp_async_wait<STAGES - 2>();  // tile kt has landed (for this thread)
        __syncthreads();              // ... for every thread; and stage (kt-1)%STAGES is free
        {
            const int nk = kt + STAGES - 1;
            if (nk < nkt) load_stage(nk % STAGES, nk * BK);
            cp_async_commit();  // always commit so the group count stays uniform
        }
        const __nv_bfloat16* as = As + (kt % STAGES) * A_STAGE;
        const __nv_bfloat16* bs = Bs + (kt % STAGES) * B_STAGE;

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned afrag[MT][4];
            unsigned bfrag[NT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + a_row_in_tile;
                const int ch = kk / 8 + a_kchunk;
                ldmatrix_x4(afrag[mi], as + row * BK + swz_a<BK>(row, ch) * 8);
            }
#pragma unroll
            for (int nj = 0; nj < NT; nj += 2) {
                const int krow = kk + b_krow_in_step;
                const int ch = (wn * WN + nj * 8) / 8 + b_nchunk;
                unsigned r[4];
                ldmatrix_x4_trans(r, bs + krow * BN + swz_b(krow, ch) * 8);
                bfrag[nj][0] = r[0];
                bfrag[nj][1] = r[1];
                bfrag[nj + 1][0] = r[2];
                bfrag[nj + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int nj = 0; nj < NT; ++nj) mma_bf16_16816(acc[mi][nj], afrag[mi], bfrag[nj]);
        }
    }
    cp_async_wait<0>();
}

// Epilogue: each lane owns (row g, cols 2c..2c+1) and (row g+8, same cols) of every 16x8
// tile; two bf16 per store, straight from registers. Rows past M are skipped. FUSED applies
// the epilogue options of `ep` (hgemm_epilogue.cuh) on the way; the default is the plain
// store.
template <class C, bool FUSED = false>
__device__ __forceinline__ void store_bf16(const typename C::Acc& acc,
                                           __nv_bfloat16* __restrict__ C_, int M, int N, int bm,
                                           int bn, const hgemm_epi::Ep& ep = hgemm_epi::Ep()) {
    constexpr int WM = C::WM, WN = C::WN, MT = C::MT, NT = C::NT;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    hgemm_epi::store_warp_tile<FUSED, MT, NT>(acc, C_, M, N, bm + wm * WM + g, bn + wn * WN + c2,
                                              ep);
}

}  // namespace spark::hgemm_tile

namespace spark {
// Variant 4 (hgemm_streamk.cu); dispatched to by hgemm_bf16 and, for the shapes its TMA tile
// does not take, by variant 6. The caller has validated the shape with
// hgemm_supports(M, N, K, 4).
void hgemm_streamk_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M,
                        int N, int K, cudaStream_t stream,
                        const HgemmEpilogue& ep = HgemmEpilogue());
}  // namespace spark
