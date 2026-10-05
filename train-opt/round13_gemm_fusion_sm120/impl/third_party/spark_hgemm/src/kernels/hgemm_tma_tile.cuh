// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// The TMA-fed 128x128 tile shared by hgemm variant 5 (hgemm_tma.cu) and variant 6
// (hgemm_tma_sk.cu). Private to src/kernels. The two kernels differ in who computes which
// (tile, K-range) piece; the stage geometry, the swizzle, what the producer lane issues per
// stage and what a consumer warp does with a landed stage are the same and live here once.
//
// A stage is a 128xBK box of A followed by two BKx64 boxes of B (128 B rows, one swizzle span
// each), all written by the copy engine with the map's swizzle: the 64-byte swizzle for A at
// BK = 32 (chunk ^= (row/2)%4), the 128-byte one otherwise (chunk ^= row%8). Both are keyed
// on the absolute shared address, so a stage must start on a 1 KB boundary. The consumer's
// ldmatrix addressing is variant 3's, unchanged; the only difference is that a warp's 32
// columns of B live in box wn/2 at chunk (wn%2)*4 + nj + lane/16.
#pragma once

#include "spark/common.cuh"

namespace spark::hgemm_tma_tile {

constexpr int BM = 128, BN = 128;
constexpr int WARPS_M = 2, WARPS_N = 4;  // consumer warp grid, as in variant 3
constexpr int CONSUMER_WARPS = WARPS_M * WARPS_N;
constexpr int CONSUMERS = CONSUMER_WARPS * 32;  // 256 threads
constexpr int THREADS = CONSUMERS + 32;         // plus the producer warp
constexpr int WM = BM / WARPS_M;                // 64
constexpr int WN = BN / WARPS_N;                // 32
constexpr int MT = WM / 16;                     // m16 tiles per warp
constexpr int NT = WN / 8;                      // n8 tiles per warp
constexpr int B_BOX_N = 64;                     // columns per TMA box of B: 128 B, one swizzle span
static_assert(BN == 2 * B_BOX_N, "B is loaded as two 64-column boxes");

using Acc = float[MT][NT][4];

// Bytes of A + B per stage: a multiple of 1 KB for both BK.
template <int BK>
constexpr int stage_bytes() {
    return (BM * BK + BK * BN) * static_cast<int>(sizeof(__nv_bfloat16));
}
// Round a shared-memory total up to a whole KB, so that with two blocks on an SM the second
// block's allocation (this plus the reserved KB) also starts on a 1 KB boundary, whichever
// address the swizzle is keyed on.
constexpr int round_kb(int bytes) {
    return (bytes + 1023) / 1024 * 1024;
}

// Physical 16-byte chunk for logical (row, chunk) of an A row of BK bf16, as the copy engine
// wrote it. Same function as variant 3's swz_a.
template <int BK>
__device__ __forceinline__ int swz_a(int row, int chunk) {
    if constexpr (BK == 32)
        return chunk ^ ((row >> 1) & 3);
    else
        return chunk ^ (row & 7);
}
// A B box row is 128 B (8 chunks) and the 128-byte swizzle XORs its chunk with row%8.
__device__ __forceinline__ int swz_b(int row, int chunk) {
    return chunk ^ (row & 7);
}

// Host: the two tensor maps. A is K-innermost with a box of BK columns x 128 rows, swizzled
// over the box width; B is N-innermost with a box of 64 columns x BK rows. Rows past M and
// columns past N are zero-filled by the copy engine and still count toward the barrier.
// Encoding is host arithmetic on 128 bytes, 27 ns per call, so maps are rebuilt per call.
template <int BK>
inline void make_maps(const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N, int K,
                      CUtensorMap& tmA, CUtensorMap& tmB) {
    constexpr CUtensorMapSwizzle a_swz =
        BK == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
    tmA = make_tensor_map_2d_bf16(A, M, K, BM, BK, a_swz);
    tmB = make_tensor_map_2d_bf16(B, K, N, BK, B_BOX_N, CU_TENSOR_MAP_SWIZZLE_128B);
}

// Producer lane: register the stage's bytes on `full` and issue its three boxes, k-tile
// [k0, k0 + BK) of the tile at (bm, bn), into `stage`.
template <int BK>
__device__ __forceinline__ void issue_stage(unsigned char* stage, const CUtensorMap* tmA,
                                            const CUtensorMap* tmB, uint64_t* full, int k0, int bm,
                                            int bn) {
    constexpr int A_BYTES = BM * BK * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr int B_BOX_BYTES = BK * B_BOX_N * static_cast<int>(sizeof(__nv_bfloat16));
    mbar_arrive_expect_tx(full, stage_bytes<BK>());
    unsigned char* bs = stage + A_BYTES;
    tma_load_2d(stage, tmA, full, k0, bm);  // (k, m): 128 rows x BK
    tma_load_2d(bs, tmB, full, bn, k0);     // (n, k): BK rows x 64
    tma_load_2d(bs + B_BOX_BYTES, tmB, full, bn + B_BOX_N, k0);
}

__device__ __forceinline__ void zero_acc(Acc& acc) {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
}

// Consumer warp (wm, wn): variant 3's ldmatrix + mma.sync body over one landed stage. The
// caller has waited on the stage's "full" barrier; afterwards it fences the proxy,
// __syncwarp()s and arrives on "empty".
template <int BK>
__device__ __forceinline__ void consume_stage(Acc& acc, const unsigned char* stage, int wm, int wn,
                                              int lane) {
    constexpr int A_STAGE = BM * BK;     // elements
    constexpr int B_BOX = BK * B_BOX_N;  // elements per B box
    const __nv_bfloat16* as = reinterpret_cast<const __nv_bfloat16*>(stage);
    const __nv_bfloat16* bs = as + A_STAGE + (wn >> 1) * B_BOX;  // the box with this warp's columns

    const int a_row_in_tile = lane & 15;
    const int a_kchunk = lane >> 4;
    const int b_krow_in_step = lane & 15;
    const int b_nchunk = lane >> 4;

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
            const int ch = ((wn & 1) * WN + nj * 8) / 8 + b_nchunk;  // chunk within the box
            unsigned r[4];
            ldmatrix_x4_trans(r, bs + krow * B_BOX_N + swz_b(krow, ch) * 8);
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

}  // namespace spark::hgemm_tma_tile
