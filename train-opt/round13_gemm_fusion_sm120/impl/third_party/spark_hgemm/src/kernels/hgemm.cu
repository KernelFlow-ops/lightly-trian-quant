// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// bf16 tensor-core GEMM ladder for Blackwell sm_12x: RTX 5090 (sm_120), GB10 (sm_121).
//
//   C[M,N] = A[M,K] * B[K,N]      row-major, bf16 in/out, fp32 accumulate
//
// All variants use the WMMA API, which lowers to mma.sync tensor-core instructions.
// The RTX 5090 and GB10 are consumer/workstation-lineage Blackwell (sm_120 / sm_121): they
// support mma.sync but NOT the datacenter-only tcgen05 / TMA path of sm_100 (B200). WMMA /
// mma.sync is therefore the right portable tool here.
//
//   variant 0: one warp per 16x16 C tile, fragments loaded straight from global memory.
//   variant 1: 128x128x32 block tile, 8 warps (2x4), shared-memory staged with +8 padding.
//   variant 2: variant 1 + two-stage cp.async pipeline (load tile k+1 while computing tile k).
//   variant 3: raw mma.sync + ldmatrix, XOR-swizzled smem, 3-stage cp.async pipeline,
//              register-direct epilogue, split-K on the last partial wave of tiles.
//   variant 4: the same tile on a persistent Stream-K schedule (hgemm_streamk.cu).
//   variant 5: variant 3's tile and k-loop fed by TMA loads through a warp-specialized
//              mbarrier pipeline (src/kernels/hgemm_tma.cu).
//   variant 6: variant 5's TMA mainloop on variant 4's Stream-K schedule
//              (src/kernels/hgemm_tma_sk.cu).
//
// Tile sizes come from the GB10 analysis (48 SMs, 273 GB/s, 24 MB L2) in
// docs/design/hgemm.md; re-tune on the RTX 5090 (170 SMs, 1,792 GB/s GDDR7).

#include <mma.h>  // after cuda_bf16.h (pulled in by common.cuh)

#include <algorithm>

#include "hgemm_internal.cuh"
#include "hgemm_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

using namespace nvcuda;

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

using FragA =
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>;
using FragB =
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

// Convert 8 fp32 values (16-byte aligned source) into 8 bf16 packed as one uint4 store.
__device__ __forceinline__ uint4 pack8_bf16(const float* src) {
    __nv_bfloat162 h0 = __floats2bfloat162_rn(src[0], src[1]);
    __nv_bfloat162 h1 = __floats2bfloat162_rn(src[2], src[3]);
    __nv_bfloat162 h2 = __floats2bfloat162_rn(src[4], src[5]);
    __nv_bfloat162 h3 = __floats2bfloat162_rn(src[6], src[7]);
    uint4 out;
    out.x = *reinterpret_cast<unsigned*>(&h0);
    out.y = *reinterpret_cast<unsigned*>(&h1);
    out.z = *reinterpret_cast<unsigned*>(&h2);
    out.w = *reinterpret_cast<unsigned*>(&h3);
    return out;
}

// ---------------------------------------------------------------------------------------
// Variant 0: WMMA baseline. Block = 4 warps covering a 32x32 region (2x2 warp tiles of 16x16).
// Each warp walks K in steps of 16 loading fragments directly from global memory.
// ---------------------------------------------------------------------------------------
constexpr int V0_TILE = 32;
constexpr int V0_THREADS = 128;

__global__ void __launch_bounds__(V0_THREADS)
    hgemm_v0_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    __shared__ __align__(32) float scratch[4][WMMA_M * WMMA_N];

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile_row = blockIdx.y * V0_TILE + (warp_id >> 1) * WMMA_M;
    const int tile_col = blockIdx.x * V0_TILE + (warp_id & 1) * WMMA_N;
    // M, N are multiples of 16, so a tile is either fully inside or fully outside.
    if (tile_row >= M || tile_col >= N) return;

    FragA a_frag;
    FragB b_frag;
    FragC c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + static_cast<size_t>(tile_row) * K + k0, K);
        wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(k0) * N + tile_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* s = scratch[warp_id];
    wmma::store_matrix_sync(s, c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();
    // 32 lanes x 8 elements = 256 = one 16x16 tile.
    const int r = lane >> 1;
    const int c0 = (lane & 1) * 8;
    const size_t out = static_cast<size_t>(tile_row + r) * N + tile_col + c0;
    *reinterpret_cast<uint4*>(C + out) = pack8_bf16(s + r * WMMA_N + c0);
}

// ---------------------------------------------------------------------------------------
// Shared tiling parameters for variants 1 and 2.
// ---------------------------------------------------------------------------------------
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int PAD = 8;             // +8 bf16 = +16 bytes per row: kills bank conflicts
constexpr int A_LD = BK + PAD;     // 40 elements (80 bytes) per smem row of A
constexpr int B_LD = BN + PAD;     // 136 elements (272 bytes) per smem row of B
constexpr int TILE_THREADS = 256;  // 8 warps
constexpr int WARPS_M = 2;         // warp grid 2 (M) x 4 (N)
constexpr int WARPS_N = 4;
constexpr int WARP_TILE_M = BM / WARPS_M;      // 64
constexpr int WARP_TILE_N = BN / WARPS_N;      // 32
constexpr int FRAGS_M = WARP_TILE_M / WMMA_M;  // 4
constexpr int FRAGS_N = WARP_TILE_N / WMMA_N;  // 2

static_assert(WARPS_M * WARPS_N * 32 == TILE_THREADS, "warp layout must match block size");
static_assert(A_LD % 8 == 0 && B_LD % 8 == 0, "WMMA ldm must be a multiple of 8 bf16");

// A tile: BM x BK = 4096 bf16 = 512 chunks of 8; B tile: BK x BN = 4096 bf16 = 512 chunks.
constexpr int A_CHUNKS_PER_ROW = BK / 8;                         // 4
constexpr int B_CHUNKS_PER_ROW = BN / 8;                         // 16
constexpr int CHUNKS_PER_THREAD = (BM * BK / 8) / TILE_THREADS;  // 2
static_assert(CHUNKS_PER_THREAD * TILE_THREADS == BM * BK / 8, "A tile must divide evenly");
static_assert(CHUNKS_PER_THREAD * TILE_THREADS == BK * BN / 8, "B tile must divide evenly");

// Bounds-checked (zero-filled) register-path tile load used by variant 1.
__device__ __forceinline__ void load_tile_sync(__nv_bfloat16 (*As)[A_LD], __nv_bfloat16 (*Bs)[B_LD],
                                               const __nv_bfloat16* __restrict__ A,
                                               const __nv_bfloat16* __restrict__ B, int M, int N,
                                               int K, int bm, int bn, int k0) {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int i = 0; i < CHUNKS_PER_THREAD; ++i) {
        const int c = threadIdx.x + i * TILE_THREADS;
        // A: row = c / 4, col = (c % 4) * 8
        {
            const int row = c / A_CHUNKS_PER_ROW;
            const int col = (c % A_CHUNKS_PER_ROW) * 8;
            const int grow = bm + row;
            const int gcol = k0 + col;
            uint4 v = zero;
            if (grow < M && gcol < K) {  // K % 16 == 0 => a chunk is fully in or fully out
                v = *reinterpret_cast<const uint4*>(A + static_cast<size_t>(grow) * K + gcol);
            }
            *reinterpret_cast<uint4*>(&As[row][col]) = v;
        }
        // B: row = c / 16, col = (c % 16) * 8
        {
            const int row = c / B_CHUNKS_PER_ROW;
            const int col = (c % B_CHUNKS_PER_ROW) * 8;
            const int grow = k0 + row;
            const int gcol = bn + col;
            uint4 v = zero;
            if (grow < K && gcol < N) {  // N % 16 == 0 => a chunk is fully in or fully out
                v = *reinterpret_cast<const uint4*>(B + static_cast<size_t>(grow) * N + gcol);
            }
            *reinterpret_cast<uint4*>(&Bs[row][col]) = v;
        }
    }
}

// In-bounds cp.async tile load used by variant 2 (host guarantees M,N % 128 == 0, K % 32 == 0).
__device__ __forceinline__ void load_tile_async(__nv_bfloat16 (*As)[A_LD],
                                                __nv_bfloat16 (*Bs)[B_LD],
                                                const __nv_bfloat16* __restrict__ A,
                                                const __nv_bfloat16* __restrict__ B, int N, int K,
                                                int bm, int bn, int k0) {
#pragma unroll
    for (int i = 0; i < CHUNKS_PER_THREAD; ++i) {
        const int c = threadIdx.x + i * TILE_THREADS;
        {
            const int row = c / A_CHUNKS_PER_ROW;
            const int col = (c % A_CHUNKS_PER_ROW) * 8;
            cp_async_16(&As[row][col], A + static_cast<size_t>(bm + row) * K + k0 + col);
        }
        {
            const int row = c / B_CHUNKS_PER_ROW;
            const int col = (c % B_CHUNKS_PER_ROW) * 8;
            cp_async_16(&Bs[row][col], B + static_cast<size_t>(k0 + row) * N + bn + col);
        }
    }
}

// One BK=32 step: two k-slices of 16, FRAGS_M x FRAGS_N mma per slice per warp.
__device__ __forceinline__ void compute_tile(FragC (&acc)[FRAGS_M][FRAGS_N],
                                             __nv_bfloat16 (*As)[A_LD], __nv_bfloat16 (*Bs)[B_LD],
                                             int warp_m, int warp_n) {
#pragma unroll
    for (int kk = 0; kk < BK; kk += WMMA_K) {
        FragA a_frag[FRAGS_M];
        FragB b_frag[FRAGS_N];
#pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
            wmma::load_matrix_sync(a_frag[i], &As[warp_m * WARP_TILE_M + i * WMMA_M][kk], A_LD);
        }
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::load_matrix_sync(b_frag[j], &Bs[kk][warp_n * WARP_TILE_N + j * WMMA_N], B_LD);
        }
#pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }
    }
}

// Epilogue: stage one 16x16 fp32 fragment per warp at a time, convert to bf16, vector store.
__device__ __forceinline__ void store_tile(FragC (&acc)[FRAGS_M][FRAGS_N], float* warp_scratch,
                                           __nv_bfloat16* __restrict__ C, int M, int N, int bm,
                                           int bn, int warp_m, int warp_n, int lane) {
    const int r = lane >> 1;
    const int c0 = (lane & 1) * 8;
#pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::store_matrix_sync(warp_scratch, acc[i][j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            const int grow = bm + warp_m * WARP_TILE_M + i * WMMA_M + r;
            const int gcol = bn + warp_n * WARP_TILE_N + j * WMMA_N + c0;
            if (grow < M && gcol < N) {  // N % 16 == 0 => the 8-wide chunk is fully inside
                *reinterpret_cast<uint4*>(C + static_cast<size_t>(grow) * N + gcol) =
                    pack8_bf16(warp_scratch + r * WMMA_N + c0);
            }
            __syncwarp();
        }
    }
}

// ---------------------------------------------------------------------------------------
// Variant 1: shared-memory staged 128x128x32 tile, single buffer.
// ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE_THREADS)
    hgemm_v1_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    __shared__ __align__(32) __nv_bfloat16 As[BM][A_LD];                    // 10,240 B
    __shared__ __align__(32) __nv_bfloat16 Bs[BK][B_LD];                    //  8,704 B
    __shared__ __align__(32) float Cs[TILE_THREADS / 32][WMMA_M * WMMA_N];  // 8,192 B

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    FragC acc[FRAGS_M][FRAGS_N];
#pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tile_sync(As, Bs, A, B, M, N, K, bm, bn, k0);
        __syncthreads();
        compute_tile(acc, As, Bs, warp_m, warp_n);
        __syncthreads();
    }

    store_tile(acc, Cs[warp_id], C, M, N, bm, bn, warp_m, warp_n, lane);
}

// ---------------------------------------------------------------------------------------
// Variant 2: two-stage cp.async pipeline. Global->shared copies for tile k+1 are in flight
// while the tensor cores chew on tile k.
// ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE_THREADS)
    hgemm_v2_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    __shared__ __align__(32) __nv_bfloat16 As[2][BM][A_LD];  // 20,480 B
    __shared__ __align__(32) __nv_bfloat16 Bs[2][BK][B_LD];  // 17,408 B
    __shared__ __align__(
        32) float Cs[TILE_THREADS / 32][WMMA_M * WMMA_N];  // 8,192 B  (46,080 B total)

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    FragC acc[FRAGS_M][FRAGS_N];
#pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }

    const int num_tiles = K / BK;

    // Prologue: kick off tile 0.
    load_tile_async(As[0], Bs[0], A, B, N, K, bm, bn, 0);
    cp_async_commit();

    for (int t = 0; t < num_tiles; ++t) {
        const int cur = t & 1;
        if (t + 1 < num_tiles) {
            load_tile_async(As[cur ^ 1], Bs[cur ^ 1], A, B, N, K, bm, bn, (t + 1) * BK);
            cp_async_commit();
            cp_async_wait<1>();  // everything except the newest group (tile t+1) has landed
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();  // make tile t visible to all warps
        compute_tile(acc, As[cur], Bs[cur], warp_m, warp_n);
        __syncthreads();  // everyone is done reading stage `cur` before it is refilled at t+1
    }

    store_tile(acc, Cs[warp_id], C, M, N, bm, bn, warp_m, warp_n, lane);
}

// ---------------------------------------------------------------------------------------
// Variant 3: what the WMMA rungs leave on the table, taken back.
//   * raw mma.sync.m16n8k16 + ldmatrix instead of WMMA: fragments come straight out of
//     shared memory in one instruction, and the epilogue writes bf16 pairs straight from
//     the accumulator registers (no smem staging);
//   * XOR-swizzled smem tiles instead of padded ones: bank-conflict-free ldmatrix on both
//     operands with zero wasted bytes;
//   * a 3-stage cp.async pipeline (two tiles in flight) instead of 2 stages, so DRAM latency
//     is covered even when the tensor cores drain a 128x128x32 tile in under a microsecond;
//   * one __syncthreads per BK tile, no epilogue barriers;
//   * the tiles of the last, partial wave are split along K over the idle SMs (see `launch`);
//   * the tile shrinks to 64x128 or 64x64 when a 128x128 grid would not fill the card, and
//     rows past M are zero-filled, so decode shapes (M = 16..64) run on the same kernel.
// Block tile 128x128xBK, 8 warps as 2 (M) x 4 (N), warp tile 64x32 = 4 x 4 mma tiles.
// Host guarantees M,N % 128 == 0, K % BK == 0.
// ---------------------------------------------------------------------------------------
namespace v3 {

using hgemm_tile::smem_bytes;
using hgemm_tile::THREADS;

// Work assignment (see `plan` below). Blocks [0, dp_tiles) each own one full output tile;
// blocks from dp_tiles on split the remaining tiles `split` ways along K, accumulate their
// K-slice into the fp32 workspace `ws` with atomics, and the last slice to finish a tile
// converts it to bf16 (`counters` is one arrival counter per tail tile, zeroed with `ws`).
struct Sched {
    int tiles_n;   // tiles along N (tile index = tm * tiles_n + tn)
    int dp_tiles;  // tiles handled whole
    int split;     // K-slices per tail tile (1: no tail)
    float* ws;     // (tiles - dp_tiles) x BM x BN fp32
    int* counters;
};

// The tile (swizzle, pipeline, mma loop, register epilogue) is hgemm_tile.cuh. Tile rows
// past M are zero-filled on the way in and skipped on the way out, so any M % 16 == 0 works
// with any BM; N % BN == 0 and K % BK == 0 are required.
template <int BM, int BN, int BK, int STAGES>
__global__ void __launch_bounds__(THREADS)
    hgemm_v3_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K, Sched sched) {
    using Cfg = hgemm_tile::Cfg<BM, BN, BK, STAGES>;
    constexpr int WM = Cfg::WM, WN = Cfg::WN, MT = Cfg::MT, NT = Cfg::NT;
    extern __shared__ __align__(128) unsigned char smem_raw[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / hgemm_tile::WARPS_N;
    const int wn = warp % hgemm_tile::WARPS_N;

    const int KT = K / BK;
    int tile, kt_begin, kt_end, slice = 0;
    if (static_cast<int>(blockIdx.x) < sched.dp_tiles) {
        tile = blockIdx.x;
        kt_begin = 0;
        kt_end = KT;
    } else {
        const int r = blockIdx.x - sched.dp_tiles;
        tile = sched.dp_tiles + r / sched.split;
        slice = r % sched.split;
        kt_begin = static_cast<int>(static_cast<long long>(slice) * KT / sched.split);
        kt_end = static_cast<int>(static_cast<long long>(slice + 1) * KT / sched.split);
    }
    const int bm = (tile / sched.tiles_n) * BM;
    const int bn = (tile % sched.tiles_n) * BN;
    const int m_valid = M - bm;  // rows of this tile that exist

    typename Cfg::Acc acc;
    hgemm_tile::mainloop<Cfg>(A, B, N, K, bm, bn, m_valid, kt_begin, kt_end - kt_begin, smem_raw,
                              acc);

    if (tile < sched.dp_tiles) {
        hgemm_tile::store_bf16<Cfg>(acc, C, M, N, bm, bn);
        return;
    }

    // Tail tile: accumulate this K-slice into the fp32 workspace tile. The adds are
    // performed at L2, so no ordering between the slices is needed.
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    float* wt = sched.ws + static_cast<size_t>(tile - sched.dp_tiles) * BM * BN;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) {
            const int r0 = wm * WM + mi * 16 + g;
            const int c0 = wn * WN + nj * 8 + c2;
            if (r0 >= m_valid) continue;
            atomicAdd(wt + r0 * BN + c0, acc[mi][nj][0]);
            atomicAdd(wt + r0 * BN + c0 + 1, acc[mi][nj][1]);
            if (r0 + 8 >= m_valid) continue;
            atomicAdd(wt + (r0 + 8) * BN + c0, acc[mi][nj][2]);
            atomicAdd(wt + (r0 + 8) * BN + c0 + 1, acc[mi][nj][3]);
        }
    }
    // Last slice to arrive converts the finished tile to bf16. The fence orders every
    // thread's adds before the counter; the smem flag broadcasts the outcome to the block.
    __shared__ int s_last;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(sched.counters + (tile - sched.dp_tiles), 1) == sched.split - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int rows_here = m_valid < BM ? m_valid : BM;
    for (int i = tid; i < rows_here * BN / 2; i += THREADS) {  // two elements per thread per step
        const int r = (2 * i) / BN;
        const int c = (2 * i) % BN;
        const float2 v = __ldcg(reinterpret_cast<const float2*>(wt + r * BN + c));  // L2, not L1
        *reinterpret_cast<__nv_bfloat162*>(C + static_cast<size_t>(bm + r) * N + bn + c) =
            __floats2bfloat162_rn(v.x, v.y);
    }
}

// Pipeline configuration of variant 3 (tuned on the RTX 5090, see docs/design/hgemm.md). The
// tile is chosen per call: 128x128 when the grid fills the card, 64x128 / 64x64 for small or
// decode-sized (M <= 64) problems, where the 128-row tile would leave most SMs idle.
constexpr int V3_BK = 32;
constexpr int V3_STAGES = 3;

// Wave quantization. With P blocks resident at once (2 per SM, 340 on the RTX 5090) a
// 4096^3 GEMM has 1024 tiles = 3.01 waves: the last 4 tiles run alone for as long as a
// full wave, a quarter of the runtime. So the tiles past the last full wave are split
// `split` ways along K across the otherwise idle blocks (split-K on the tail only, in the
// spirit of Stream-K): the extra wave then lasts 1/split of a tile instead of a whole one.
// The tail's fp32 partials go to a workspace (grown on demand, one per device, zeroed on
// the stream before the launch); the tile's last slice converts it to bf16 in-kernel.
struct Workspace {
    float* ws = nullptr;
    int* counters = nullptr;
    size_t tiles = 0;
};

template <int BM, int BN, int BK, int STAGES>
int resident_blocks() {
    constexpr int bytes = smem_bytes<BM, BN, BK, STAGES>();
    static int resident = 0;  // blocks resident per GPU; also the > 48 KB smem opt-in
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(hgemm_v3_kernel<BM, BN, BK, STAGES>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, hgemm_v3_kernel<BM, BN, BK, STAGES>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int BM, int BN, int BK, int STAGES>
void launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K,
            cudaStream_t stream) {
    constexpr int bytes = smem_bytes<BM, BN, BK, STAGES>();
    const int resident = resident_blocks<BM, BN, BK, STAGES>();

    Sched s;
    s.tiles_n = N / BN;
    const int tiles = cdiv(M, BM) * s.tiles_n;
    const int KT = K / BK;
    const int tail = tiles % resident;
    s.split = tail > 0 ? std::min(KT, resident / tail) : 1;
    if (s.split <= 1) {  // no tail, or a tail too large to be worth splitting
        s.dp_tiles = tiles;
        s.split = 1;
        s.ws = nullptr;
        s.counters = nullptr;
        hgemm_v3_kernel<BM, BN, BK, STAGES><<<tiles, THREADS, bytes, stream>>>(A, B, C, M, N, K, s);
        return;
    }
    s.dp_tiles = tiles - tail;
    static Workspace w;
    if (w.tiles < static_cast<size_t>(tail)) {
        if (w.ws) SPARK_CUDA_CHECK(cudaFree(w.ws));
        if (w.counters) SPARK_CUDA_CHECK(cudaFree(w.counters));
        w.tiles = tail;
        SPARK_CUDA_CHECK(cudaMalloc(&w.ws, w.tiles * BM * BN * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&w.counters, w.tiles * sizeof(int)));
    }
    s.ws = w.ws;
    s.counters = w.counters;
    SPARK_CUDA_CHECK(
        cudaMemsetAsync(w.ws, 0, static_cast<size_t>(tail) * BM * BN * sizeof(float), stream));
    SPARK_CUDA_CHECK(
        cudaMemsetAsync(w.counters, 0, static_cast<size_t>(tail) * sizeof(int), stream));
    const int grid = s.dp_tiles + tail * s.split;
    hgemm_v3_kernel<BM, BN, BK, STAGES><<<grid, THREADS, bytes, stream>>>(A, B, C, M, N, K, s);
}

// Tile selection: the biggest tile whose grid is at least one full wave; for anything
// smaller the 64x64 tile with the split-K tail. M <= 64 (decode shapes) always gets a 64-row
// tile: the extra rows would be zero-filled work on a problem that is bound by streaming B.
void launch_auto(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                 int K, cudaStream_t stream) {
    auto tiles = [&](int bm, int bn) { return cdiv(M, bm) * (N / bn); };
    if (M <= 64 && hgemm_decode::supports(M, N, K)) {  // decode: stream B (hgemm_decode.cu)
        hgemm_decode::launch(A, B, C, M, N, K, stream);
        return;
    }
    if (M > 64 && N % 128 == 0 &&
        tiles(128, 128) >= resident_blocks<128, 128, V3_BK, V3_STAGES>()) {
        launch<128, 128, V3_BK, V3_STAGES>(A, B, C, M, N, K, stream);
    } else if (N % 128 == 0 && tiles(64, 128) >= resident_blocks<64, 128, V3_BK, V3_STAGES>()) {
        launch<64, 128, V3_BK, V3_STAGES>(A, B, C, M, N, K, stream);
    } else {
        // Small grids: BK = 64 halves the per-tile pipeline overhead that dominates when a
        // block owns only a dozen K-tiles; decode shapes (M <= 64, bound by streaming B)
        // gain from a fourth stage, everything else loses a little to the lower occupancy.
        if (M <= 64) {
            launch<64, 64, 64, 4>(A, B, C, M, N, K, stream);
        } else {
            launch<64, 64, 64, 3>(A, B, C, M, N, K, stream);
        }
    }
}

}  // namespace v3

}  // namespace

// Variant 5 lives in hgemm_tma.cu, variant 6 in hgemm_tma_sk.cu.
bool hgemm_tma_supports(int M, int N, int K);
void hgemm_tma(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
               int K, cudaStream_t stream);
bool hgemm_tma_sk_supports(int M, int N, int K);
void hgemm_tma_sk(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                  int K, cudaStream_t stream, const HgemmEpilogue& ep);

int hgemm_num_variants() {
    return 7;
}

bool hgemm_supports(int M, int N, int K, int variant) {
    if (variant < 0 || variant >= hgemm_num_variants()) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if (N % WMMA_N != 0 || K % WMMA_K != 0) return false;
    if (M % WMMA_M != 0 && variant < 3) return false;  // variants 3, 4 and 6 zero-fill rows past M
    if (variant == 2) return M % BM == 0 && N % BN == 0 && K % BK == 0;
    if (variant == 3 || variant == 4) return N % 64 == 0 && K % 64 == 0;  // any M >= 1
    if (variant == 5) return hgemm_tma_supports(M, N, K);
    if (variant == 6) return hgemm_tma_sk_supports(M, N, K);
    return true;
}

void hgemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                int K, int variant, cudaStream_t stream, const HgemmEpilogue& ep) {
    SPARK_REQUIRE(A != nullptr && B != nullptr && C != nullptr, "hgemm: null pointer");
    SPARK_REQUIRE(M > 0 && N > 0 && K > 0, "hgemm: M, N, K must be positive");
    SPARK_REQUIRE(N % WMMA_N == 0 && K % WMMA_K == 0, "hgemm: N, K must be multiples of 16");
    SPARK_REQUIRE(M % WMMA_M == 0 || variant == 3 || variant == 4 || variant == 6,
                  "hgemm: M must be a multiple of 16");
    SPARK_REQUIRE(variant >= 0 && variant < hgemm_num_variants(), "hgemm: unknown variant");
    if (!hgemm_epi::is_plain(ep)) {
        SPARK_REQUIRE(variant == 4 || variant == 6,
                      "hgemm: the fused epilogue (bias, act, residual, swiglu) is only supported "
                      "by variants 4 and 6, which need N % 64 == 0 and K % 64 == 0");
        SPARK_REQUIRE(ep.act >= HGEMM_ACT_NONE && ep.act <= HGEMM_ACT_RELU,
                      "hgemm: unknown activation");
        SPARK_REQUIRE(!ep.swiglu || N % 2 == 0, "hgemm: swiglu needs an even N");
        SPARK_REQUIRE(ep.bias == nullptr || is_aligned16(ep.bias),
                      "hgemm: bias must be 16-byte aligned");
        SPARK_REQUIRE(ep.residual == nullptr || is_aligned16(ep.residual),
                      "hgemm: residual must be 16-byte aligned");
    }

    switch (variant) {
        case 0: {
            const dim3 grid(cdiv(N, V0_TILE), cdiv(M, V0_TILE));
            hgemm_v0_kernel<<<grid, V0_THREADS, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case 1: {
            const dim3 grid(cdiv(N, BN), cdiv(M, BM));
            hgemm_v1_kernel<<<grid, TILE_THREADS, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case 2: {
            SPARK_REQUIRE(hgemm_supports(M, N, K, 2),
                          "hgemm variant 2: requires M % 128 == 0, N % 128 == 0, K % 32 == 0");
            const dim3 grid(N / BN, M / BM);
            hgemm_v2_kernel<<<grid, TILE_THREADS, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case 3: {
            SPARK_REQUIRE(hgemm_supports(M, N, K, 3),
                          "hgemm variant 3: requires N % 64 == 0, K % 64 == 0");
            v3::launch_auto(A, B, C, M, N, K, stream);
            break;
        }
        case 4: {
            SPARK_REQUIRE(hgemm_supports(M, N, K, 4),
                          "hgemm variant 4: requires N % 64 == 0, K % 64 == 0");
            hgemm_streamk_bf16(A, B, C, M, N, K, stream, ep);
            break;
        }
        case 5:
            hgemm_tma(A, B, C, M, N, K, stream);
            break;
        case 6:
            hgemm_tma_sk(A, B, C, M, N, K, stream, ep);
            break;
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
