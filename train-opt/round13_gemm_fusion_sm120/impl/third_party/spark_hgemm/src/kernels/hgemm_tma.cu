// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// bf16 tensor-core GEMM, variant 5: TMA loads and a warp-specialized mbarrier pipeline.
//
//   C[M,N] = A[M,K] * B[K,N]      row-major, bf16 in/out, fp32 accumulate
//
// Same 128x128 block tile, 2x4 consumer warp grid, ldmatrix + mma.sync k-loop body, register
// direct epilogue and tail split-K as variant 3 (src/kernels/hgemm.cu). What changes is how the
// operand tiles reach shared memory:
//   * one producer warp (a single elected lane) issues three cp.async.bulk.tensor loads per
//     stage (a 128xBK box of A, two BKx64 boxes of B) against tensor maps built on the host;
//     the copy engine generates every address and applies the 128-byte (BK=64) or 64-byte
//     (BK=32) swizzle, which is bit for bit the XOR pattern variant 3 computes per thread;
//   * each stage has a "full" mbarrier (producer arrives with the byte count, the copy engine
//     completes it) and an "empty" mbarrier (each consumer warp arrives once it has read the
//     stage); the producer waits on "empty" before it refills a stage, the consumers wait on
//     "full" before they read one, and nothing in the block ever executes __syncthreads in
//     the k-loop;
//   * the consumer warps' instruction stream holds ldmatrix, mma.sync and one arrive per
//     stage: no cp.async, no address arithmetic for the copies.
// The stage geometry, the swizzle, the producer's issue and the consumer's stage body are in
// hgemm_tma_tile.cuh, shared with variant 6, which runs them on the Stream-K schedule.
// Host guarantees M % 128 == 0, N % 128 == 0, K % 64 == 0 and a grid of at least one wave.

#include <algorithm>
#include <cstdlib>

#include "hgemm_tma_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace v5 {

using namespace hgemm_tma_tile;

constexpr int TAIL_BARRIER = 1;  // named barrier id for the 256 consumer threads

// The stages sit at the start of the dynamic region (which starts 1 KB-aligned when the kernel
// declares no static shared memory; checked in the kernel), the barriers go after them. The
// 5090 allows 101,376 B per block and 102,400 B per SM with 1 KB of it reserved per block, so
// two blocks per SM need at most 49 KB each.
template <int BK, int STAGES>
constexpr int smem_bytes() {
    return round_kb(STAGES * stage_bytes<BK>() + 2 * STAGES * 8 + 16);
}

// Work assignment, identical to variant 3's: blocks [0, dp_tiles) own one output tile each;
// the rest split the remaining tiles `split` ways along K into the fp32 workspace `ws`, and
// the last slice to arrive at a tile's counter converts it to bf16.
struct Sched {
    int tiles_n;
    int dp_tiles;
    int split;
    float* ws;
    int* counters;
};

template <int BK, int STAGES, int MIN_BLOCKS>
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
    hgemm_v5_kernel(const __grid_constant__ CUtensorMap tmA,
                    const __grid_constant__ CUtensorMap tmB, __nv_bfloat16* __restrict__ C, int M,
                    int N, int K, Sched sched) {
    static_assert(BK == 32 || BK == 64, "the swizzle helpers assume 64 B or 128 B rows of A");
    constexpr int STAGE_BYTES = stage_bytes<BK>();

    extern __shared__ __align__(1024) unsigned char smem[];
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem + STAGES * STAGE_BYTES);
    uint64_t* empty_bar = full_bar + STAGES;
    int* s_last = reinterpret_cast<int*>(empty_bar + STAGES);
    // The swizzle pattern the copy engine applies is keyed on the absolute smem address, so
    // a misaligned base would put the data where the ldmatrix addressing does not expect it.
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

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
    const int nkt = kt_end - kt_begin;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);                // the producer's arrive.expect_tx
            mbar_init(&empty_bar[s], CONSUMER_WARPS);  // one arrive per consumer warp
        }
        fence_mbar_init();  // the copy engine must see the initialized barriers
    }
    __syncthreads();

    if (warp == CONSUMER_WARPS) {
        // Producer warp. Lane 0 does all of it; the rest of the warp leaves. Stage s is used
        // for k-tiles s, s+STAGES, ...; its n-th use waits for the consumers' n-1-th release
        // (parity (n-1)&1) and then registers STAGE_BYTES of transactions on "full".
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmB);
            for (int kt = 0; kt < nkt; ++kt) {
                const int s = kt % STAGES;
                const unsigned use = kt / STAGES;
                if (kt >= STAGES) mbar_wait(&empty_bar[s], (use - 1) & 1);
                issue_stage<BK>(smem + s * STAGE_BYTES, &tmA, &tmB, &full_bar[s],
                                (kt_begin + kt) * BK, bm, bn);
            }
        }
        return;
    }

    // Consumer warps: variant 3's k-loop body on the stage the producer has filled.
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;

    Acc acc;
    zero_acc(acc);

    for (int kt = 0; kt < nkt; ++kt) {
        const int s = kt % STAGES;
        mbar_wait(&full_bar[s], (kt / STAGES) & 1);
        consume_stage<BK>(acc, smem + s * STAGE_BYTES, wm, wn, lane);
        // The ldmatrix reads of this stage went through the generic proxy and the refill will
        // come through the async proxy; the proxy fence orders the reads before anything the
        // copy engine does after the release, then one arrive per warp hands the stage back.
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[s]);
    }

    // Epilogue, as in variant 3. Each lane owns (row g, cols 2c..2c+1) and (row g+8, same).
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    if (tile < sched.dp_tiles) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const int row = bm + wm * WM + mi * 16 + g;
                const int col = bn + wn * WN + nj * 8 + c2;
                __nv_bfloat16* p0 = C + static_cast<size_t>(row) * N + col;
                __nv_bfloat16* p1 = p0 + static_cast<size_t>(8) * N;
                *reinterpret_cast<__nv_bfloat162*>(p0) =
                    __floats2bfloat162_rn(acc[mi][nj][0], acc[mi][nj][1]);
                *reinterpret_cast<__nv_bfloat162*>(p1) =
                    __floats2bfloat162_rn(acc[mi][nj][2], acc[mi][nj][3]);
            }
        }
        return;
    }

    // Tail tile: this K-slice goes into the fp32 workspace with L2 atomics; the last slice
    // converts. The producer warp has left, so the consumers sync on a named barrier.
    float* wt = sched.ws + static_cast<size_t>(tile - sched.dp_tiles) * BM * BN;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) {
            const int r0 = wm * WM + mi * 16 + g;
            const int c0 = wn * WN + nj * 8 + c2;
            atomicAdd(wt + r0 * BN + c0, acc[mi][nj][0]);
            atomicAdd(wt + r0 * BN + c0 + 1, acc[mi][nj][1]);
            atomicAdd(wt + (r0 + 8) * BN + c0, acc[mi][nj][2]);
            atomicAdd(wt + (r0 + 8) * BN + c0 + 1, acc[mi][nj][3]);
        }
    }
    __threadfence();
    named_barrier_sync(TAIL_BARRIER, CONSUMERS);
    if (tid == 0) {
        *s_last = atomicAdd(sched.counters + (tile - sched.dp_tiles), 1) == sched.split - 1;
    }
    named_barrier_sync(TAIL_BARRIER, CONSUMERS);
    if (!*s_last) return;
    __threadfence();
    for (int i = tid; i < BM * BN / 2; i += CONSUMERS) {
        const int r = (2 * i) / BN;
        const int c = (2 * i) % BN;
        const float2 v = __ldcg(reinterpret_cast<const float2*>(wt + r * BN + c));
        *reinterpret_cast<__nv_bfloat162*>(C + static_cast<size_t>(bm + r) * N + bn + c) =
            __floats2bfloat162_rn(v.x, v.y);
    }
}

// ---- host side --------------------------------------------------------------------------

struct Workspace {
    float* ws = nullptr;
    int* counters = nullptr;
    size_t tiles = 0;
};

template <int BK, int STAGES, int MIN_BLOCKS>
int resident_blocks() {
    constexpr int bytes = smem_bytes<BK, STAGES>();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(hgemm_v5_kernel<BK, STAGES, MIN_BLOCKS>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, hgemm_v5_kernel<BK, STAGES, MIN_BLOCKS>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int BK, int STAGES, int MIN_BLOCKS>
void launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K,
            cudaStream_t stream) {
    constexpr int bytes = smem_bytes<BK, STAGES>();
    const int resident = resident_blocks<BK, STAGES, MIN_BLOCKS>();

    CUtensorMap tmA, tmB;
    make_maps<BK>(A, B, M, N, K, tmA, tmB);

    Sched s;
    s.tiles_n = N / BN;
    const int tiles = (M / BM) * s.tiles_n;
    const int KT = K / BK;
    const int tail = tiles % resident;
    s.split = tail > 0 ? std::min(KT, resident / tail) : 1;
    if (s.split <= 1) {
        s.dp_tiles = tiles;
        s.split = 1;
        s.ws = nullptr;
        s.counters = nullptr;
        hgemm_v5_kernel<BK, STAGES, MIN_BLOCKS>
            <<<tiles, THREADS, bytes, stream>>>(tmA, tmB, C, M, N, K, s);
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
    hgemm_v5_kernel<BK, STAGES, MIN_BLOCKS>
        <<<grid, THREADS, bytes, stream>>>(tmA, tmB, C, M, N, K, s);
}

// Pipeline configurations (BK, stages, blocks per SM the compiler is asked to fit), see the
// sweep in docs/design/hgemm.md. Two are used: index 0 when the operands stream from DRAM
// and index 2 when A and B fit in the L2 together. SPARK_HGEMM_V5_CONFIG=<index> forces one.
using LaunchFn = void (*)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int,
                          cudaStream_t);
constexpr LaunchFn CONFIGS[] = {
    launch<64, 2, 1>,  // 0: 64 KB of smem, one block per SM
    launch<64, 3, 1>,  // 1: 96 KB, one block per SM
    launch<32, 3, 2>,  // 2: variant 3's tile and depth, 48 KB, two blocks per SM (96 regs)
    launch<32, 2, 2>,  // 3: 32 KB, two blocks per SM
    launch<32, 3, 1>,  // 4: like 2 without the register cap
    launch<32, 6, 1>,  // 5: 96 KB, one block per SM
};
constexpr int NUM_CONFIGS = static_cast<int>(sizeof(CONFIGS) / sizeof(CONFIGS[0]));
constexpr int CONFIG_DRAM = 0;
constexpr int CONFIG_L2 = 2;

int forced_config() {
    static int idx = -2;
    if (idx == -2) {
        idx = -1;
        if (const char* e = std::getenv("SPARK_HGEMM_V5_CONFIG")) {
            const int v = std::atoi(e);
            if (v >= 0 && v < NUM_CONFIGS) idx = v;
        }
    }
    return idx;
}

int l2_bytes() {
    static int bytes = 0;
    if (bytes == 0) {
        int dev = 0;
        SPARK_CUDA_CHECK(cudaGetDevice(&dev));
        SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, dev));
        if (bytes <= 0) bytes = 96 << 20;
    }
    return bytes;
}

void launch_auto(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                 int K, cudaStream_t stream) {
    int cfg = forced_config();
    if (cfg < 0) {
        const double operands = 2.0 * (static_cast<double>(M) * K + static_cast<double>(K) * N);
        cfg = operands <= static_cast<double>(l2_bytes()) ? CONFIG_L2 : CONFIG_DRAM;
    }
    CONFIGS[cfg](A, B, C, M, N, K, stream);
}

}  // namespace v5

}  // namespace

bool hgemm_tma_supports(int M, int N, int K) {
    // One wave of 128x128 tiles at one block per SM; smaller grids are variant 3's job (it
    // picks a 64-row tile for them) and would only lose here.
    return M % v5::BM == 0 && N % v5::BN == 0 && K % 64 == 0 &&
           (M / v5::BM) * (N / v5::BN) >= num_sms();
}

void hgemm_tma(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
               int K, cudaStream_t stream) {
    SPARK_REQUIRE(hgemm_tma_supports(M, N, K),
                  "hgemm variant 5: requires M % 128 == 0, N % 128 == 0, K % 64 == 0 and at "
                  "least one 128x128 tile per SM");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(B),
                  "hgemm variant 5: A and B must be 16-byte aligned");
    v5::launch_auto(A, B, C, M, N, K, stream);
}

}  // namespace spark
