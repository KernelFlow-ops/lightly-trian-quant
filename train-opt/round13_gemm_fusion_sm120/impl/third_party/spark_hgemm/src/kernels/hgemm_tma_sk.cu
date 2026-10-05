// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// hgemm variant 6: variant 5's TMA mainloop on variant 4's Stream-K schedule.
//
//   C[M,N] = A[M,K] * B[K,N]      row-major, bf16 in/out, fp32 accumulate
//
// A persistent grid of 288-thread blocks (one or two per SM) walks the pieces that variant
// 4's plan hands out (hgemm_streamk.cuh): static (tile, k-range) ranges up to a few waves,
// whole tiles from a queue plus geometrically shrinking K-passes above that, the grouped
// raster, and the K-ordered chain fixup through one fp32 slot and one epoch-tagged flag per
// tile, no memset and the same bits every run. The mainloop is variant 5's
// (hgemm_tma_tile.cuh): one producer lane issues three cp.async.bulk.tensor boxes per stage,
// eight consumer warps wait on the stage's "full" mbarrier, run ldmatrix + mma.sync, fence
// the proxy and arrive on "empty". No __syncthreads anywhere after the barrier init.
//
// What is new is the handoff across pieces. The producer lane owns the schedule: it walks
// the block's static range or takes items from the queue, and publishes each piece (tile,
// kt_begin, kt_end) through a two-deep ring of int4 slots in shared memory, each ring slot
// with its own full/empty mbarrier pair, before it issues the piece's first stage. The
// consumers read the ring in order, so both sides walk the same sequence without a block
// barrier; a tile of -1 ends it. Stage and parity bookkeeping is one running k-tile counter
// per side that never resets between pieces, so the pipeline stays full across a piece
// boundary: while the consumers add the slot, store bf16 or publish their partial for piece
// i, the producer is already filling stages for piece i+1, up to STAGES k-tiles ahead,
// bounded by the "empty" barriers. That is the prologue-hiding variant 4 could not do.
//
// Proxy fences. The TMA writes shared memory through the async proxy; the one generic-proxy
// access to a stage is the consumers' ldmatrix, fenced before the release as in variant 5.
// The fixup reads and writes its slot in global memory straight from registers (__ldcg and
// float4 stores) and the epilogue stores bf16 from registers, so neither touches shared
// memory and neither needs a fence. The item ring is written and read through the generic
// proxy on both sides and ordered by its mbarriers.
//
// Requires N % 64 == 0, K % 64 == 0, any M >= 1. The TMA tile runs when N % 128 == 0 and each
// block's share of the (tile, k-step) space is at least kMinShare k-steps; rows past M are
// zero-filled by the copy engine (an out-of-range box row) and skipped by the epilogue.
// Decode shapes (M <= 64) go to hgemm_decode.cu and everything else to variant 4's 64-row
// tiles, the same routing variant 4 does itself.
//
// The fused epilogue (hgemm_epilogue.cuh) is applied by whichever piece stores a tile: a
// whole tile, or the piece that ends a Stream-K chain after it has added the slot. The
// partials that pass through the slot are plain fp32 sums of A B, so bias, activation,
// residual and the gate/up product all happen once, on the finished sum. The FUSED
// instantiation is separate from the plain one so the plain kernel is unchanged.

#include <cstdlib>

#include "hgemm_epilogue.cuh"
#include "hgemm_internal.cuh"
#include "hgemm_streamk.cuh"
#include "hgemm_tile.cuh"  // declares hgemm_streamk_bf16
#include "hgemm_tma_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace v6 {

using namespace hgemm_tma_tile;
using hgemm_sk::ld_acquire_gpu;
using hgemm_sk::Params;
using hgemm_sk::pass_range;
using hgemm_sk::range_start;
using hgemm_sk::raster;
using hgemm_sk::slot_idx;
using hgemm_sk::st_release_gpu;

constexpr int ITEM_DEPTH = 2;     // pieces the producer may publish ahead of the consumers
constexpr int FIXUP_BARRIER = 1;  // named barrier id for the 256 consumer threads

// Layout of the dynamic region: the stages (1 KB aligned, see hgemm_tma_tile.cuh), then the
// item ring (16 B each), then the barriers: full/empty per stage, full/empty per ring slot,
// and one "done" barrier for the serialized-prologue experiment.
template <int BK, int STAGES>
constexpr int smem_bytes() {
    return round_kb(STAGES * stage_bytes<BK>() + ITEM_DEPTH * 16 +
                    (2 * STAGES + 2 * ITEM_DEPTH + 1) * 8);
}

template <int BK, int STAGES, int MIN_BLOCKS, bool FUSED>
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
    hgemm_v6_kernel(const __grid_constant__ CUtensorMap tmA,
                    const __grid_constant__ CUtensorMap tmB, __nv_bfloat16* __restrict__ C, int M,
                    int N, int K, Params p, int serial, hgemm_epi::Ep ep) {
    static_assert(BK == 32 || BK == 64, "the swizzle helpers assume 64 B or 128 B rows of A");
    constexpr int STAGE_BYTES = stage_bytes<BK>();

    extern __shared__ __align__(1024) unsigned char smem[];
    int4* items = reinterpret_cast<int4*>(smem + STAGES * STAGE_BYTES);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(items + ITEM_DEPTH);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* item_full = empty_bar + STAGES;
    uint64_t* item_empty = item_full + ITEM_DEPTH;
    uint64_t* done_bar = item_empty + ITEM_DEPTH;
    // The swizzle the copy engine applies is keyed on the absolute smem address.
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);                // the producer's arrive.expect_tx
            mbar_init(&empty_bar[s], CONSUMER_WARPS);  // one arrive per consumer warp
        }
#pragma unroll
        for (int i = 0; i < ITEM_DEPTH; ++i) {
            mbar_init(&item_full[i], 1);                // the producer's publish
            mbar_init(&item_empty[i], CONSUMER_WARPS);  // one arrive per warp once it has read
        }
        mbar_init(done_bar, CONSUMER_WARPS);  // one arrive per warp after a piece's epilogue
        fence_mbar_init();
    }
    __syncthreads();  // the only block-wide barrier in the kernel

    if (warp == CONSUMER_WARPS) {
        // Producer warp: lane 0 owns the schedule and the loads, the rest of the warp leaves.
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmB);
            unsigned g_kt = 0;   // k-tiles issued so far, across pieces
            unsigned n_pub = 0;  // pieces published so far
            // Publish (tile, kb, ke) through the ring, then issue its k-tiles. Ring slot r is
            // used for pieces r, r+DEPTH, ...; its n-th use waits for the consumers' n-1-th
            // release, exactly like a stage. Stage g_kt % STAGES is used for the
            // g_kt / STAGES-th time whichever piece it belongs to.
            auto piece = [&](int tile, int kb, int ke) {
                const int r = n_pub % ITEM_DEPTH;
                const unsigned use = n_pub / ITEM_DEPTH;
                if (n_pub >= ITEM_DEPTH) mbar_wait(&item_empty[r], (use - 1) & 1);
                items[r] = make_int4(tile, kb, ke, 0);
                mbar_arrive(&item_full[r]);
                ++n_pub;
                if (tile < 0) return;
                // Experiment (SPARK_HGEMM_V6_SERIAL=1): hold the next piece's first stage
                // until the consumers have finished the previous piece's epilogue, which is
                // what a per-piece launch or variant 4's prologue amounts to.
                if (serial && n_pub >= 2) mbar_wait(done_bar, (n_pub - 2) & 1);
                int tm, tn;
                raster(p, tile, tm, tn);
                const int bm = tm * BM, bn = tn * BN;
                for (int kt = kb; kt < ke; ++kt, ++g_kt) {
                    const int s = g_kt % STAGES;
                    const unsigned use_s = g_kt / STAGES;
                    if (g_kt >= STAGES) mbar_wait(&empty_bar[s], (use_s - 1) & 1);
                    issue_stage<BK>(smem + s * STAGE_BYTES, &tmA, &tmB, &full_bar[s], kt * BK, bm,
                                    bn);
                }
            };

            if (p.passes == 0) {
                // Static ranges, walked from the top: the piece that starts a tile (and
                // never waits) comes first, the piece that ends one last. dp_tiles is 0, so
                // a tile's raster index is its slot index.
                const int it_begin = range_start(p, blockIdx.x);
                int e = range_start(p, blockIdx.x + 1);
                while (e > it_begin) {
                    const int tile = (e - 1) / p.KT;
                    const int k0 = tile * p.KT;
                    const int s = max(it_begin, k0);
                    piece(tile, s - k0, e - k0);
                    e = s;
                }
            } else {
                // Queue: whole tiles first, then the Stream-K region pass by pass, tile
                // interleaved, so a pass's predecessor in K was taken a whole region earlier.
                const int sk_tiles = p.tiles - p.dp_tiles;
                const int n_items = p.dp_tiles + sk_tiles * p.passes;
                for (;;) {
                    const int item = static_cast<int>(atomicAdd(p.queue, 1u));
                    if (item >= n_items) break;
                    if (item < p.dp_tiles) {
                        piece(item, 0, p.KT);
                    } else {
                        const int q = item - p.dp_tiles;
                        int kb, ke;
                        pass_range(p, q / sk_tiles, kb, ke);
                        piece(p.dp_tiles + q % sk_tiles, kb, ke);
                    }
                }
                // The last block out resets the queue for the next launch. Every block has
                // made its final, failing grab before it counts itself done.
                __threadfence();
                if (atomicAdd(p.queue + 1, 1u) == static_cast<unsigned>(p.grid) - 1) {
                    p.queue[0] = 0u;
                    p.queue[1] = 0u;
                    __threadfence();
                }
            }
            piece(-1, 0, 0);  // end of work
        }
        return;
    }

    // Consumer warps.
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    unsigned g_kt = 0;    // k-tiles consumed so far, across pieces
    unsigned n_item = 0;  // pieces taken from the ring so far
    Acc acc;

    for (;;) {
        const int r = n_item % ITEM_DEPTH;
        mbar_wait(&item_full[r], (n_item / ITEM_DEPTH) & 1);
        const int4 it = items[r];
        __syncwarp();  // every lane has its copy before the warp hands the ring slot back
        if (lane == 0) mbar_arrive(&item_empty[r]);
        ++n_item;
        if (it.x < 0) break;
        const int tile = it.x, kb = it.y, ke = it.z;
        int tm, tn;
        raster(p, tile, tm, tn);
        const int bm = tm * BM, bn = tn * BN;
        const int m_valid = M - bm;

        // A piece that ends its tile applies the epilogue; its bias and residual are loaded
        // now, before the k-loop, so their DRAM round trip hides under the mma work (40
        // registers, dead again after the store).
        const hgemm_epi::Geo geo{M, N, bm + wm * WM + g, bn + wn * WN + c2};
        hgemm_epi::Pre<MT, NT> pre;
        if constexpr (FUSED) {
            if (ke == p.KT) hgemm_epi::prefetch(pre, ep, geo);
        }

        zero_acc(acc);
        for (int kt = kb; kt < ke; ++kt, ++g_kt) {
            const int s = g_kt % STAGES;
            mbar_wait(&full_bar[s], (g_kt / STAGES) & 1);
            consume_stage<BK>(acc, smem + s * STAGE_BYTES, wm, wn, lane);
            // ldmatrix read the stage through the generic proxy; the refill comes through the
            // async proxy. The fence orders the reads before the release, one arrive per warp.
            fence_proxy_async_smem();
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty_bar[s]);
        }

        // Fixup, as in variant 4, with a named barrier in place of __syncthreads: the
        // producer lane is not part of it (it is already loading the next piece).
        bool finish = true;
        if (kb > 0 || ke < p.KT) {
            const int tile_r = tile - p.dp_tiles;
            float4* slot =
                reinterpret_cast<float4*>(p.slots + static_cast<size_t>(tile_r) * (BM * BN));
            unsigned long long* flag = p.flags + tile_r;
            const unsigned long long tag = static_cast<unsigned long long>(p.epoch) << 32;
            if (kb > 0) {
                // Everything before kb is in the slot once the flag says so.
                if (tid == 0) {
                    while (ld_acquire_gpu(flag) != (tag | static_cast<unsigned>(kb))) {
                    }
                }
                named_barrier_sync(FIXUP_BARRIER, CONSUMERS);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    if (wm * WM + mi * 16 >= m_valid) continue;  // the whole m16 tile is past M
#pragma unroll
                    for (int nj = 0; nj < NT; ++nj) {
                        const float4 v = __ldcg(slot + slot_idx<MT, NT>(warp, lane, mi, nj));
                        acc[mi][nj][0] += v.x;
                        acc[mi][nj][1] += v.y;
                        acc[mi][nj][2] += v.z;
                        acc[mi][nj][3] += v.w;
                    }
                }
            }
            if (ke < p.KT) {
                // Hand the running sum to the piece that continues the tile.
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    if (wm * WM + mi * 16 >= m_valid) continue;
#pragma unroll
                    for (int nj = 0; nj < NT; ++nj)
                        slot[slot_idx<MT, NT>(warp, lane, mi, nj)] = make_float4(
                            acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3]);
                }
                __threadfence();
                named_barrier_sync(FIXUP_BARRIER, CONSUMERS);
                if (tid == 0) st_release_gpu(flag, tag | static_cast<unsigned>(ke));
                finish = false;
            }
        }
        if (finish) {
            // Epilogue from registers: each lane owns (row g, cols 2c..2c+1) and (row g+8,
            // same) of every m16n8 fragment. M % 16 == 0 is not required: row and row + 8
            // are checked one by one. The fused form (hgemm_epilogue.cuh) is warp-uniform.
            if constexpr (FUSED)
                hgemm_epi::store_warp_tile_fused(acc, pre, C, geo, ep);
            else
                hgemm_epi::store_warp_tile<false, MT, NT>(acc, C, M, N, geo.row0, geo.col0, ep);
        }
        if (serial) {
            __syncwarp();
            if (lane == 0) mbar_arrive(done_bar);
        }
    }
}

// ---- host side --------------------------------------------------------------------------

// Schedule knobs (swept on the RTX 5090, see docs/design/hgemm.md, "Variant 6"). Against
// variant 4's: static ranges for any number of waves when A and B fit in L2 (they beat the
// queue by 3.5% at 4096^3 and by 3% at twelve waves), and the L2-footprint raster group
// (G = 16 instead of 8 at 8192^3 on 170 blocks, 243 against 241 TFLOPS; G = 32 at
// 4096x11008x4096, 243 against 239).
constexpr hgemm_sk::Knobs kKnobs = {hgemm_sk::kMinShare, hgemm_sk::kMinPass, hgemm_sk::kStaticWaves,
                                    /*static_waves_l2=*/0,
                                    /*group=*/-1};

// Sweep overrides, read once: SPARK_HGEMM_V6_CONFIG forces a pipeline configuration (index
// into CONFIGS below), SPARK_HGEMM_V6_GROUP the raster group, SPARK_HGEMM_V6_STATIC_WAVES
// and SPARK_HGEMM_V6_STATIC_WAVES_L2 the static/queue crossover, SPARK_HGEMM_V6_MIN_SHARE
// the least k-steps per block, SPARK_HGEMM_V6_MIN_PASS the shortest K-pass,
// SPARK_HGEMM_V6_SERIAL=1 the serialized-prologue experiment.
struct Env {
    int cfg = -1;
    int serial = 0;
    hgemm_sk::Knobs knobs = kKnobs;
};
const Env& env() {
    static Env e;
    static bool read = false;
    if (!read) {
        read = true;
        auto geti = [](const char* name, int fallback) {
            const char* s = std::getenv(name);
            return s ? std::atoi(s) : fallback;
        };
        e.cfg = geti("SPARK_HGEMM_V6_CONFIG", -1);
        e.serial = geti("SPARK_HGEMM_V6_SERIAL", 0);
        e.knobs.group = geti("SPARK_HGEMM_V6_GROUP", e.knobs.group);
        e.knobs.static_waves = geti("SPARK_HGEMM_V6_STATIC_WAVES", e.knobs.static_waves);
        e.knobs.static_waves_l2 = geti("SPARK_HGEMM_V6_STATIC_WAVES_L2", e.knobs.static_waves_l2);
        e.knobs.min_share = geti("SPARK_HGEMM_V6_MIN_SHARE", e.knobs.min_share);
        e.knobs.min_pass = geti("SPARK_HGEMM_V6_MIN_PASS", e.knobs.min_pass);
    }
    return e;
}

template <int BK, int STAGES, int MIN_BLOCKS, bool FUSED>
int resident_blocks() {
    constexpr int bytes = smem_bytes<BK, STAGES>();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(hgemm_v6_kernel<BK, STAGES, MIN_BLOCKS, FUSED>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, hgemm_v6_kernel<BK, STAGES, MIN_BLOCKS, FUSED>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int BK, int STAGES, int MIN_BLOCKS, bool FUSED>
void launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K,
            cudaStream_t stream, const HgemmEpilogue& ep) {
    constexpr int bytes = smem_bytes<BK, STAGES>();
    CUtensorMap tmA, tmB;
    make_maps<BK>(A, B, M, N, K, tmA, tmB);
    static hgemm_sk::Workspace w;  // one per pipeline configuration and epilogue instantiation
    Params p;
    hgemm_sk::plan(p, w, M, N, K, BM, BN, BK, resident_blocks<BK, STAGES, MIN_BLOCKS, FUSED>(),
                   env().knobs);
    hgemm_v6_kernel<BK, STAGES, MIN_BLOCKS, FUSED><<<p.grid, THREADS, bytes, stream>>>(
        tmA, tmB, C, M, N, K, p, env().serial, hgemm_epi::make(ep));
}

// Pipeline configurations (BK, stages, blocks per SM), the same six as variant 5. Index 0
// wins or ties on every shape (docs/design/hgemm.md, "Variant 6"): the two-block
// configurations spill inside the k-loop at the 96 registers the ninth warp leaves them.
// Each configuration has a plain and a fused instantiation; the plain one's residency is
// what the tile choice below is made from (the two agree: smem sets it).
struct Config {
    void (*launch)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int,
                   cudaStream_t, const HgemmEpilogue&);
    void (*launch_fused)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int,
                         cudaStream_t, const HgemmEpilogue&);
    int (*resident)();
    int bk;
};
constexpr Config CONFIGS[] = {
    // 0: 64 KB of smem, one block per SM
    {launch<64, 2, 1, false>, launch<64, 2, 1, true>, resident_blocks<64, 2, 1, false>, 64},
    // 1: 96 KB, one block per SM
    {launch<64, 3, 1, false>, launch<64, 3, 1, true>, resident_blocks<64, 3, 1, false>, 64},
    // 2: 48 KB, two blocks per SM
    {launch<32, 3, 2, false>, launch<32, 3, 2, true>, resident_blocks<32, 3, 2, false>, 32},
    // 3: 32 KB, two blocks per SM
    {launch<32, 2, 2, false>, launch<32, 2, 2, true>, resident_blocks<32, 2, 2, false>, 32},
    // 4: like 2 without the register cap
    {launch<32, 3, 1, false>, launch<32, 3, 1, true>, resident_blocks<32, 3, 1, false>, 32},
    // 5: 96 KB, one block per SM
    {launch<32, 6, 1, false>, launch<32, 6, 1, true>, resident_blocks<32, 6, 1, false>, 32},
};
constexpr int NUM_CONFIGS = static_cast<int>(sizeof(CONFIGS) / sizeof(CONFIGS[0]));
constexpr int CONFIG_DEFAULT = 0;

void launch_auto(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                 int K, cudaStream_t stream, const HgemmEpilogue& ep) {
    if (M <= 64) {  // decode: stream B (hgemm_decode.cu), or variant 4's 64-row tile
        if (hgemm_decode::supports(M, N, K))
            hgemm_decode::launch(A, B, C, M, N, K, stream, ep);
        else
            hgemm_streamk_bf16(A, B, C, M, N, K, stream, ep);
        return;
    }
    int cfg = env().cfg;
    if (cfg < 0 || cfg >= NUM_CONFIGS) cfg = CONFIG_DEFAULT;
    const Config& c = CONFIGS[cfg];
    // k-steps per block on the 128x128 tile; below kMinShare the fixup outweighs the mma
    // work and variant 4's smaller tiles take the shape.
    const long long share =
        static_cast<long long>(cdiv(M, BM)) * (N / BN) * (K / c.bk) / c.resident();
    if (N % BN != 0 || share < env().knobs.min_share) {
        hgemm_streamk_bf16(A, B, C, M, N, K, stream, ep);
        return;
    }
    if (hgemm_epi::is_plain(ep))
        c.launch(A, B, C, M, N, K, stream, ep);
    else
        c.launch_fused(A, B, C, M, N, K, stream, ep);
}

}  // namespace v6

}  // namespace

bool hgemm_tma_sk_supports(int M, int N, int K) {
    return M >= 1 && N % 64 == 0 && K % 64 == 0;  // variant 4's rules
}

void hgemm_tma_sk(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                  int K, cudaStream_t stream, const HgemmEpilogue& ep) {
    SPARK_REQUIRE(hgemm_tma_sk_supports(M, N, K),
                  "hgemm variant 6: requires N % 64 == 0 and K % 64 == 0");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(B),
                  "hgemm variant 6: A and B must be 16-byte aligned");
    v6::launch_auto(A, B, C, M, N, K, stream, ep);
}

}  // namespace spark
