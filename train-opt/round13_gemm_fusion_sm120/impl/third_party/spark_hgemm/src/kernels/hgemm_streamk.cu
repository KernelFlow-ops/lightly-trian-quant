// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// hgemm variant 4: the variant 3 tile (hgemm_tile.cuh) on a persistent Stream-K schedule.
//
//   C[M,N] = A[M,K] * B[K,N]      row-major, bf16 in/out, fp32 accumulate
//
// What changes against variant 3 is only who does which work. A persistent grid of at most
// `resident` blocks (2 per SM for the 128x128 tile, 340 on the RTX 5090) walks the tiles in
// a grouped raster order (G tile rows column-major, so the tiles in flight share A rows and
// B columns and fit in L2). Two schedules, picked per call:
//
//   * static ranges (up to two waves, or four when both operands fit in L2): the flattened
//     (tile, k-step) space is cut into equal contiguous ranges, one per block. Every block
//     gets the same number of k-steps, so there is no quantization at all; a tile is
//     finished by two or three blocks.
//   * queue (more waves): whole tiles are handed out from a queue (one atomicAdd per tile,
//     so a block on a slow SM simply takes fewer), and the last wave plus the tail is handed
//     out from the same queue in K-passes of geometrically shrinking length (half the tile,
//     a quarter, ... the last one 9 to 16 k-steps), so the kernel ends within a few k-steps
//     of work on every SM instead of within one tile.
//
// Both use the same fixup. Each Stream-K tile has one fp32 slot and one flag that says how
// many k-steps have been accumulated into the slot. A piece that does not start the tile
// waits (acquire) until the flag reads its kt_begin, then adds the slot into its
// accumulators; a piece that does not end the tile writes the slot and publishes its kt_end
// (release); the piece that ends the tile stores bf16. Partials are always summed in K
// order, so the result is the same bits every run, whichever block computed which piece.
// The flag carries a per-launch epoch in its high word, so a stale value can never match
// and nothing is cleared between launches; the queue is reset by the last block out.
//
// A piece only ever waits for the piece before it in K, which was handed out earlier: in
// queue mode by a block that is running because it took it from the queue, in static mode
// by the block with the next lower index, which was dispatched no later than this one and
// is resident because the grid never exceeds the resident block count. Chains end at the
// piece that starts the tile, which never waits, so there is no cycle.
//
// The schedule (parameters, raster, passes, ranges, slot layout, the host plan) lives in
// hgemm_streamk.cuh, shared with variant 6, which runs it with the TMA tile.
//
// Requires N % 64 == 0, K % 64 == 0, any M >= 1 (the same rules as variant 3: rows past M
// are zero-filled by the tile and skipped by the epilogue).

#include "hgemm_internal.cuh"
#include "hgemm_streamk.cuh"
#include "hgemm_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace sk {

using hgemm_sk::ld_acquire_gpu;
using hgemm_sk::Params;
using hgemm_sk::pass_range;
using hgemm_sk::range_start;
using hgemm_sk::raster;
using hgemm_sk::slot_idx;
using hgemm_sk::st_release_gpu;
using hgemm_tile::THREADS;

// FUSED: the storing piece applies the epilogue options in `ep` (hgemm_epilogue.cuh); the
// plain instantiation carries none of that code.
template <int BM, int BN, int BK, int STAGES, bool FUSED>
__global__ void __launch_bounds__(THREADS)
    hgemm_v4_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K, Params p,
                    hgemm_epi::Ep ep) {
    using Cfg = hgemm_tile::Cfg<BM, BN, BK, STAGES>;
    constexpr int WM = Cfg::WM, MT = Cfg::MT, NT = Cfg::NT;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ int s_item;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / hgemm_tile::WARPS_N;
    const int cta = blockIdx.x;
    typename Cfg::Acc acc;

    // One piece of Stream-K tile `tile_r` (index within the region): k-steps [kt_begin,
    // kt_end). Leaves the pipeline's smem free for the next call.
    auto piece = [&](int tile_r, int kt_begin, int kt_end) {
        int tm, tn;
        raster(p, p.dp_tiles + tile_r, tm, tn);
        const int bm = tm * BM, bn = tn * BN;
        const int m_valid = M - bm;
        hgemm_tile::mainloop<Cfg>(A, B, N, K, bm, bn, m_valid, kt_begin, kt_end - kt_begin,
                                  smem_raw, acc);
        if (kt_begin > 0 || kt_end < p.KT) {
            float4* slot =
                reinterpret_cast<float4*>(p.slots + static_cast<size_t>(tile_r) * (BM * BN));
            unsigned long long* flag = p.flags + tile_r;
            const unsigned long long tag = static_cast<unsigned long long>(p.epoch) << 32;
            if (kt_begin > 0) {
                // Everything before kt_begin is in the slot once the flag says so.
                if (tid == 0) {
                    while (ld_acquire_gpu(flag) != (tag | static_cast<unsigned>(kt_begin))) {
                    }
                }
                __syncthreads();
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
            if (kt_end < p.KT) {
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
                __syncthreads();
                if (tid == 0) st_release_gpu(flag, tag | static_cast<unsigned>(kt_end));
                return;
            }
        }
        hgemm_tile::store_bf16<Cfg, FUSED>(acc, C, M, N, bm, bn, ep);
        __syncthreads();  // the pipeline's last stage is free before the next prologue
    };

    if (p.passes == 0) {
        // Static ranges, walked from the top: the piece that starts a tile (and never waits)
        // comes first, the piece that ends one (and may wait for the block below) last.
        const int it_begin = range_start(p, cta);
        int e = range_start(p, cta + 1);
        while (e > it_begin) {
            const int tile_r = (e - 1) / p.KT;
            const int tile_k0 = tile_r * p.KT;
            const int s = max(it_begin, tile_k0);
            piece(tile_r, s - tile_k0, e - tile_k0);
            e = s;
        }
        return;
    }

    // Queue: whole tiles first, then the Stream-K region pass by pass. Passes are handed
    // out tile-interleaved (every tile's first pass, then every tile's second, ...), so a
    // pass's predecessor in K was taken a whole region earlier and has almost always been
    // published by the time it is needed.
    const int sk_tiles = p.tiles - p.dp_tiles;
    const int items = p.dp_tiles + sk_tiles * p.passes;
    for (;;) {
        if (tid == 0) s_item = static_cast<int>(atomicAdd(p.queue, 1u));
        __syncthreads();
        const int item = s_item;
        if (item >= items) break;
        if (item < p.dp_tiles) {
            int tm, tn;
            raster(p, item, tm, tn);
            const int bm = tm * BM, bn = tn * BN;
            hgemm_tile::mainloop<Cfg>(A, B, N, K, bm, bn, M - bm, 0, p.KT, smem_raw, acc);
            hgemm_tile::store_bf16<Cfg, FUSED>(acc, C, M, N, bm, bn, ep);
            __syncthreads();
        } else {
            const int q = item - p.dp_tiles;
            int kb, ke;
            pass_range(p, q / sk_tiles, kb, ke);
            piece(q % sk_tiles, kb, ke);
        }
    }
    // The last block out resets the queue for the next launch. Every block has made its
    // final, failing grab before it counts itself done, so nothing touches the queue after
    // the reset until the next launch.
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(p.queue + 1, 1u) == static_cast<unsigned>(p.grid) - 1) {
            p.queue[0] = 0u;
            p.queue[1] = 0u;
            __threadfence();
        }
    }
}

template <int BM, int BN, int BK, int STAGES, bool FUSED>
int resident_blocks() {
    constexpr int bytes = hgemm_tile::smem_bytes<BM, BN, BK, STAGES>();
    static int resident = 0;  // blocks resident per GPU; also the > 48 KB smem opt-in
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(hgemm_v4_kernel<BM, BN, BK, STAGES, FUSED>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, hgemm_v4_kernel<BM, BN, BK, STAGES, FUSED>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int BM, int BN, int BK, int STAGES, bool FUSED>
void launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K,
            cudaStream_t stream, const HgemmEpilogue& ep) {
    constexpr int bytes = hgemm_tile::smem_bytes<BM, BN, BK, STAGES>();
    static hgemm_sk::Workspace w;  // one per tile configuration and epilogue instantiation
    Params p;
    hgemm_sk::plan(p, w, M, N, K, BM, BN, BK, resident_blocks<BM, BN, BK, STAGES, FUSED>());
    hgemm_v4_kernel<BM, BN, BK, STAGES, FUSED>
        <<<p.grid, THREADS, bytes, stream>>>(A, B, C, M, N, K, p, hgemm_epi::make(ep));
}

constexpr int BK = 32;
constexpr int STAGES = 3;

// Tile selection. The 128x128 tile has the best smem-side intensity and wins whenever each
// block's share of the (tile, k-step) space is long enough to amortize the pipeline and
// the fixup; below that the 64-row tiles. Decode shapes (M <= 64, bound by streaming B) go
// to the same weight-streaming kernel as variant 3 (hgemm_decode.cu): it streams B at the
// single-launch floor, which the 64x64x64 tile on this schedule does not quite reach
// (1,352 against 1,459 GB/s at 64x4096x4096).
template <bool FUSED>
void launch_auto(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                 int K, cudaStream_t stream, const HgemmEpilogue& ep) {
    auto share = [&](int bm, int bn, int bk, int resident) {  // k-steps per block
        return static_cast<long long>(cdiv(M, bm)) * (N / bn) * (K / bk) / resident;
    };
    if (M <= 64 && hgemm_decode::supports(M, N, K)) {
        hgemm_decode::launch(A, B, C, M, N, K, stream, ep);
    } else if (M <= 64) {
        launch<64, 64, 64, 4, FUSED>(A, B, C, M, N, K, stream, ep);
    } else if (N % 128 == 0 &&
               share(128, 128, BK, resident_blocks<128, 128, BK, STAGES, FUSED>()) >=
                   hgemm_sk::kMinShare) {
        launch<128, 128, BK, STAGES, FUSED>(A, B, C, M, N, K, stream, ep);
    } else if (N % 128 == 0 && share(64, 128, BK, resident_blocks<64, 128, BK, STAGES, FUSED>()) >=
                                   hgemm_sk::kMinShare) {
        launch<64, 128, BK, STAGES, FUSED>(A, B, C, M, N, K, stream, ep);
    } else {
        launch<64, 64, 64, 3, FUSED>(A, B, C, M, N, K, stream, ep);
    }
}

}  // namespace sk

}  // namespace

void hgemm_streamk_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M,
                        int N, int K, cudaStream_t stream, const HgemmEpilogue& ep) {
    if (hgemm_epi::is_plain(ep))
        sk::launch_auto<false>(A, B, C, M, N, K, stream, ep);
    else
        sk::launch_auto<true>(A, B, C, M, N, K, stream, ep);
}

}  // namespace spark
