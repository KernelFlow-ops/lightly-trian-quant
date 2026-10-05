// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// The Stream-K schedule shared by hgemm variant 4 (hgemm_streamk.cu, the cp.async tile) and
// variant 6 (hgemm_tma_sk.cu, the TMA tile). Private to src/kernels. The two kernels differ
// in how a stage reaches shared memory; who computes which (tile, k-range) piece, in what
// order, and how the pieces of a tile are summed is the same and lives here once:
//
//   * the per-launch parameters (Params) and the host plan() that fills them: static ranges
//     up to a few waves, a tile queue with geometrically shrinking K-passes above that;
//   * the grouped raster (G tile rows walked column-major, so the tiles in flight share A
//     rows and B columns in L2);
//   * the fixup: one fp32 slot and one epoch-tagged flag per Stream-K tile, pieces summed in
//     K order through the slot, so the output is the same bits every run.
//
// The reasoning behind each constant is in docs/design/hgemm.md, "Variant 4".
#pragma once

#include <algorithm>
#include <cstddef>

#include "spark/common.cuh"

namespace spark::hgemm_sk {

struct Params {
    int tiles_m, tiles_n, tiles;
    int KT;        // k-steps per tile
    int group;     // tile rows per raster group (1: plain row-major over N)
    int dp_tiles;  // queue mode: tiles [0, dp_tiles) in raster order are whole; the rest are
                   // the Stream-K region. Static mode: 0.
    int passes;    // queue mode: K-passes per Stream-K tile; 0 = static ranges
    int sk_iters;  // static mode: tiles * KT, cut into `grid` equal ranges
    int grid;      // blocks in the launch, all resident at once
    float* slots;  // (tiles - dp_tiles) x BM x BN fp32, one per Stream-K tile
    unsigned long long* flags;  // per Stream-K tile: (epoch << 32) | k-steps accumulated
    unsigned* queue;            // [0]: next work item, [1]: blocks done
    unsigned epoch;
};

__device__ __forceinline__ void st_release_gpu(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_acquire_gpu(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// Grouped rasterization: tile indices walk G tile rows column-major before moving to the
// next G rows. Consecutive indices then share B columns (same tn) across G tiles and A
// rows across the tiles of one group, instead of sharing A only along a whole tile row.
__device__ __forceinline__ void raster(const Params& p, int tile, int& tm, int& tn) {
    const int per_group = p.group * p.tiles_n;
    const int g = tile / per_group;
    const int first = g * p.group;
    const int gsz = min(p.group, p.tiles_m - first);
    const int r = tile - g * per_group;
    tm = first + r % gsz;
    tn = r / gsz;
}

// Queue mode, pass j of a Stream-K tile: k-steps [KT - KT/2^j, KT - KT/2^(j+1)), the last
// pass running to KT. Half the tile, then a quarter, ... so the last passes are short (the
// kernel ends within one of them on every SM) while a tile has only `passes` chain links,
// 4 at K = 4096, 5 at 8192, 6 at 11008.
__device__ __forceinline__ void pass_range(const Params& p, int j, int& kb, int& ke) {
    kb = j == 0 ? 0 : p.KT - (p.KT >> j);
    ke = j + 1 == p.passes ? p.KT : p.KT - (p.KT >> (j + 1));
}

// Static mode: first iteration owned by block c; ranges are [start(c), start(c+1)).
__device__ __forceinline__ int range_start(const Params& p, int c) {
    return static_cast<int>(static_cast<long long>(c) * p.sk_iters / p.grid);
}

// A partial tile in its slot, in fragment order: lane-contiguous float4s, so every warp
// writes and reads 512-byte runs and a lane finds its own accumulator elements at the same
// place whichever block wrote them. MT x NT is the warp's grid of m16n8 accumulators.
template <int MT, int NT>
__device__ __forceinline__ int slot_idx(int warp, int lane, int mi, int nj) {
    return ((warp * MT + mi) * NT + nj) * 32 + lane;
}

// Schedule constants (swept on the RTX 5090, see docs/design/hgemm.md).
constexpr int kMinShare = 32;      // static mode: least k-steps per block; tile choice threshold
constexpr int kMinPass = 8;        // queue mode: no pass shorter than this; the last is 9 to 16
constexpr int kStaticWaves = 2;    // static ranges up to this many waves ...
constexpr int kStaticWavesL2 = 4;  // ... or this many when A and B fit in L2 together
constexpr double kL2Share = 0.75;  // "fit": A + B <= this share of the L2

// The knobs plan() takes, defaulted to variant 4's constants above; variant 6 (one block
// per SM, half the blocks in flight, BK = 64) re-swept them and passes its own.
struct Knobs {
    int min_share = kMinShare;
    int min_pass = kMinPass;
    int static_waves = kStaticWaves;
    int static_waves_l2 = kStaticWavesL2;  // <= 0: static ranges whenever A and B fit in L2
    int group = 0;  // > 0: fixed; 0: about the square root of the blocks in flight (variant
                    // 4); -1: the largest group whose A rows fit in L2 next to the B columns
                    // in flight (variant 6, see raster_group below)
};

inline size_t l2_bytes() {
    static size_t bytes = 0;
    if (bytes == 0) {
        int dev = 0, l2 = 0;
        SPARK_CUDA_CHECK(cudaGetDevice(&dev));
        SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
        bytes = l2 > 0 ? static_cast<size_t>(l2) : size_t{96} << 20;
    }
    return bytes;
}

// Raster group for a grid of `grid` blocks in flight over tiles_m x tiles_n tiles of BM x BN
// with K deep operands. In queue mode consecutive raster indices are in flight, so a group's
// G tile rows of A (G x BM x K bf16) are re-read once per tile column and stay in L2 while
// the group runs, and B is read from DRAM once per group, i.e. tiles_m / G times in all. The
// fewer groups the less DRAM traffic, until a group's A no longer fits next to the grid / G
// tile columns of B in flight (at 8192^3 on 170 blocks: G = 16 is 32 + 22 MB, G = 32 is 76
// MB and measures the same, G = 64 does not fit and reads A once per column instead). So:
// the largest power of two whose footprint fits in kL2Share of the L2, else the one with the
// smallest footprint.
inline int raster_group(int grid, int tiles_m, int BM, int BN, int K) {
    const double a_row = static_cast<double>(BM) * K * sizeof(__nv_bfloat16);
    const double b_col = static_cast<double>(BN) * K * sizeof(__nv_bfloat16);
    const double budget = kL2Share * static_cast<double>(l2_bytes());
    int best_fit = 0, best_any = 1;
    double smallest = -1.0;
    for (int g = 1; g <= tiles_m; g *= 2) {
        const double footprint = g * a_row + cdiv(grid, g) * b_col;
        if (footprint <= budget) best_fit = g;
        if (smallest < 0.0 || footprint < smallest) {
            smallest = footprint;
            best_any = g;
        }
    }
    return best_fit > 0 ? best_fit : best_any;
}

// Per (kernel, tile configuration) device memory: grown on demand, never cleared by the
// host. Not safe to share between streams that run the same kernel concurrently.
struct Workspace {
    float* slots = nullptr;
    unsigned long long* flags = nullptr;
    unsigned* queue = nullptr;
    size_t tiles = 0;  // Stream-K tiles the slots and flags can hold
    unsigned epoch = 0;
};

// Fills `p` for a BM x BN tile with K / BK k-steps on a grid of at most `resident` blocks,
// grows `w` to hold the plan's Stream-K tiles and bumps its epoch.
inline void plan(Params& p, Workspace& w, int M, int N, int K, int BM, int BN, int BK, int resident,
                 const Knobs& k = Knobs()) {
    p.tiles_m = cdiv(M, BM);
    p.tiles_n = N / BN;
    p.tiles = p.tiles_m * p.tiles_n;
    p.KT = K / BK;
    const long long total = static_cast<long long>(p.tiles) * p.KT;
    const double operands = 2.0 * (static_cast<double>(M) * K + static_cast<double>(K) * N);
    const bool in_l2 = operands <= kL2Share * static_cast<double>(l2_bytes());
    const long long static_waves = in_l2 ? k.static_waves_l2 : k.static_waves;
    const bool use_static = (in_l2 && k.static_waves_l2 <= 0) ||
                            p.tiles <= static_cast<long long>(resident) * static_waves;

    int sk_tiles;
    if (use_static) {
        // Static ranges. Never more blocks than there are pieces of min_share k-steps (a
        // shorter piece costs more in fixup traffic than it does in mma work), but never
        // fewer than there are tiles: with one block per tile there are no partials at all.
        p.grid = static_cast<int>(
            std::min<long long>(resident, std::max<long long>(p.tiles, total / k.min_share)));
        p.dp_tiles = 0;
        p.passes = 0;
        p.sk_iters = p.tiles * p.KT;
        sk_tiles = p.tiles;
    } else {
        // Queue: whole tiles, then the tail and one full wave in geometric passes.
        p.grid = resident;
        const int tail = p.tiles % p.grid;
        sk_tiles = tail + (tail > 0 ? p.grid : 0);
        p.dp_tiles = p.tiles - sk_tiles;
        p.passes = 1;
        while ((p.KT >> p.passes) > k.min_pass) ++p.passes;
        p.sk_iters = 0;
    }

    // Raster group: G tile rows per group. Variant 4's rule is about the square root of the
    // blocks in flight, so a wave touches G rows of A and grid/G columns of B; variant 6's
    // is the L2-footprint rule above.
    int g = k.group;
    if (g == 0) {
        g = 1;
        while ((g * 2) * (g * 2) <= p.grid) g *= 2;
    } else if (g < 0) {
        g = raster_group(p.grid, p.tiles_m, BM, BN, K);
    }
    p.group = std::max(1, std::min(g, p.tiles_m));

    if (w.tiles < static_cast<size_t>(sk_tiles)) {
        if (w.slots) SPARK_CUDA_CHECK(cudaFree(w.slots));
        if (w.flags) SPARK_CUDA_CHECK(cudaFree(w.flags));
        w.tiles = std::max<size_t>(sk_tiles, 2 * static_cast<size_t>(resident));
        SPARK_CUDA_CHECK(cudaMalloc(&w.slots, w.tiles * BM * BN * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&w.flags, w.tiles * sizeof(unsigned long long)));
        SPARK_CUDA_CHECK(cudaMemset(w.flags, 0, w.tiles * sizeof(unsigned long long)));  // once
    }
    if (!w.queue) {
        SPARK_CUDA_CHECK(cudaMalloc(&w.queue, 2 * sizeof(unsigned)));
        SPARK_CUDA_CHECK(cudaMemset(w.queue, 0, 2 * sizeof(unsigned)));  // once
    }
    if (++w.epoch == 0) w.epoch = 1;
    p.slots = w.slots;
    p.flags = w.flags;
    p.queue = w.queue;
    p.epoch = w.epoch;
}

}  // namespace spark::hgemm_sk
