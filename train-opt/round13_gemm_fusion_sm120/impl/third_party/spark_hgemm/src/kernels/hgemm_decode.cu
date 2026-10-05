// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../LICENSE).
// Decode-shape bf16 GEMM: C[M,N] = A[M,K] * B[K,N] for M <= 64, where the whole job is to
// stream the weight matrix B from DRAM once at the copy roof. Reached from hgemm variant 3's
// launch_auto (see hgemm_internal.cuh); the rules of variant 3 apply (row-major, fp32
// accumulation, mma.sync + ldmatrix, XOR-swizzled smem, cp.async pipeline). Measurements and
// the arithmetic behind the constants: docs/design/hgemm.md, "Decode: streaming the weights".
//
// What is different from the 64x64x64 tile of hgemm.cu on these shapes:
//   * the block tile is BM x BN with BM = 16, 32 or 64 picked from M, so a 16-token decode
//     step does one m16 row block instead of four (three of them zero-filled);
//   * one CTA per column strip of B, no split-K on the shapes that matter: 64 strips of 64
//     columns (or 128 of 32) already pull the full bandwidth, and every split-K launch pays a
//     flat 2 us for the atomics, fence, counter and last-arriver chain. K is split only when
//     a narrow N leaves fewer than 64 strips;
//   * when K is split, each slice adds its fp32 partial tile into a workspace with atomics
//     and the last slice to arrive converts the strip to bf16 and puts the workspace and the
//     arrival counter back to zero, so no memset launch precedes the kernel (the workspace is
//     zeroed once, when it is allocated, and every launch leaves it zero);
//   * the warp grid is WM x WN over the tile: at M = 64 the tensor work per column strip is
//     what limits a CTA, so the 64-row tile runs 8 warps on a 64x32 strip, 128 CTAs wide.
#include <algorithm>

#include "hgemm_epilogue.cuh"
#include "hgemm_internal.cuh"
#include "spark/common.cuh"

namespace spark::hgemm_decode {

namespace {

// Physical 16-byte chunk for logical (row, chunk) of a smem tile whose rows are CPR chunks
// long, so that the 8 rows an ldmatrix touches fall in 8 different bank groups. Rows of 2
// chunks (32 B): chunk ^= (row/4) % 2; 4 chunks (64 B): chunk ^= (row/2) % 4; 8 chunks or
// more (128 B+): chunk ^= row % 8.
template <int CPR>
__device__ __forceinline__ int swz(int row, int chunk) {
    if constexpr (CPR == 2)
        return chunk ^ ((row >> 2) & 1);
    else if constexpr (CPR == 4)
        return chunk ^ ((row >> 1) & 3);
    else
        return chunk ^ (row & 7);
}

struct Params {
    const __nv_bfloat16* A;
    const __nv_bfloat16* B;
    __nv_bfloat16* C;
    int M, N, K;
    int split;         // K-slices per column strip; block = slice * strips + strip
    float* ws;         // BM x N fp32 partial sums, zero between launches (split > 1 only)
    int* counters;     // one arrival counter per strip, zero between launches
    hgemm_epi::Ep ep;  // the fused epilogue (FUSED kernels only)
};

template <int BM, int BN, int BK, int STAGES>
constexpr int smem_bytes() {
    return STAGES * (BM * BK + BK * BN) * static_cast<int>(sizeof(__nv_bfloat16));
}

// Warps tile the block WM (rows) x WN (columns); each warp owns BM/WM rows by BN/WN columns.
// FUSED applies p.ep (hgemm_epilogue.cuh) wherever bf16 is stored: by the block itself when K
// is not split, by the last K-slice to arrive at a strip when it is.
template <int BM, int BN, int BK, int STAGES, int WM, int WN, bool FUSED>
__global__ void __launch_bounds__(WM * WN * 32) decode_kernel(Params p) {
    constexpr int THREADS = WM * WN * 32;
    constexpr int WROWS = BM / WM;  // rows per warp
    constexpr int WCOLS = BN / WN;  // columns per warp
    constexpr int MT = WROWS / 16;  // m16 tiles per warp
    constexpr int NT = WCOLS / 8;   // n8 tiles per warp
    static_assert(MT >= 1 && MT * 16 * WM == BM, "");
    static_assert(NT >= 2 && NT % 2 == 0, "ldmatrix.x4.trans fills two n8 tiles at a time");
    constexpr int A_CPR = BK / 8;         // 16-byte chunks per smem row of A
    constexpr int B_CPR = BN / 8;         // ... of B
    constexpr int A_CHUNKS = BM * A_CPR;  // chunks per stage
    constexpr int B_CHUNKS = BK * B_CPR;
    constexpr int A_ITERS = cdiv(A_CHUNKS, THREADS);
    constexpr int B_ITERS = cdiv(B_CHUNKS, THREADS);
    constexpr int A_STAGE = BM * BK;  // elements
    constexpr int B_STAGE = BK * BN;
    static_assert(A_CPR == 4 || A_CPR % 8 == 0, "rows of 64 B or 128 B+");
    static_assert(B_CPR == 2 || B_CPR == 4 || B_CPR % 8 == 0, "rows of 32 B, 64 B or 128 B+");

    extern __shared__ __align__(128) unsigned char smem_raw[];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Bs = As + STAGES * A_STAGE;
    __shared__ int s_last;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / WN;
    const int wn = warp % WN;
    const int M = p.M, N = p.N, K = p.K;
    const int strips = N / BN;
    const int KT = K / BK;

    // Blocks of one slice are consecutive, so the CTAs of a wave read the same rows of B
    // across all strips: whole contiguous rows of B, not a column of 64 B pieces.
    const int slice = blockIdx.x / strips;
    const int strip = blockIdx.x % strips;
    const int kt_begin = static_cast<int>(static_cast<long long>(slice) * KT / p.split);
    const int kt_end = static_cast<int>(static_cast<long long>(slice + 1) * KT / p.split);
    const int nkt = kt_end - kt_begin;

    const __nv_bfloat16* Ab = p.A + static_cast<size_t>(kt_begin) * BK;
    const __nv_bfloat16* Bb = p.B + static_cast<size_t>(kt_begin) * BK * N + strip * BN;

    auto load_stage = [&](int stage, int k0) {
        __nv_bfloat16* as = As + stage * A_STAGE;
        __nv_bfloat16* bs = Bs + stage * B_STAGE;
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int c = tid + i * THREADS;
            if (A_CHUNKS % THREADS == 0 || c < A_CHUNKS) {
                const int row = c / A_CPR;
                const int ch = c % A_CPR;
                const bool ok = row < M;  // rows past M read nothing and land as zeros
                cp_async_16_zfill(as + row * BK + swz<A_CPR>(row, ch) * 8,
                                  Ab + static_cast<size_t>(ok ? row : 0) * K + k0 + ch * 8, ok);
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
            const int c = tid + i * THREADS;
            if (B_CHUNKS % THREADS == 0 || c < B_CHUNKS) {
                const int row = c / B_CPR;
                const int ch = c % B_CPR;
                cp_async_16(bs + row * BN + swz<B_CPR>(row, ch) * 8,
                            Bb + static_cast<size_t>(k0 + row) * N + ch * 8);
            }
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    // Prologue: STAGES-1 tiles in flight before any compute.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nkt) load_stage(s, s * BK);
        cp_async_commit();
    }

    // Per-lane ldmatrix selectors, constant across the loop.
    const int a_row = lane & 15;     // row within an m16 tile
    const int a_kchunk = lane >> 4;  // k 0-7 or 8-15 of the k16 step
    const int b_krow = lane & 15;    // k row within the k16 step
    const int b_nchunk = lane >> 4;  // which n8 tile of the pair

    for (int kt = 0; kt < nkt; ++kt) {
        cp_async_wait<STAGES - 2>();  // tile kt has landed for this thread
        __syncthreads();              // ... for every thread; stage (kt-1) % STAGES is free
        {
            const int nk = kt + STAGES - 1;
            if (nk < nkt) load_stage(nk % STAGES, nk * BK);
            cp_async_commit();  // always, so the group count stays uniform
        }
        const __nv_bfloat16* as = As + (kt % STAGES) * A_STAGE;
        const __nv_bfloat16* bs = Bs + (kt % STAGES) * B_STAGE;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned afrag[MT][4];
            unsigned bfrag[NT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WROWS + mi * 16 + a_row;
                const int ch = kk / 8 + a_kchunk;
                ldmatrix_x4(afrag[mi], as + row * BK + swz<A_CPR>(row, ch) * 8);
            }
#pragma unroll
            for (int nj = 0; nj < NT; nj += 2) {
                const int krow = kk + b_krow;
                const int ch = (wn * WCOLS + nj * 8) / 8 + b_nchunk;
                unsigned r[4];
                ldmatrix_x4_trans(r, bs + krow * BN + swz<B_CPR>(krow, ch) * 8);
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

    // Epilogue. Each lane owns (row g, cols 2c..2c+1) and (row g+8, same) of every 16x8
    // accumulator tile; rows past M are skipped one by one, so any M works.
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    const int row0 = wm * WROWS + g;
    const int col0 = strip * BN + wn * WCOLS + c2;
    if (p.split == 1) {
        hgemm_epi::store_warp_tile<FUSED, MT, NT>(acc, p.C, M, N, row0, col0, p.ep);
        return;
    }

    // Split K: add this slice into the fp32 workspace (the adds happen at L2, in any order).
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) {
            const int row = row0 + mi * 16;
            const int col = col0 + nj * 8;
            if (row < M) {
                atomicAdd(p.ws + static_cast<size_t>(row) * N + col, acc[mi][nj][0]);
                atomicAdd(p.ws + static_cast<size_t>(row) * N + col + 1, acc[mi][nj][1]);
            }
            if (row + 8 < M) {
                atomicAdd(p.ws + static_cast<size_t>(row + 8) * N + col, acc[mi][nj][2]);
                atomicAdd(p.ws + static_cast<size_t>(row + 8) * N + col + 1, acc[mi][nj][3]);
            }
        }
    // The fence orders every thread's adds before the arrival; the last slice to arrive
    // converts the strip to bf16 and restores the zero state for the next launch.
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(p.counters + strip, 1) == p.split - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int rows = M < BM ? M : BM;
    auto finish = [&](auto store) {
        for (int i = tid; i < rows * (BN / 4); i += THREADS) {
            const int r = i / (BN / 4);
            const int c = strip * BN + (i % (BN / 4)) * 4;
            float4* w = reinterpret_cast<float4*>(p.ws + static_cast<size_t>(r) * N + c);
            store(__ldcg(w), r, c);  // from L2, where the atomics landed
            __stcg(w, make_float4(0.f, 0.f, 0.f, 0.f));
        }
    };
    if constexpr (FUSED) {
        hgemm_epi::dispatch(p.ep, [&](auto mode) {
            finish([&](float4 v, int r, int c) {
                hgemm_epi::store_vec4_mode<decltype(mode)>(v, p.C, N, r, c, p.ep);
            });
        });
    } else {
        finish([&](float4 v, int r, int c) { hgemm_epi::store_vec4_plain(v, p.C, N, r, c); });
    }
    if (tid == 0) atomicExch(p.counters + strip, 0);
}

// ---- host side ---------------------------------------------------------------------------

// Workspace: BM x N fp32 partials (BM <= 64) plus one counter per strip, zeroed when it is
// allocated and left zero by every launch. One per process, grown on demand.
struct Workspace {
    float* ws = nullptr;
    int* counters = nullptr;
    size_t floats = 0;
    size_t strips = 0;
};

Workspace& workspace(size_t floats, size_t strips) {
    static Workspace w;
    if (w.floats < floats) {
        if (w.ws) SPARK_CUDA_CHECK(cudaFree(w.ws));
        w.floats = std::max(floats, w.floats * 2);
        SPARK_CUDA_CHECK(cudaMalloc(&w.ws, w.floats * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMemset(w.ws, 0, w.floats * sizeof(float)));
    }
    if (w.strips < strips) {
        if (w.counters) SPARK_CUDA_CHECK(cudaFree(w.counters));
        w.strips = std::max(strips, w.strips * 2);
        SPARK_CUDA_CHECK(cudaMalloc(&w.counters, w.strips * sizeof(int)));
        SPARK_CUDA_CHECK(cudaMemset(w.counters, 0, w.strips * sizeof(int)));
    }
    return w;
}

// Resident CTAs per GPU for one configuration; also the > 48 KB smem opt-in.
template <int BM, int BN, int BK, int STAGES, int WM, int WN, bool FUSED>
int slots() {
    constexpr int bytes = smem_bytes<BM, BN, BK, STAGES>();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(decode_kernel<BM, BN, BK, STAGES, WM, WN, FUSED>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, decode_kernel<BM, BN, BK, STAGES, WM, WN, FUSED>, WM * WN * 32, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// Measured on the RTX 5090 (docs/design/hgemm.md, "Decode"): 64 CTAs with 3 x 8 KB of B in
// flight each already stream B at the copy roof, and every split-K launch pays a flat 2 us
// for the atomics, fence, counter and last-arriver chain. So K is split only when the column
// strips alone would leave fewer than 64 CTAs.
constexpr int kMinCtas = 64;

template <int BM, int BN, int BK, int STAGES, int WM, int WN, bool FUSED>
void launch_cfg(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                int K, cudaStream_t stream, const HgemmEpilogue& ep) {
    constexpr int bytes = smem_bytes<BM, BN, BK, STAGES>();
    (void)slots<BM, BN, BK, STAGES, WM, WN, FUSED>();  // the smem opt-in, if not done by fits()
    Params p;
    p.A = A;
    p.B = B;
    p.C = C;
    p.M = M;
    p.N = N;
    p.K = K;
    p.ep = hgemm_epi::make(ep);
    const int strips = N / BN;
    p.split = std::min(K / BK, std::max(1, cdiv(kMinCtas, strips)));
    p.ws = nullptr;
    p.counters = nullptr;
    if (p.split > 1) {
        Workspace& w = workspace(static_cast<size_t>(BM) * N, static_cast<size_t>(strips));
        p.ws = w.ws;
        p.counters = w.counters;
    }
    decode_kernel<BM, BN, BK, STAGES, WM, WN, FUSED>
        <<<strips * p.split, WM * WN * 32, bytes, stream>>>(p);
}

// Two configurations per row tile, in order of preference: the first whose strips fit in one
// wave of resident CTAs runs (172 strips on 170 slots would be a full wave plus two stragglers
// that take as long again), otherwise the second, which has more slots.
template <int BM, int BN, int BK, int STAGES, int WM, int WN, bool FUSED>
bool fits(int N, int K) {
    return N % BN == 0 && K % BK == 0 && N / BN <= slots<BM, BN, BK, STAGES, WM, WN, FUSED>();
}

template <bool FUSED>
void launch_impl(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                 int K, cudaStream_t stream, const HgemmEpilogue& ep) {
    if (M <= 16) {
        // 16x64x64, 4 stages, 4 warps along N, 40 KB: 2 CTAs per SM, 340 slots.
        if (fits<16, 64, 64, 4, 1, 4, FUSED>(N, K))
            launch_cfg<16, 64, 64, 4, 1, 4, FUSED>(A, B, C, M, N, K, stream, ep);
        else  // 16x32x64, 24 KB: 4 per SM, 680 slots
            launch_cfg<16, 32, 64, 4, 1, 2, FUSED>(A, B, C, M, N, K, stream, ep);
    } else if (M <= 32) {
        // 32x32x64, 4 stages, 2x2 warps, 32 KB: 3 per SM, 510 slots.
        if (fits<32, 32, 64, 4, 2, 2, FUSED>(N, K))
            launch_cfg<32, 32, 64, 4, 2, 2, FUSED>(A, B, C, M, N, K, stream, ep);
        else  // 32x64x64, 2x4 warps, 40 KB: 2 per SM, 340 slots, half the strips
            launch_cfg<32, 64, 64, 4, 2, 4, FUSED>(A, B, C, M, N, K, stream, ep);
    } else {
        // 64x32x128, 4 stages, 4x2 warps, 96 KB: 1 per SM, 170 slots. The tensor work of a
        // 64-row strip is what limits the CTA, so the tile is narrow and the warps many.
        if (fits<64, 32, 128, 4, 4, 2, FUSED>(N, K))
            launch_cfg<64, 32, 128, 4, 4, 2, FUSED>(A, B, C, M, N, K, stream, ep);
        else  // 64x64x64, 3 stages, 2x4 warps, 48 KB: 2 per SM, 340 slots
            launch_cfg<64, 64, 64, 3, 2, 4, FUSED>(A, B, C, M, N, K, stream, ep);
    }
}

}  // namespace

bool supports(int M, int N, int K) {
    return M >= 1 && M <= 64 && N > 0 && K > 0 && N % 64 == 0 && K % 64 == 0;
}

void launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K,
            cudaStream_t stream, const HgemmEpilogue& ep) {
    if (hgemm_epi::is_plain(ep))
        launch_impl<false>(A, B, C, M, N, K, stream, ep);
    else
        launch_impl<true>(A, B, C, M, N, K, stream, ep);
}

}  // namespace spark::hgemm_decode
