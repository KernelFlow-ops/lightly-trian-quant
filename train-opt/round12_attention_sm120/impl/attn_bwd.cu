// Round12 backward attention for sm_120: packed dqkv output, fused inverse RoPE, query bitmask, tail-aware tiles.
//
// FlashAttention-2 style, adapted from spark-kernels attention_bwd.cu variants 2/3 (MIT, commit a401bfa9e363):
//   * bwd_pre   : Dv_i = dO_i . O_i, lse2 -> (lse2, Dv) float2 rows padded to a multiple of 64, and the fp32 dQ
//                 accumulator rows are zeroed (the zeroed lines are L2-resident when the atomics arrive).
//   * bwd_kv    : one CTA owns 128 keys of one (b, h) (8 warps x 16 keys), K resident in shared memory (rotated in
//                 place by the rotator warp), V in registers; it streams 64-row Q / dO tiles through a 2-stage TMA
//                 ring (Q rotated in place by the rotator warp), keeps dK / dV in registers and adds its dQ share
//                 (dS K) to the fp32 accumulator with vector atomics.  dK gets the inverse RoPE in registers and dK /
//                 dV are written straight into the packed dqkv [B][N][3][H][64].
//   * bwd_dq    : dQ = scale * dq_acc, inverse RoPE, bf16, written into the q slot of dqkv.
// The query loop covers N exactly (a last tile of 5 / 41 rows runs only the n8 / k16 steps it needs), and warps
// whose 16 keys lie past N retire (the tail CTA of N = 1029 has one active warp).
#include <type_traits>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "attn_common.cuh"
#include "attn_api.h"

namespace a12 {

namespace {

constexpr int KB = 128, QB = 64, CW = 8, STAGES = 3;
constexpr int THREADS = CW * 32;  // 8 warps only: 2 warps per scheduler -> up to 255 registers per thread
constexpr int K_BYTES = KB * 128;
constexpr int STG_TILE = QB * 128;
constexpr int STG_BYTES = 2 * STG_TILE;  // Q tile then dO tile
constexpr int DS_BYTES = KB * QB * 2;
constexpr int LD_BYTES = QB * 8;
constexpr int OFF_STAGE = K_BYTES;                          // 1 KB aligned (128-byte TMA swizzle)
constexpr int OFF_DS = OFF_STAGE + STAGES * STG_BYTES;       // 1 KB aligned: buffer 1 also receives the V tile by TMA
constexpr int OFF_BAR = OFF_DS + 2 * DS_BYTES;
constexpr int OFF_LD = OFF_BAR + 128;
constexpr int SMEM_BYTES = OFF_LD + STAGES * LD_BYTES;
static_assert(SMEM_BYTES <= 99 * 1024, "sm_120 allows 99 KB of dynamic shared memory per block");

struct BwdArgs {
    const float2* ld;
    float* dq_acc;
    bf16* dqkv;
    const float* sinT;
    const float* cosT;
    const unsigned* bits;
    int N, H, BH, Npad, prefix, mask_rows, mask_key0, nq, kt_tiles;
    float scale_log2, scale;
};

// ---- preprocess ---------------------------------------------------------------------------------------------------
__global__ void bwd_pre_kernel(const bf16* __restrict__ O, const bf16* __restrict__ dO, const float* __restrict__ lse2, float2* __restrict__ ld,
                               float* __restrict__ dq_acc, int N, int H, int Npad, int rows_total) {
    const int idx = blockIdx.x * 32 + (threadIdx.x >> 3);
    const int l8 = threadIdx.x & 7;
    if (idx >= rows_total) return;
    const int bh = idx / Npad, row = idx - bh * Npad;
    const int b = bh / H, h = bh - b * H;
    float2 out = make_float2(0.f, 0.f);
    if (row < N) {
        const size_t off = (((size_t)b * N + row) * H + h) * 64 + l8 * 8;
        const uint4 o4 = *reinterpret_cast<const uint4*>(O + off);
        const uint4 d4 = *reinterpret_cast<const uint4*>(dO + off);
        const unsigned ow[4] = {o4.x, o4.y, o4.z, o4.w}, dw[4] = {d4.x, d4.y, d4.z, d4.w};
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc = fmaf(bf16lo(ow[i]), bf16lo(dw[i]), fmaf(bf16hi(ow[i]), bf16hi(dw[i]), acc));
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        acc += __shfl_xor_sync(0xffffffffu, acc, 2);
        acc += __shfl_xor_sync(0xffffffffu, acc, 4);
        out = make_float2(lse2[(size_t)bh * N + row], acc);
    }
    if (l8 == 0) ld[idx] = out;
    float4* dst = reinterpret_cast<float4*>(dq_acc + (size_t)idx * 64 + l8 * 8);
    dst[0] = make_float4(0.f, 0.f, 0.f, 0.f);
    dst[1] = make_float4(0.f, 0.f, 0.f, 0.f);
}

// ---- dQ conversion: scale, inverse RoPE, bf16, packed store -----------------------------------------------------
template <bool ROPE>
__global__ void bwd_dq_kernel(const float* __restrict__ dq_acc, bf16* __restrict__ dqkv, const float* __restrict__ sinT, const float* __restrict__ cosT,
                              int N, int H, int Npad, int prefix, float scale, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = i & 3, r = i >> 2;
    const int bh = r / N, row = r - bh * N;
    const int b = bh / H, h = bh - b * H;
    const float* src = dq_acc + ((size_t)bh * Npad + row) * 64 + 8 * c;
    float x1[8], x2[8];
    *reinterpret_cast<float4*>(x1) = *reinterpret_cast<const float4*>(src);
    *reinterpret_cast<float4*>(x1 + 4) = *reinterpret_cast<const float4*>(src + 4);
    *reinterpret_cast<float4*>(x2) = *reinterpret_cast<const float4*>(src + 32);
    *reinterpret_cast<float4*>(x2 + 4) = *reinterpret_cast<const float4*>(src + 36);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        x1[j] *= scale;
        x2[j] *= scale;
    }
    const int pos = row - prefix;
    if (ROPE && pos >= 0 && pos < 1024) {
        const int y = pos >> 5, x = pos & 31;
        const int ti = c < 2 ? (y * ROPE_W) * 64 + 8 * c : x * 64 + 16 + 8 * (c - 2);
        float cv[8], sv[8];
        *reinterpret_cast<float4*>(cv) = *reinterpret_cast<const float4*>(cosT + ti);
        *reinterpret_cast<float4*>(cv + 4) = *reinterpret_cast<const float4*>(cosT + ti + 4);
        *reinterpret_cast<float4*>(sv) = *reinterpret_cast<const float4*>(sinT + ti);
        *reinterpret_cast<float4*>(sv + 4) = *reinterpret_cast<const float4*>(sinT + ti + 4);
#pragma unroll
        for (int j = 0; j < 8; ++j) {  // dx1 = dy1*cos + dy2*sin ; dx2 = dy2*cos - dy1*sin (rope_apply backward)
            const float d1 = x1[j], d2 = x2[j];
            x1[j] = __fadd_rn(__fmul_rn(d1, cv[j]), __fmul_rn(d2, sv[j]));
            x2[j] = __fsub_rn(__fmul_rn(d2, cv[j]), __fmul_rn(d1, sv[j]));
        }
    }
    uint4 o1, o2;
    o1.x = pack_bf16x2(x1[0], x1[1]); o1.y = pack_bf16x2(x1[2], x1[3]); o1.z = pack_bf16x2(x1[4], x1[5]); o1.w = pack_bf16x2(x1[6], x1[7]);
    o2.x = pack_bf16x2(x2[0], x2[1]); o2.y = pack_bf16x2(x2[2], x2[3]); o2.z = pack_bf16x2(x2[4], x2[5]); o2.w = pack_bf16x2(x2[6], x2[7]);
    bf16* dst = dqkv + (((size_t)b * N + row) * 3 + 0) * H * 64 + h * 64 + 8 * c;
    *reinterpret_cast<uint4*>(dst) = o1;
    *reinterpret_cast<uint4*>(dst + 32) = o2;
}

// ---- main kernel ------------------------------------------------------------------------------------------------------
// q / k come already rotated (the forward saved qk_rot), so there is no rotator warp: 8 warps, tid 0 issues the TMA loads
// (3-stage ring of Q|dO tiles + (lse2, Dv) rows), V is staged once through the dS buffer 1 and held in registers.
template <bool ROPE, bool MASK>
__global__ void __launch_bounds__(THREADS, 1) bwd_kv_kernel(const __grid_constant__ CUtensorMap tmK, const __grid_constant__ CUtensorMap tmQ,
                                                            const __grid_constant__ CUtensorMap tmV, const __grid_constant__ CUtensorMap tmDO, BwdArgs a) {
    extern __shared__ __align__(1024) unsigned char smem[];
    bf16* Ks = reinterpret_cast<bf16*>(smem);
    unsigned char* stage = smem + OFF_STAGE;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
    uint64_t* k_full = bars;
    uint64_t* v_full = bars + 1;
    uint64_t* full = bars + 2;      // [3]
    uint64_t* empty = bars + 5;     // [3]
    uint64_t* ds_full = bars + 8;   // [2] dS^T buffer written by every warp
    uint64_t* ds_empty = bars + 10; // [2] dS^T buffer consumed by every warp
    float* ldS = reinterpret_cast<float*>(smem + OFF_LD);
    bf16* dS = reinterpret_cast<bf16*>(smem + OFF_DS);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int bh = blockIdx.x / a.kt_tiles, kt = blockIdx.x % a.kt_tiles;  // head-major: the CTAs of one head run together (shared Q/dO and dQ rows in L2)
    const int b = bh / a.H, h = bh - b * a.H;
    const int N = a.N, H = a.H, nq = a.nq;
    const int k0 = kt * KB;
    const int n_active = min(CW, cdiv(N - k0, 16));
    const int qstart = (int)(((long long)kt * nq) / a.kt_tiles);  // stagger the query order: concurrent CTAs of a head hit different dQ rows

    if (tid == 0) {
        mbar_init(k_full, 1);
        mbar_init(v_full, 1);
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], n_active);
        }
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            mbar_init(&ds_full[s], n_active);
            mbar_init(&ds_empty[s], n_active);
        }
        fence_mbar_init();
    }
    __syncthreads();

    // load of query tile i (loop position) into stage i % 3
    auto produce = [&](int i) {
        const int s = i % STAGES, use = i / STAGES;
        if (use > 0) mbar_wait(&empty[s], (use - 1) & 1);
        const int q0 = ((i + qstart) % nq) * QB;
        unsigned char* st = stage + s * STG_BYTES;
        mbar_arrive_expect_tx(&full[s], STG_BYTES + LD_BYTES);
        tma_load_4d(st, &tmQ, &full[s], 0, h, q0, b);
        tma_load_4d(st + STG_TILE, &tmDO, &full[s], 0, h, q0, b);
        bulk_load(ldS + s * (LD_BYTES / 4), a.ld + (size_t)bh * a.Npad + q0, LD_BYTES, &full[s]);
    };
    bf16* Vs = dS + KB * QB;  // dS buffer 1 stages the V tile
    if (tid == 0) {
        prefetch_tensormap(&tmK);
        prefetch_tensormap(&tmQ);
        prefetch_tensormap(&tmV);
        prefetch_tensormap(&tmDO);
        mbar_arrive_expect_tx(k_full, K_BYTES);
        tma_load_4d(Ks, &tmK, k_full, 0, H + h, k0, b);
        mbar_arrive_expect_tx(v_full, K_BYTES);
        tma_load_4d(Vs, &tmV, v_full, 0, 2 * H + h, k0, b);
        for (int i = 0; i < min(nq, STAGES); ++i) produce(i);
    }

    // ---- V into registers (all warps), then one block barrier: the dS buffer 1 is reused afterwards
    const int arow = warp * 16 + (lane & 15);
    unsigned vf[4][4];
    if (warp < n_active) {
        mbar_wait(v_full, 0);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) ldmatrix_x4(vf[kk], Vs + swz_off(arow, 2 * kk + (lane >> 4)));
    }
    __syncthreads();
    if (warp >= n_active) return;
    mbar_wait(k_full, 0);

    float dk[8][4], dv[8][4];
#pragma unroll
    for (int dj = 0; dj < 8; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) dk[dj][e] = dv[dj][e] = 0.f;

    const int kw0 = k0 + warp * 16;
    const bool kv_tail = k0 + KB > N;
    float* dq_base = a.dq_acc + (size_t)bh * a.Npad * 64;

    auto dq_product = [&](int j) {
        const int q0 = ((j + qstart) % nq) * QB;
        const int qv = min(QB, N - q0);
        const bf16* dsb = dS + (j & 1) * (KB * QB);
        if (n_active == CW && qv == QB) {  // common case: all eight warps, a full 64-row tile, one task per warp, no predicates
            const int mt = warp & 3, nw = warp >> 2;
            float acc[4][4];
#pragma unroll
            for (int dj = 0; dj < 4; ++dj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[dj][e] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                unsigned af[4];
                {
                    const int i = lane >> 3;
                    const int krow = kk * 16 + (i >> 1) * 8 + (lane & 7);
                    ldmatrix_x4_trans(af, dsb + swz_off(krow, mt * 2 + (i & 1)));
                }
#pragma unroll
                for (int dj = 0; dj < 4; dj += 2) {
                    const int row = kk * 16 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4_trans(r, Ks + swz_off(row, nw * 4 + dj + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[1]};
                    const unsigned b1[2] = {r[2], r[3]};
                    mma_bf16_16816(acc[dj], af, b0);
                    mma_bf16_16816(acc[dj + 1], af, b1);
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&ds_empty[j & 1]);
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                float* dst = dq_base + (size_t)(q0 + mt * 16 + g + hh * 8) * 64 + nw * 32 + c2;
#pragma unroll
                for (int dj = 0; dj < 4; ++dj) red_add2(dst + dj * 8, acc[dj][2 * hh], acc[dj][2 * hh + 1]);
            }
            return;
        }
        for (int task = warp; task < 8; task += n_active) {
            const int mt = task & 3, nw = task >> 2;
            const bool last = task + n_active >= 8;  // this warp's last task: afterwards the dS buffer is no longer read by it
            if (mt * 16 >= qv) {
                if (last) {
                    __syncwarp();
                    if (lane == 0) mbar_arrive(&ds_empty[j & 1]);
                }
                continue;
            }
            float acc[4][4];
#pragma unroll
            for (int dj = 0; dj < 4; ++dj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[dj][e] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                if (kk >= n_active) break;
                unsigned af[4];
                {
                    const int i = lane >> 3;
                    const int krow = kk * 16 + (i >> 1) * 8 + (lane & 7);
                    ldmatrix_x4_trans(af, dsb + swz_off(krow, mt * 2 + (i & 1)));
                }
#pragma unroll
                for (int dj = 0; dj < 4; dj += 2) {
                    const int row = kk * 16 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4_trans(r, Ks + swz_off(row, nw * 4 + dj + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[1]};
                    const unsigned b1[2] = {r[2], r[3]};
                    mma_bf16_16816(acc[dj], af, b0);
                    mma_bf16_16816(acc[dj + 1], af, b1);
                }
            }
            if (last) {
                __syncwarp();
                if (lane == 0) mbar_arrive(&ds_empty[j & 1]);
            }
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int q = q0 + mt * 16 + g + hh * 8;
                if (q >= N) continue;
                float* dst = dq_base + (size_t)q * 64 + nw * 32 + c2;
#pragma unroll
                for (int dj = 0; dj < 4; ++dj) red_add2(dst + dj * 8, acc[dj][2 * hh], acc[dj][2 * hh + 1]);
            }
        }
    };

    for (int i = 0; i < nq; ++i) {
        const int s = i % STAGES;
        const int q0 = ((i + qstart) % nq) * QB;
        const int qv = min(QB, N - q0);  // valid query rows of this tile (5 / 41 for the tail tile)
        mbar_wait(&full[s], (i / STAGES) & 1);
        const bf16* qs = reinterpret_cast<const bf16*>(stage + s * STG_BYTES);
        const bf16* dos = qs + QB * 64;
        const float* lds = ldS + s * (LD_BYTES / 4);
        bf16* dSs = dS + (i & 1) * (KB * QB);
        unsigned pa[4][4];
        auto compute = [&](auto tail_c) {
        constexpr bool tail = decltype(tail_c)::value;

        float sacc[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) sacc[n][e] = 0.f;
        // S^T = K_w Q^T (16 keys x 64 rows)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            unsigned af[4];
            ldmatrix_x4(af, Ks + swz_off(arow, 2 * kk + (lane >> 4)));
#pragma unroll
            for (int n = 0; n < 8; n += 2) {
                if (tail && n * 8 >= qv) continue;
                const int row = n * 8 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4(r, qs + swz_off(row, 2 * kk + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[2]};
                const unsigned b1[2] = {r[1], r[3]};
                mma_bf16_16816(sacc[n], af, b0);
                if (!tail || (n + 1) * 8 < qv) mma_bf16_16816(sacc[n + 1], af, b1);
            }
        }
        // P^T and Dv
        float dvec[8][2];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            if (tail && n * 8 >= qv) {
                sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
                dvec[n][0] = dvec[n][1] = 0.f;
                continue;
            }
            const float4 l4 = *reinterpret_cast<const float4*>(lds + 2 * (n * 8 + c2));
            sacc[n][0] = ex2(fmaf(sacc[n][0], a.scale_log2, -l4.x));
            sacc[n][1] = ex2(fmaf(sacc[n][1], a.scale_log2, -l4.z));
            sacc[n][2] = ex2(fmaf(sacc[n][2], a.scale_log2, -l4.x));
            sacc[n][3] = ex2(fmaf(sacc[n][3], a.scale_log2, -l4.z));
            dvec[n][0] = l4.y;
            dvec[n][1] = l4.w;
        }
        // keys past N have zero K rows (finite, meaningless P): drop them
        if (kv_tail) {
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (kw0 + g + (e >> 1) * 8 >= N) sacc[n][e] = 0.f;
        }
        if constexpr (MASK) {
            if (q0 < a.mask_rows && kw0 + 15 >= a.mask_key0) {  // warp-uniform
                const unsigned* bb = a.bits + (size_t)b * a.mask_rows * 32;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int key = kw0 + g + 8 * hh;
                    const int p = key - a.mask_key0;
                    if (p < 0 || key >= N) continue;
                    const int wi = p >> 5, bi = p & 31;
#pragma unroll
                    for (int n = 0; n < 8; ++n)
#pragma unroll
                        for (int e01 = 0; e01 < 2; ++e01) {
                            const int q = q0 + n * 8 + c2 + e01;
                            if (q < a.mask_rows) {
                                const unsigned w = __ldg(bb + (size_t)q * 32 + wi);
                                if (!((w >> bi) & 1u)) sacc[n][2 * hh + e01] = 0.f;
                            }
                        }
                }
            }
        }
        // dP^T = V_w dO^T
        float dp[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) dp[n][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int n = 0; n < 8; n += 2) {
                if (tail && n * 8 >= qv) continue;
                const int row = n * 8 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4(r, dos + swz_off(row, 2 * kk + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[2]};
                const unsigned b1[2] = {r[1], r[3]};
                mma_bf16_16816(dp[n], vf[kk], b0);
                if (!tail || (n + 1) * 8 < qv) mma_bf16_16816(dp[n + 1], vf[kk], b1);
            }
        }
        // dV += P^T dO
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
            pa[kq][0] = pack_bf16x2(sacc[2 * kq][0], sacc[2 * kq][1]);
            pa[kq][1] = pack_bf16x2(sacc[2 * kq][2], sacc[2 * kq][3]);
            pa[kq][2] = pack_bf16x2(sacc[2 * kq + 1][0], sacc[2 * kq + 1][1]);
            pa[kq][3] = pack_bf16x2(sacc[2 * kq + 1][2], sacc[2 * kq + 1][3]);
        }
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
            if (tail && kq * 16 >= qv) continue;
#pragma unroll
            for (int dj = 0; dj < 8; dj += 2) {
                const int row = kq * 16 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4_trans(r, dos + swz_off(row, dj + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(dv[dj], pa[kq], b0);
                mma_bf16_16816(dv[dj + 1], pa[kq], b1);
            }
        }
        // dS^T = P^T (dP^T - Dv)
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            sacc[n][0] *= dp[n][0] - dvec[n][0];
            sacc[n][1] *= dp[n][1] - dvec[n][1];
            sacc[n][2] *= dp[n][2] - dvec[n][0];
            sacc[n][3] *= dp[n][3] - dvec[n][1];
        }
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
            pa[kq][0] = pack_bf16x2(sacc[2 * kq][0], sacc[2 * kq][1]);
            pa[kq][1] = pack_bf16x2(sacc[2 * kq][2], sacc[2 * kq][3]);
            pa[kq][2] = pack_bf16x2(sacc[2 * kq + 1][0], sacc[2 * kq + 1][1]);
            pa[kq][3] = pack_bf16x2(sacc[2 * kq + 1][2], sacc[2 * kq + 1][3]);
        }
        // dK += dS^T Q
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
            if (tail && kq * 16 >= qv) continue;
#pragma unroll
            for (int dj = 0; dj < 8; dj += 2) {
                const int row = kq * 16 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4_trans(r, qs + swz_off(row, dj + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(dk[dj], pa[kq], b0);
                mma_bf16_16816(dk[dj + 1], pa[kq], b1);
            }
        }
        };
        if (qv < QB) compute(std::true_type{});
        else compute(std::false_type{});
        fence_proxy_async_smem();  // this warp's reads of the Q|dO stage precede the TMA refill
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
        // dS^T to shared memory (A operand of the dQ product): rows = keys, cols = query rows
        if (i >= 2) mbar_wait(&ds_empty[i & 1], ((i >> 1) - 1) & 1);
        {
            const int r0 = warp * 16 + g;
#pragma unroll
            for (int kq = 0; kq < 4; ++kq) {
                *reinterpret_cast<unsigned*>(dSs + swz_off(r0, 2 * kq) + c2) = pa[kq][0];
                *reinterpret_cast<unsigned*>(dSs + swz_off(r0 + 8, 2 * kq) + c2) = pa[kq][1];
                *reinterpret_cast<unsigned*>(dSs + swz_off(r0, 2 * kq + 1) + c2) = pa[kq][2];
                *reinterpret_cast<unsigned*>(dSs + swz_off(r0 + 8, 2 * kq + 1) + c2) = pa[kq][3];
            }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&ds_full[i & 1]);
        if (i > 0) {  // dQ product of the previous iteration (its dS^T was published by every warp one iteration ago)
            const int j = i - 1;
            mbar_wait(&ds_full[j & 1], (j >> 1) & 1);
            if (tid == 0 && i + 2 < nq) produce(i + 2);  // every warp is past compute(i-1): refill its stage
            dq_product(j);
        }
    }
    {
        const int j = nq - 1;
        mbar_wait(&ds_full[j & 1], (j >> 1) & 1);
        dq_product(j);
    }

    // ---- epilogue: dK (scale, inverse RoPE) and dV straight into the packed dqkv
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int key = kw0 + g + 8 * hh;
        if (key >= N) continue;
        bf16* dkrow = a.dqkv + (((size_t)b * N + key) * 3 + 1) * H * 64 + h * 64;
        bf16* dvrow = a.dqkv + (((size_t)b * N + key) * 3 + 2) * H * 64 + h * 64;
        const int pos = key - a.prefix;
        const bool rot = ROPE && pos >= 0 && pos < 1024;
        const int y = pos >> 5, x = pos & 31;
#pragma unroll
        for (int dj = 0; dj < 4; ++dj) {
            float d1a = dk[dj][2 * hh] * a.scale, d1b = dk[dj][2 * hh + 1] * a.scale;
            float d2a = dk[dj + 4][2 * hh] * a.scale, d2b = dk[dj + 4][2 * hh + 1] * a.scale;
            if (rot) {
                const int d0 = dj * 8 + c2;
                const int ti = dj < 2 ? (y * ROPE_W) * 64 + d0 : x * 64 + d0;  // dims 0..15 follow y, 16..31 follow x
                const float2 cc = *reinterpret_cast<const float2*>(a.cosT + ti);
                const float2 ss = *reinterpret_cast<const float2*>(a.sinT + ti);
                const float n1a = __fadd_rn(__fmul_rn(d1a, cc.x), __fmul_rn(d2a, ss.x));
                const float n1b = __fadd_rn(__fmul_rn(d1b, cc.y), __fmul_rn(d2b, ss.y));
                const float n2a = __fsub_rn(__fmul_rn(d2a, cc.x), __fmul_rn(d1a, ss.x));
                const float n2b = __fsub_rn(__fmul_rn(d2b, cc.y), __fmul_rn(d1b, ss.y));
                d1a = n1a; d1b = n1b; d2a = n2a; d2b = n2b;
            }
            *reinterpret_cast<unsigned*>(dkrow + dj * 8 + c2) = pack_bf16x2(d1a, d1b);
            *reinterpret_cast<unsigned*>(dkrow + (dj + 4) * 8 + c2) = pack_bf16x2(d2a, d2b);
        }
#pragma unroll
        for (int dj = 0; dj < 8; ++dj) *reinterpret_cast<unsigned*>(dvrow + dj * 8 + c2) = pack_bf16x2(dv[dj][2 * hh], dv[dj][2 * hh + 1]);
    }
}

template <bool ROPE, bool MASK>
void launch_kv(const Tensor& qkv, const Tensor& qk, const Tensor& dO, const float2* ld, float* dq_acc, Tensor& dqkv, const float* sinT, const float* cosT,
               const unsigned* bits, int H, int Npad, int prefix, int mask_rows, int mask_key0, int qk_slots, cudaStream_t stream) {
    const int B = qkv.size(0), N = qkv.size(1);
    static bool attr = false;
    if (!attr) {
        A12_CUDA_CHECK(cudaFuncSetAttribute(bwd_kv_kernel<ROPE, MASK>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        attr = true;
    }
    const CUtensorMap tmK = make_slot_map(qk.data_ptr(), B, N, qk_slots * H, KB);
    const CUtensorMap tmQ = make_slot_map(qk.data_ptr(), B, N, qk_slots * H, QB);
    const CUtensorMap tmV = make_slot_map(qkv.data_ptr(), B, N, 3 * H, KB);
    const CUtensorMap tmDO = make_slot_map(dO.data_ptr(), B, N, H, QB);
    const int kt_tiles = cdiv(N, KB);
    BwdArgs a{ld, dq_acc, reinterpret_cast<bf16*>(dqkv.data_ptr()), sinT, cosT, bits, N, H, B * H, Npad, prefix, mask_rows, mask_key0, cdiv(N, QB), kt_tiles,
              0.125f * kLog2e, 0.125f};
    bwd_kv_kernel<ROPE, MASK><<<B * H * kt_tiles, THREADS, SMEM_BYTES, stream>>>(tmK, tmQ, tmV, tmDO, a);
    A12_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

// qk_rot: rotated q|k saved by attn_fwd ([B,N,2,H,64]); None when the forward ran without RoPE (then q, k are read from qkv).
Tensor attn_bwd(const Tensor& qkv, const std::optional<Tensor>& qk_rot, const Tensor& O, const Tensor& lse2, const Tensor& dO, int64_t heads,
                const std::optional<Tensor>& sinT, const std::optional<Tensor>& cosT, const std::optional<Tensor>& bits, int64_t prefix, int64_t mask_rows,
                int64_t mask_key0) {
    TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.is_contiguous(), "qkv must be contiguous bf16 cuda");
    TORCH_CHECK(O.is_contiguous() && dO.is_contiguous() && O.scalar_type() == at::kBFloat16 && dO.scalar_type() == at::kBFloat16, "O / dO must be contiguous bf16");
    const c10::cuda::CUDAGuard guard(qkv.device());
    const int B = qkv.size(0), N = qkv.size(1), H = heads;
    const int Npad = cdiv(N, QB) * QB;
    const bool rope = sinT.has_value() && prefix >= 0;
    TORCH_CHECK(!rope || qk_rot.has_value(), "RoPE backward needs the forward's rotated q|k (qk_rot)");
    Tensor dqkv = at::empty({B, N, 3, H, 64}, qkv.options());
    Tensor ld = at::empty({(int64_t)B * H * Npad, 2}, qkv.options().dtype(at::kFloat));
    Tensor dq_acc = at::empty({(int64_t)B * H * Npad, 64}, qkv.options().dtype(at::kFloat));
    const bool mask = bits.has_value() && mask_rows > 0;
    const float* s = rope ? sinT->data_ptr<float>() : nullptr;
    const float* c = rope ? cosT->data_ptr<float>() : nullptr;
    const unsigned* bp = mask ? reinterpret_cast<const unsigned*>(bits->data_ptr()) : nullptr;
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const int rows_total = B * H * Npad;
    bwd_pre_kernel<<<cdiv(rows_total, 32), 256, 0, st>>>(reinterpret_cast<const bf16*>(O.data_ptr()), reinterpret_cast<const bf16*>(dO.data_ptr()), lse2.data_ptr<float>(),
                                                          reinterpret_cast<float2*>(ld.data_ptr()), dq_acc.data_ptr<float>(), N, H, Npad, rows_total);
    A12_CUDA_CHECK(cudaGetLastError());
    const float2* ldp = reinterpret_cast<const float2*>(ld.data_ptr());
    float* dqa = dq_acc.data_ptr<float>();
    const Tensor& qk = rope ? *qk_rot : qkv;
    const int qk_slots = rope ? 2 : 3;
    const int pf = rope ? (int)prefix : (1 << 30);
    if (rope && mask) launch_kv<true, true>(qkv, qk, dO, ldp, dqa, dqkv, s, c, bp, H, Npad, pf, (int)mask_rows, (int)mask_key0, qk_slots, st);
    else if (rope) launch_kv<true, false>(qkv, qk, dO, ldp, dqa, dqkv, s, c, bp, H, Npad, pf, 0, 0, qk_slots, st);
    else if (mask) launch_kv<false, true>(qkv, qk, dO, ldp, dqa, dqkv, s, c, bp, H, Npad, pf, (int)mask_rows, (int)mask_key0, qk_slots, st);
    else launch_kv<false, false>(qkv, qk, dO, ldp, dqa, dqkv, s, c, bp, H, Npad, pf, 0, 0, qk_slots, st);
    const int total = B * H * N * 4;
    if (rope) bwd_dq_kernel<true><<<cdiv(total, 256), 256, 0, st>>>(dqa, reinterpret_cast<bf16*>(dqkv.data_ptr()), s, c, N, H, Npad, (int)prefix, 0.125f, total);
    else bwd_dq_kernel<false><<<cdiv(total, 256), 256, 0, st>>>(dqa, reinterpret_cast<bf16*>(dqkv.data_ptr()), nullptr, nullptr, N, H, Npad, pf, 0.125f, total);
    A12_CUDA_CHECK(cudaGetLastError());
    return dqkv;
}

}  // namespace a12
