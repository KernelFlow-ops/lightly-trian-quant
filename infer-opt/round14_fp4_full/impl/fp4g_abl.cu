// Round 14 step 2 EXPERIMENT (not used by the model): fp4g_core.cu with ablation switches to find the bottleneck of the main loop.
//   -DABL_NOSF: scale registers constant (no scale LDS)   -DABL_NOLDM: fragments loaded only at step 0 of a stage   -DABL_NOCP: no cp.async in the loop   -DABL_NOMMA: no mma
// Round 14 step 2: own block-scaled FP4 GEMM for sm_120a.
//   C[M,N] = sum_k (sfa[m][k/V] A[m][k]) (sfw[n][k/V] W[n][k])   (then * ts, + epilogue)
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale  --  FMT 0: NVFP4 (scale_vec::4X, ue4m3 per 16, K64 step = 1 scale word)
//                                                                      FMT 1: MXFP4 (scale_vec::2X, ue8m0 per 32, 1 scale word = 2 k64 steps, byte-id 0 / 2)
// Structure: round10 mg8_core.cu (cubin per config, Params struct by value, cp.async multi-stage, ldmatrix, XOR-swizzled 16B chunks, R5 epilogues),
// MMA / scale-register layout from aidxhxr/5090-blackwell-kernels (Spark, MIT) src/kernels/fp4gemm_tile.cuh: operands are K-major packed e2m1 (2 per byte), the m16n8k64
// fragments are the m16n8k32 fp8 fragments byte for byte (one ldmatrix.x4 per 16 rows x 32 bytes), block scales come in the 128x4 blocked layout (cuBLASLt / QuACK):
// tile = 512 B, byte (r % 32) * 16 + ((r / 32) % 4) * 4 + j; scale registers are read straight from shared memory (4 aligned bytes per lane, no repacking).
// Config macros: BM BN BKB(bytes of K per stage row, 64 or 128) ST WM WN MINB FMT EPI
// EPI 0: C bf16 [M,N]  = acc * ts                                            (head ConvT)
//     1: C fp32 [M,N]  = R + gamma * (acc * ts + bias)                      (proj / fc2)
//     2: C fp16 [3,B,Ntok,H,64] = bias + RoPE(fp32) + layout                 (qkv; BN multiple of 64, whole heads)
//     3: GELU(acc * ts + bias) -> FP4 (same format as the input), q [M,N/2] + blocked scales, static per-tensor scale of the output (fc1 -> fc2 operand)
//     4: C bf16 [M,N]  = GELU(acc * ts + bias)                               (tests)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef BM
#define BM 128
#define BN 128
#define BKB 64
#define ST 4
#define WM 2
#define WN 4
#define MINB 1
#define FMT 0
#define EPI 0
#endif
#ifndef PDLW
#define PDLW 0
#endif

struct Params {
    const uint8_t* A; const uint8_t* W; const uint8_t* SA; const uint8_t* SW;
    const float* ts; const float* bias; const float* gamma; const float* R; void* C;
    const float* sin_t; const float* cos_t;
    uint8_t* Qo; uint8_t* SFo; const float* tso;
    int M, N, K, Ntok, prefix, d, H, sfo_kt, pad0, pad1;
};

constexpr int NTH = WM * WN * 32;
constexpr int WTM = BM / WM, WTN = BN / WN;
constexpr int MT = WTM / 16, NT = WTN / 8;
constexpr int STEPS = BKB / 32;                   // k64 steps per stage
constexpr int SPT = FMT == 0 ? 1 : 2;             // k64 steps per scale word
constexpr int TILES = STEPS / SPT;                // scale tiles (512 B per 128 rows) per stage and row block
constexpr int CPR = BKB / 16;                     // 16 B chunks per smem row
constexpr int SA_RB = BM > 128 ? BM / 128 : 1, SB_RB = BN > 128 ? BN / 128 : 1;
constexpr int SA_STRIDE = TILES * 512, SB_STRIDE = TILES * 512;
constexpr int A_STAGE = BM * BKB, B_STAGE = BN * BKB;
constexpr int SA_OFF = A_STAGE + B_STAGE, SB_OFF = SA_OFF + SA_RB * SA_STRIDE;
constexpr int STAGE = SB_OFF + SB_RB * SB_STRIDE;
constexpr int A_CH = BM * CPR, B_CH = BN * CPR;
constexpr int A_LD = (A_CH + NTH - 1) / NTH, B_LD = (B_CH + NTH - 1) / NTH;
constexpr int SFA_CH = SA_RB * TILES * 32, SFB_CH = SB_RB * TILES * 32, SF_CH = SFA_CH + SFB_CH;
constexpr int SF_LD = (SF_CH + NTH - 1) / NTH;
static_assert(BKB == 64 || BKB == 128, "BKB");
static_assert(BM % (16 * WM) == 0 && BN % (8 * WN) == 0 && (BM <= 128 ? 128 % BM == 0 : BM % 128 == 0), "tile");
static_assert(NT % 2 == 0, "NT even");
static_assert(STEPS % SPT == 0, "stage = whole scale tiles");
static_assert(EPI != 2 || (BN % 64 == 0 && WN % (BN / 64) == 0 && (NT / 2) * (WN / (BN / 64)) == 4), "rope warp layout");
static_assert(EPI != 3 || (FMT == 0 ? NT % 2 == 0 : NT % 4 == 0), "quant-out blocks must be whole inside a warp tile");

constexpr float RCP6 = 1.0f / 6.0f;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int sz) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

template <int TA, int TB>
__device__ __forceinline__ void mma_nv4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, %12}, %11, {0, %13};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(TA), "n"(TB));
}
template <int BID, int TA, int TB>
__device__ __forceinline__ void mma_mx4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%12, %14};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BID), "n"(TA), "n"(TB));
}
template <int BID>
__device__ __forceinline__ void mma_sel(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb, int ta, int tb) {
    if constexpr (FMT == 0) {
        if (ta == 0 && tb == 0) mma_nv4<0, 0>(d, a, b0, b1, sa, sb);
        if (ta == 0 && tb == 1) mma_nv4<0, 1>(d, a, b0, b1, sa, sb);
        if (ta == 1 && tb == 0) mma_nv4<1, 0>(d, a, b0, b1, sa, sb);
        if (ta == 1 && tb == 1) mma_nv4<1, 1>(d, a, b0, b1, sa, sb);
    } else {
        if (ta == 0 && tb == 0) mma_mx4<BID, 0, 0>(d, a, b0, b1, sa, sb);
        if (ta == 0 && tb == 1) mma_mx4<BID, 0, 1>(d, a, b0, b1, sa, sb);
        if (ta == 1 && tb == 0) mma_mx4<BID, 1, 0>(d, a, b0, b1, sa, sb);
        if (ta == 1 && tb == 1) mma_mx4<BID, 1, 1>(d, a, b0, b1, sa, sb);
    }
}

// swizzled 16 B unit index of (row, chunk) in a [rows][BKB] byte tile
__device__ __forceinline__ int swz_unit(int r, int c) {
    int u = r * CPR + c;
    return u ^ ((u >> 3) & 7);
}
// which 8-column block of the CTA tile does warp column wn's nj-th fragment cover?  (EPI 2: RoPE partners = blocks b and b + 4 of a head sit in one thread)
__device__ __forceinline__ int blk_of(int wn, int nj) {
    if constexpr (EPI == 2) {
        constexpr int WPH = WN / (BN / 64);
        int head = wn / WPH, lw = wn % WPH;
        int pi = lw * (NT / 2) + (nj % (NT / 2));
        return head * 8 + pi + 4 * (nj / (NT / 2));
    }
    return wn * NT + nj;
}
// byte offset of the scale word of block-relative row `rel` inside one scale tile of an operand with RB row blocks; q0 = (row0 / 32) % 4
template <int RB, int STRIDE>
__device__ __forceinline__ int sf_word_off(int rel, int q0) {
    const int r = rel & 127;
    return (rel >> 7) * STRIDE + (r & 31) * 16 + ((q0 + (r >> 5)) & 3) * 4;
}

// ---- FP4 quantisation helpers (bit-spec: train-opt fp4_ref.py, == infer-opt quant_ref.py)
__device__ __forceinline__ uint32_t f32_to_e4m3(float t) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(t));
    return (uint32_t)(r & 0xffu);
}
__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((uint16_t)b));
    return __half2float(__ushort_as_half((unsigned short)(h & 0xffffu)));
}
__device__ __forceinline__ uint32_t pack2_e2m1(float lo, float hi) {   // lo -> low nibble
    uint16_t r;
    asm("{\n.reg .b8 b;\ncvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\nmov.b16 %0, {b, 0};\n}" : "=h"(r) : "f"(lo), "f"(hi));
    return (uint32_t)(r & 0xffu);
}
__device__ __forceinline__ uint32_t sf_blocked_off(int row, int kb, int kt_tiles) {   // byte offset of scale (row, k-block kb) in the 128x4 blocked layout
    return ((uint32_t)(row >> 7) * kt_tiles + (kb >> 2)) * 512u + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (kb & 3);
}

extern "C" __global__ void __launch_bounds__(NTH, MINB) gemm_kernel(const Params p) {
    extern __shared__ __align__(1024) uint8_t smem[];
#ifdef ABL_EMPTY
    return;
#endif
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int K2 = p.K >> 1;                              // bytes per operand row
#ifdef ABL_NOLOOP
    const int KT = 0;
#else
    const int KT = p.K / (2 * BKB);
#endif
    const int ktiles_sf = FMT == 0 ? p.K / 64 : p.K / 128;   // scale tiles along K
    const uint32_t sm = smem_u32(smem);

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // ---- per-thread loader state
    const uint8_t* a_src[A_LD]; uint32_t a_dst[A_LD]; int a_sz[A_LD];
    const uint8_t* w_src[B_LD]; uint32_t w_dst[B_LD];
    const uint8_t* f_src[SF_LD]; uint32_t f_dst[SF_LD]; int f_sz[SF_LD];
    const int nrb_a = (p.M + 127) >> 7;
#pragma unroll
    for (int i = 0; i < A_LD; i++) {
        int u = tid + i * NTH;
        if (A_CH % NTH != 0 && u >= A_CH) u = 0;
        const int r = u / CPR, c = u % CPR;
        const bool ok = (m0 + r) < p.M;
        a_src[i] = p.A + (size_t)(ok ? (m0 + r) : 0) * K2 + c * 16;
        a_dst[i] = sm + (uint32_t)(swz_unit(r, c) * 16);
        a_sz[i] = ok ? 16 : 0;
    }
#pragma unroll
    for (int i = 0; i < B_LD; i++) {
        int u = tid + i * NTH;
        if (B_CH % NTH != 0 && u >= B_CH) u = 0;
        const int r = u / CPR, c = u % CPR;
        w_src[i] = p.W + (size_t)(n0 + r) * K2 + c * 16;
        w_dst[i] = sm + A_STAGE + (uint32_t)(swz_unit(r, c) * 16);
    }
#pragma unroll
    for (int i = 0; i < SF_LD; i++) {
        int u = tid + i * NTH;
        if (SF_CH % NTH != 0 && u >= SF_CH) u = 0;
        if (u < SFA_CH) {
            const int rb = u / (TILES * 32), t = (u / 32) % TILES, r = u % 32;
            const int rbg = m0 / 128 + rb;
            const bool ok = rbg < nrb_a;
            f_src[i] = p.SA + ((size_t)(ok ? rbg : 0) * ktiles_sf + t) * 512 + r * 16;
            f_dst[i] = sm + SA_OFF + rb * SA_STRIDE + t * 512 + r * 16;
            f_sz[i] = ok ? 16 : 0;
        } else {
            const int u2 = u - SFA_CH;
            const int rb = u2 / (TILES * 32), t = (u2 / 32) % TILES, r = u2 % 32;
            f_src[i] = p.SW + ((size_t)(n0 / 128 + rb) * ktiles_sf + t) * 512 + r * 16;
            f_dst[i] = sm + SB_OFF + rb * SB_STRIDE + t * 512 + r * 16;
            f_sz[i] = 16;
        }
    }
    // ---- ldmatrix offsets (bytes inside a stage), hoisted
    uint32_t a_ld[STEPS][MT], w_ld[STEPS][NT / 2];
#pragma unroll
    for (int s = 0; s < STEPS; s++) {
#pragma unroll
        for (int mi = 0; mi < MT; mi++) {
            const int r = wm * WTM + mi * 16 + (lane & 15), c = s * 2 + (lane >> 4);
            a_ld[s][mi] = swz_unit(r, c) * 16;
        }
#pragma unroll
        for (int j2 = 0; j2 < NT / 2; j2++) {
            const int blk = blk_of(wn, 2 * j2 + (lane >> 4));
            const int r = blk * 8 + (lane & 7), c = s * 2 + ((lane >> 3) & 1);
            w_ld[s][j2] = A_STAGE + swz_unit(r, c) * 16;
        }
    }
    // ---- scale word offsets (bytes inside a stage, tile 0), hoisted
    const int g = lane >> 2, cq = lane & 3;
    const int qa0 = (m0 >> 5) & 3, qb0 = (n0 >> 5) & 3;
    constexpr int NSA = (MT + 1) / 2, NSB = NT / 2;
    int sa_off[NSA], sb_off[NSB];
#pragma unroll
    for (int pp = 0; pp < NSA; pp++) {
        const int rel = wm * WTM + (MT >= 2 ? (2 * pp + (cq >> 1)) * 16 : 0) + g + 8 * (cq & 1);
        sa_off[pp] = SA_OFF + sf_word_off<SA_RB, SA_STRIDE>(rel, qa0);
    }
#pragma unroll
    for (int pp = 0; pp < NSB; pp++) {
        const int rel = blk_of(wn, 2 * pp + (cq & 1)) * 8 + g;
        sb_off[pp] = SB_OFF + sf_word_off<SB_RB, SB_STRIDE>(rel, qb0);
    }

    auto load_stage = [&](int st, int kt) {
        const uint32_t so = st * STAGE;
#pragma unroll
        for (int i = 0; i < A_LD; i++)
            if (A_CH % NTH == 0 || tid + i * NTH < A_CH) cp_async16(a_dst[i] + so, a_src[i] + (size_t)kt * BKB, a_sz[i]);
#pragma unroll
        for (int i = 0; i < B_LD; i++)
            if (B_CH % NTH == 0 || tid + i * NTH < B_CH) cp_async16(w_dst[i] + so, w_src[i] + (size_t)kt * BKB, 16);
#pragma unroll
        for (int i = 0; i < SF_LD; i++)
            if (SF_CH % NTH == 0 || tid + i * NTH < SF_CH) cp_async16(f_dst[i] + so, f_src[i] + (size_t)kt * TILES * 512, f_sz[i]);
    };

    // ---- prologue.  Weights / scales of W do not depend on the producer kernel; with PDL the dependent loads (A, SA) wait on the grid dependency.
    if (PDLW) pdl_wait();
#pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < KT) load_stage(s, s);
        cp_commit();
    }
    if (PDLW) pdl_trigger();

#pragma unroll 1
    for (int kt = 0; kt < KT; kt++) {
        cp_wait<ST - 2>();
        __syncthreads();
        {
            const int nk = kt + ST - 1;
#ifndef ABL_NOCP
            if (nk < KT) load_stage(nk % ST, nk);
#endif
            cp_commit();
        }
        const uint32_t sb_ = sm + (kt % ST) * STAGE;
        const uint8_t* sbase = smem + (kt % ST) * STAGE;
        uint32_t sfa[NSA], sfb[NSB];
#pragma unroll
        for (int s = 0; s < STEPS; s++) {
            if (s % SPT == 0) {
                const int t = s / SPT;
#ifdef ABL_NOSF
#pragma unroll
                for (int pp = 0; pp < NSA; pp++) sfa[pp] = 0x38383838u + (uint32_t)kt;
#pragma unroll
                for (int pp = 0; pp < NSB; pp++) sfb[pp] = 0x38383838u + (uint32_t)kt;
#else
#pragma unroll
                for (int pp = 0; pp < NSA; pp++) sfa[pp] = *reinterpret_cast<const uint32_t*>(sbase + sa_off[pp] + t * 512);
#pragma unroll
                for (int pp = 0; pp < NSB; pp++) sfb[pp] = *reinterpret_cast<const uint32_t*>(sbase + sb_off[pp] + t * 512);
#endif
            }
            uint32_t af[MT][4], bfr[NT][2];
#ifdef ABL_NOLDM
            if (s == 0) {
#endif
#pragma unroll
            for (int mi = 0; mi < MT; mi++) ldsm_x4(sb_ + a_ld[s][mi], af[mi][0], af[mi][1], af[mi][2], af[mi][3]);
#pragma unroll
            for (int j2 = 0; j2 < NT / 2; j2++) ldsm_x4(sb_ + w_ld[s][j2], bfr[2 * j2][0], bfr[2 * j2][1], bfr[2 * j2 + 1][0], bfr[2 * j2 + 1][1]);
#ifdef ABL_NOLDM
            } else {
#pragma unroll
                for (int mi = 0; mi < MT; mi++) { af[mi][0] = a_ld[s][mi]; af[mi][1] = af[mi][0] + 1; af[mi][2] = af[mi][0] + 2; af[mi][3] = af[mi][0] + 3; }
#pragma unroll
                for (int j = 0; j < NT; j++) { bfr[j][0] = w_ld[s][j / 2] + j; bfr[j][1] = bfr[j][0] + 1; }
            }
#endif
#pragma unroll
            for (int mi = 0; mi < MT; mi++)
#pragma unroll
                for (int nj = 0; nj < NT; nj++) {
#ifdef ABL_NOMMA
                    acc[mi][nj][0] += (float)(af[mi][0] ^ bfr[nj][0] ^ sfa[mi >> 1] ^ sfb[nj >> 1]); continue;
#endif
                    if constexpr (FMT == 0) mma_sel<0>(acc[mi][nj], af[mi], bfr[nj][0], bfr[nj][1], sfa[mi >> 1], sfb[nj >> 1], MT >= 2 ? (mi & 1) : 0, nj & 1);
                    else if ((s & 1) == 0) mma_sel<0>(acc[mi][nj], af[mi], bfr[nj][0], bfr[nj][1], sfa[mi >> 1], sfb[nj >> 1], MT >= 2 ? (mi & 1) : 0, nj & 1);
                    else mma_sel<2>(acc[mi][nj], af[mi], bfr[nj][0], bfr[nj][1], sfa[mi >> 1], sfb[nj >> 1], MT >= 2 ? (mi & 1) : 0, nj & 1);
                }
        }
    }
    cp_wait<0>();

    // ---------------------------------------------------------------- epilogues
    const float tsv = p.ts ? __ldg(p.ts) : 1.f;
#ifdef ABL_STG
    if (EPI == 0) {   // staged epilogue: bf16 tile -> smem [BM][BN+8] -> 16 B coalesced stores
        __syncthreads();
        constexpr int SLD = BN + 8;
        __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(smem);
#pragma unroll
        for (int mi = 0; mi < MT; mi++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int rl = wm * WTM + mi * 16 + g + hh * 8;
#pragma unroll
                for (int nj = 0; nj < NT; nj++)
                    *reinterpret_cast<__nv_bfloat162*>(stg + rl * SLD + (wn * WTN + nj * 8 + cq * 2)) =
                        __floats2bfloat162_rn(acc[mi][nj][hh * 2] * tsv, acc[mi][nj][hh * 2 + 1] * tsv);
            }
        __syncthreads();
        constexpr int CHR = BN / 8;                          // 16 B chunks per tile row
        for (int idx = tid; idx < BM * CHR; idx += NTH) {
            const int r = idx / CHR, ch = idx % CHR;
            const int grow = m0 + r;
            if (grow < p.M) *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(p.C) + (size_t)grow * p.N + n0 + ch * 8) = *reinterpret_cast<const uint4*>(stg + r * SLD + ch * 8);
        }
        return;
    }
#endif
    if (EPI == 3) {
        const float ts_o = (FMT == 0) ? __ldg(p.tso) : 1.f;
#pragma unroll
        for (int mi = 0; mi < MT; mi++) {
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int row = m0 + wm * WTM + mi * 16 + g + hh * 8;
                float y[NT][2];
#pragma unroll
                for (int nj = 0; nj < NT; nj++) {
                    const int col = n0 + wn * WTN + nj * 8 + cq * 2;
                    const float2 b2 = *reinterpret_cast<const float2*>(p.bias + col);
                    float x0 = acc[mi][nj][hh * 2] * tsv + b2.x, x1 = acc[mi][nj][hh * 2 + 1] * tsv + b2.y;
                    y[nj][0] = x0 * 0.5f * (1.0f + erff(x0 * 0.7071067811865476f));
                    y[nj][1] = x1 * 0.5f * (1.0f + erff(x1 * 0.7071067811865476f));
                }
                constexpr int NB = (FMT == 0) ? 2 : 4;          // 8-col tiles per quantisation block
#pragma unroll
                for (int b = 0; b < NT / NB; b++) {
                    float amax = 0.f;
#pragma unroll
                    for (int j = 0; j < NB; j++) amax = fmaxf(amax, fmaxf(fabsf(y[b * NB + j][0]), fabsf(y[b * NB + j][1])));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                    float inv_mul = 0.f;                       // MX: v = x * inv_mul ; NV: v = x / denom
                    float denom = 1.f;
                    uint32_t sfbyte;
                    if (FMT == 0) {
                        float t = amax * RCP6;
                        t = __fdiv_rn(t, ts_o);
                        t = fminf(t, 448.f);
                        sfbyte = f32_to_e4m3(t);
                        denom = e4m3_to_f32(sfbyte) * ts_o;
                    } else {
                        const uint32_t r = __float_as_uint(amax * RCP6);
                        const uint32_t e = (r >> 23) & 0xffu, fr = (r & 0x7fffffu) != 0 ? 1u : 0u;
                        sfbyte = min(e + fr, 254u);
                        inv_mul = __uint_as_float((254u - sfbyte) << 23);        // 2^(127 - byte); byte 0 -> exponent field 254 = 2^127 (all-zero block: 0 * 2^127 = 0)
                    }
                    if (row < p.M) {
#pragma unroll
                        for (int j = 0; j < NB; j++) {
                            float v0, v1;
                            if (FMT == 0) {
                                v0 = denom > 0.f ? __fdiv_rn(y[b * NB + j][0], denom) : 0.f;
                                v1 = denom > 0.f ? __fdiv_rn(y[b * NB + j][1], denom) : 0.f;
                            } else {
                                v0 = y[b * NB + j][0] * inv_mul;
                                v1 = y[b * NB + j][1] * inv_mul;
                            }
                            const int col = n0 + wn * WTN + (b * NB + j) * 8 + cq * 2;
                            p.Qo[(size_t)row * (p.N >> 1) + (col >> 1)] = (uint8_t)pack2_e2m1(v0, v1);
                        }
                        if (cq == 0) {
                            const int kb = (n0 + wn * WTN + b * NB * 8) / (FMT == 0 ? 16 : 32);
                            p.SFo[sf_blocked_off(row, kb, p.sfo_kt)] = (uint8_t)sfbyte;
                        }
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int mi = 0; mi < MT; mi++) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int row = m0 + wm * WTM + mi * 16 + g + hh * 8;
            if (row >= p.M) continue;
            if (EPI == 2) {
                const int b_i = row / p.Ntok, n_i = row % p.Ntok, pp = n_i - p.prefix;
                const int which = n0 / p.d;
                const bool do_rope = (which < 2) && (pp >= 0);
                const int pc = pp > 0 ? pp : 0;
                constexpr int HALF = NT / 2;
#pragma unroll
                for (int nj = 0; nj < HALF; nj++) {
                    const int blo = blk_of(wn, nj), bhi = blk_of(wn, nj + HALF);
                    const int col_lo = n0 + blo * 8 + cq * 2, col_hi = n0 + bhi * 8 + cq * 2;
                    const float2 bl = *reinterpret_cast<const float2*>(p.bias + col_lo), bh = *reinterpret_cast<const float2*>(p.bias + col_hi);
                    float l0 = acc[mi][nj][hh * 2] * tsv + bl.x, l1 = acc[mi][nj][hh * 2 + 1] * tsv + bl.y;
                    float h0 = acc[mi][nj + HALF][hh * 2] * tsv + bh.x, h1 = acc[mi][nj + HALF][hh * 2 + 1] * tsv + bh.y;
                    if (do_rope) {
                        const int cl = (col_lo % p.d) % 64;
                        const float2 cs_l = *reinterpret_cast<const float2*>(p.cos_t + (size_t)pc * 64 + cl);
                        const float2 sn_l = *reinterpret_cast<const float2*>(p.sin_t + (size_t)pc * 64 + cl);
                        const float2 cs_h = *reinterpret_cast<const float2*>(p.cos_t + (size_t)pc * 64 + cl + 32);
                        const float2 sn_h = *reinterpret_cast<const float2*>(p.sin_t + (size_t)pc * 64 + cl + 32);
                        const float nl0 = l0 * cs_l.x - h0 * sn_l.x, nl1 = l1 * cs_l.y - h1 * sn_l.y;
                        const float nh0 = h0 * cs_h.x + l0 * sn_h.x, nh1 = h1 * cs_h.y + l1 * sn_h.y;
                        l0 = nl0; l1 = nl1; h0 = nh0; h1 = nh1;
                    }
                    const int hd = (col_lo % p.d) / 64, cl2 = (col_lo % p.d) % 64;
                    const size_t o = ((((size_t)which * (p.M / p.Ntok) + b_i) * p.Ntok + n_i) * p.H + hd) * 64 + cl2;
                    __half* Out = reinterpret_cast<__half*>(p.C);
                    *reinterpret_cast<__half2*>(Out + o) = __floats2half2_rn(l0, l1);
                    *reinterpret_cast<__half2*>(Out + o + 32) = __floats2half2_rn(h0, h1);
                }
            } else {
#pragma unroll
                for (int nj = 0; nj < NT; nj++) {
                    const int col = n0 + blk_of(wn, nj) * 8 + cq * 2;
                    const float a0 = acc[mi][nj][hh * 2] * tsv, a1 = acc[mi][nj][hh * 2 + 1] * tsv;
                    const size_t o = (size_t)row * p.N + col;
                    if (EPI == 0) {
                        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(p.C) + o) = __floats2bfloat162_rn(a0, a1);
                    } else if (EPI == 1) {
                        const float2 b2 = *reinterpret_cast<const float2*>(p.bias + col), g2 = *reinterpret_cast<const float2*>(p.gamma + col);
                        const float2 r2 = *reinterpret_cast<const float2*>(p.R + o);
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + o) = make_float2(r2.x + g2.x * (a0 + b2.x), r2.y + g2.y * (a1 + b2.y));
                    } else {   // EPI 4
                        const float2 b2 = *reinterpret_cast<const float2*>(p.bias + col);
                        const float x0 = a0 + b2.x, x1 = a1 + b2.y;
                        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(p.C) + o) =
                            __floats2bfloat162_rn(x0 * 0.5f * (1.0f + erff(x0 * 0.7071067811865476f)), x1 * 0.5f * (1.0f + erff(x1 * 0.7071067811865476f)));
                    }
                }
            }
        }
    }
}
