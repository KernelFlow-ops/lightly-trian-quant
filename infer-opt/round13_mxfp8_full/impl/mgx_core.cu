// Round 13 `mgx`: own MXFP8 block-scaled GEMM for sm_120 (structure derived from round10_lowp_probe/impl/mg8_core.cu, read-only source).
//   C[M,N] = A[M,K] @ W[N,K]^T,  A / W e4m3 bytes K-contiguous, one UE8M0 scale per row per 32 k (cuBLAS blocked layout [R/128, K/128, 32, 4, 4]),
//   mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0, fp32 accumulate (scales applied by the tensor core).
//   cp.async multi-stage pipeline, XOR-swizzled smem + ldmatrix (identical to mg8 / Spark fp8gemm_tile.cuh), fused epilogues:
//     EPI 0  bias + GELU(erf) -> MX8 quantize-out (e4m3 bytes Cq [M,N] + blocked scales Csf; the 1x32 block runs along N = tile-local)   (fc1; fc2's operand)
//     EPI 1  out_f32 = R + gamma * (acc + bias)                                                                                         (proj / fc2)
//     EPI 2  bias + RoPE(fp32) + fp16 [3,B,Ntok,H,64] layout                                                                            (qkv; r12 attention input)
//     EPI 4  plain acc -> bf16 [M,N]                                                                                                     (head ConvT GEMMs)
//   Scales: per 128-k word one 512 B atom per operand is cp.async'ed into a small smem ring; lane (g, c) takes the word of row g + 8*(c&1) (A) / column g (B)
//   (Spark docs/design/fp8gemm.md: the instruction reads A row g from lane 4g, row g+8 from lane 4g+1, B column g from lane 4g with thread-id 0) and the byte for the k32 step is an immediate.
// Compiled to a cubin per config (macros):  BM BN BKB(64|128 bytes of K per stage) ST WM WN MINB EPI [ORDER UNR NFULL STG]
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef BM
#define BM 128
#define BN 128
#define BKB 64
#define ST 3
#define WM 2
#define WN 4
#define MINB 2
#define EPI 1
#endif
#ifndef ORDER
#define ORDER 0
#endif
#ifndef UNR
#define UNR 1
#endif
#ifndef NFULL
#define NFULL 1   // N % BN == 0 (W rows are never predicated; the quantize-out / RoPE epilogues need it anyway)
#endif
#ifndef STG
#define STG 0     // 1: stage fp16 qkv outputs (EPI 2) through smem and write 16 B rows
#endif

typedef __nv_bfloat16 bf16;
static_assert(BKB == 64 || BKB == 128, "BKB");

struct Params {
    const uint8_t* A; const uint8_t* W; const uint8_t* SFA; const uint8_t* SFB;
    const float* bias; const float* gamma; const float* R; void* C;
    const float* sin_t; const float* cos_t; uint8_t* Cq; uint8_t* Csf;
    int M, N, K, Ntok, prefix, d, H, pdl;
};

constexpr int NT = WM * WN * 32;
constexpr int WTM = BM / WM, WTN = BN / WN;
constexpr int MI = WTM / 16, NJ = WTN / 8;
static_assert(WTM % 16 == 0 && NJ % 2 == 0 && MI >= 1, "warp tile");
constexpr int CPR = BKB / 16;                 // 16 B chunks per smem row
constexpr int A_CH = BM * CPR, W_CH = BN * CPR;
constexpr int KPW = 128 / BKB;                // k-tiles per 128-k scale word
constexpr int SSLOTS = (ST - 1 + KPW - 1) / KPW + 1;
constexpr int KS = BKB / 32;                  // k32 steps per k-tile
__host__ __device__ constexpr int nat_of(int B) { return (B % 128 == 0) ? B / 128 : ((128 % B == 0) ? 1 : (B + 112 + 127) / 128); }
constexpr int NATA = nat_of(BM), NATB = nat_of(BN);
constexpr int SA_BYTES = NATA * 512, SB_BYTES = NATB * 512;
constexpr int DATA_BYTES = ST * (BM + BN) * BKB;
constexpr int STG_LD = BN + 8;                // fp16 staging row stride (elements)
constexpr int STQ_LD = BN + 16;               // fp8 staging row stride (bytes)
static_assert(EPI != 0 || (NJ % 4 == 0), "quantize-out needs warp tiles of whole 32-column blocks");
static_assert(!STG || EPI == 2, "STG only for EPI 2");
static_assert(!STG || DATA_BYTES >= BM * STG_LD * 2, "fp16 staging does not fit");
static_assert(EPI != 0 || DATA_BYTES >= BM * STQ_LD, "fp8 staging does not fit");
static_assert(EPI != 2 || (BN % 64 == 0 && WN % (BN / 64 > 0 ? BN / 64 : 1) == 0 && (NJ / 2) * (WN / (BN / 64 > 0 ? BN / 64 : 1)) == 4), "rope warp layout");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
template <int BID>
__device__ __forceinline__ void mma_mx(float* c, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, 0}, %11, {%12, 0};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BID));
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ int swz_unit(int r, int c) {   // swizzled 16 B unit of (row, chunk) in a [rows][BKB bytes] tile
    int u = r * CPR + c;
    return u ^ ((u >> 3) & 7);
}
__device__ __forceinline__ int blk_of(int wn, int nj) {   // which 8-col block of the CTA tile does warp column wn's nj-th fragment cover?
    if constexpr (EPI == 2) {  // keep RoPE partners (block b and b+4 inside a 64-col head) in the same thread
        constexpr int WPH = WN / (BN / 64 > 0 ? BN / 64 : 1);
        int head = wn / WPH, lw = wn % WPH;
        int pi = lw * (NJ / 2) + (nj % (NJ / 2));
        return head * 8 + pi + 4 * (nj / (NJ / 2));
    }
    return wn * NJ + nj;
}
__device__ __forceinline__ int sf_swz(int rr) { return ((rr >> 7) << 9) + ((rr & 31) << 4) + (((rr >> 5) & 3) << 2); }   // byte offset of row rr's word inside the atom ring

// ---- MX8 quantize-out helpers (same arithmetic as mx13.cu / r17)
__device__ __forceinline__ uint32_t rceil_scale_byte(uint32_t amax_bits) {
    float ratio = __fdiv_rn(__uint_as_float(amax_bits), 448.0f);
    uint32_t b = __float_as_uint(ratio);
    uint32_t e = (b >> 23) & 0xffu;
    uint32_t frac = (b & 0x7fffffu) != 0u;
    uint32_t s = e + frac;
    return s > 255u ? 255u : s;
}
__device__ __forceinline__ float inv_scale_of(uint32_t s) { return __uint_as_float((254u - (s == 0u ? 1u : s)) << 23); }
__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return (uint32_t)r;
}

extern "C" __global__ void __launch_bounds__(NT, MINB) gemm_kernel(const Params p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint8_t* As = smem_raw;
    uint8_t* Ws = As + ST * BM * BKB;
    uint8_t* SAs = smem_raw + DATA_BYTES;                  // [SSLOTS][NATA][512]
    uint8_t* SBs = SAs + SSLOTS * SA_BYTES;                // [SSLOTS][NATB][512]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int K = p.K;
    const int KT = K / BKB;
    const int rk = K >> 7;
    float acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    const uint32_t As_u = smem_u32(As), Ws_u = smem_u32(Ws), SAs_u = smem_u32(SAs), SBs_u = smem_u32(SBs);
    // ---- per-thread loader state
    constexpr int A_LD = (A_CH + NT - 1) / NT, W_LD = (W_CH + NT - 1) / NT;
    constexpr bool A_DIV = (A_CH % NT == 0), W_DIV = (W_CH % NT == 0);
    const uint8_t* a_src[A_LD]; uint32_t a_dst[A_LD]; int a_sz[A_LD];
    const uint8_t* w_src[W_LD]; uint32_t w_dst[W_LD]; int w_sz[W_LD];
#pragma unroll
    for (int i = 0; i < A_LD; i++) {
        int u = tid + i * NT;
        if (!A_DIV && u >= A_CH) u = 0;
        int r = u / CPR, c = u % CPR;
        int gr = m0 + r;
        bool ok = gr < p.M;
        a_src[i] = p.A + (size_t)(ok ? gr : 0) * K + c * 16;
        a_dst[i] = As_u + (uint32_t)(swz_unit(r, c) * 16);
        a_sz[i] = ok ? 16 : 0;
    }
#pragma unroll
    for (int i = 0; i < W_LD; i++) {
        int u = tid + i * NT;
        if (!W_DIV && u >= W_CH) u = 0;
        int r = u / CPR, c = u % CPR;
        bool ok = NFULL ? true : ((n0 + r) < p.N);
        w_src[i] = p.W + (size_t)(ok ? (n0 + r) : 0) * K + c * 16;
        w_dst[i] = Ws_u + (uint32_t)(swz_unit(r, c) * 16);
        w_sz[i] = ok ? 16 : 0;
    }
    // scale atoms: thread t < (NATA+NATB)*32 copies 16 B chunk (t % 32) of atom (t / 32)
    const int nat_m = (p.M + 127) >> 7, nat_n = p.N >> 7;
    constexpr int SF_CH = (NATA + NATB) * 32;
    constexpr int SF_LD = (SF_CH + NT - 1) / NT;
    const uint8_t* sf_src[SF_LD];
    uint32_t sf_dst[SF_LD];
#pragma unroll
    for (int i = 0; i < SF_LD; i++) {
        int c = tid + i * NT;
        if (c < SF_CH) {
            int ai = c >> 5, ch = c & 31;
            if (ai < NATA) {
                int at = min((m0 >> 7) + ai, nat_m - 1);
                sf_src[i] = p.SFA + (size_t)at * rk * 512 + ch * 16;
                sf_dst[i] = SAs_u + ai * 512 + ch * 16;
            } else {
                int bi = ai - NATA;
                int at = min((n0 >> 7) + bi, nat_n - 1);
                sf_src[i] = p.SFB + (size_t)at * rk * 512 + ch * 16;
                sf_dst[i] = SBs_u + bi * 512 + ch * 16;
            }
        }
    }
    // ---- ldmatrix offsets (bytes inside a stage), hoisted
    uint32_t a_ld[KS][MI], w_ld[KS][NJ / 2];
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            int r = wm * WTM + mi * 16 + (lane & 15), c = ks * 2 + (lane >> 4);
            a_ld[ks][mi] = swz_unit(r, c) * 16;
        }
#pragma unroll
        for (int j2 = 0; j2 < NJ / 2; j2++) {
            int blk = blk_of(wn, 2 * j2 + (lane >> 4));
            int r = blk * 8 + (lane & 7), c = ks * 2 + ((lane >> 3) & 1);
            w_ld[ks][j2] = swz_unit(r, c) * 16;
        }
    }
    // ---- scale word offsets (bytes inside an atom ring slot)
    uint32_t sa_off[MI], sb_off[NJ];
#pragma unroll
    for (int mi = 0; mi < MI; mi++) sa_off[mi] = sf_swz((m0 & 127) + wm * WTM + mi * 16 + (lane >> 2) + 8 * (lane & 1));
#pragma unroll
    for (int nj = 0; nj < NJ; nj++) sb_off[nj] = sf_swz((n0 & 127) + blk_of(wn, nj) * 8 + (lane >> 2));

    // W operand (static weights, their scale atoms) and A operand (predecessor's output, its scale atoms) are issued separately so that the prologue can
    // prefetch the weights before griddepcontrol.wait (PDL)
    auto load_w = [&](int st, int kt) {
#pragma unroll
        for (int i = 0; i < W_LD; i++)
            if (W_DIV || tid + i * NT < W_CH)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(w_dst[i] + st * (BN * BKB)), "l"(w_src[i] + (size_t)kt * BKB), "r"(w_sz[i]));
        if (KPW == 1 || (kt % KPW) == 0) {
            const int wd = kt / KPW, slot = wd % SSLOTS;
#pragma unroll
            for (int i = 0; i < SF_LD; i++)
                if (tid + i * NT < SF_CH && ((tid + i * NT) >> 5) >= NATA)
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sf_dst[i] + slot * SB_BYTES), "l"(sf_src[i] + (size_t)wd * 512));
        }
    };
    auto load_a = [&](int st, int kt) {
#pragma unroll
        for (int i = 0; i < A_LD; i++)
            if (A_DIV || tid + i * NT < A_CH)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a_dst[i] + st * (BM * BKB)), "l"(a_src[i] + (size_t)kt * BKB), "r"(a_sz[i]));
        if (KPW == 1 || (kt % KPW) == 0) {
            const int wd = kt / KPW, slot = wd % SSLOTS;
#pragma unroll
            for (int i = 0; i < SF_LD; i++)
                if (tid + i * NT < SF_CH && ((tid + i * NT) >> 5) < NATA)
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sf_dst[i] + slot * SA_BYTES), "l"(sf_src[i] + (size_t)wd * 512));
        }
    };
    auto load_tile = [&](int st, int kt) { load_w(st, kt); load_a(st, kt); };

    // ---- prologue: weights first (independent of the predecessor kernel), then wait for it (PDL), then activations
#pragma unroll
    for (int s = 0; s < ST - 1; s++)
        if (s < KT) load_w(s, s);
    pdl_wait();
#pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < KT) load_a(s, s);
        cp_commit();
    }
    pdl_trigger();

#if UNR == 2
#pragma unroll 2
#elif UNR == 4
#pragma unroll 4
#else
#pragma unroll 1
#endif
    for (int kt = 0; kt < KT; kt++) {
        if (ORDER == 0) {
            cp_wait<ST - 2>();
            __syncthreads();
            int nk = kt + ST - 1;
            if (nk < KT) load_tile(nk % ST, nk);
            cp_commit();
        } else {
            __syncthreads();
            int nk = kt + ST - 1;
            if (nk < KT) load_tile(nk % ST, nk);
            cp_commit();
            cp_wait<ST - 1>();
            __syncthreads();
        }
        const int st = kt % ST;
        const uint32_t a_base = As_u + st * BM * BKB, w_base = Ws_u + st * BN * BKB;
        // scale words of this k-tile
        const int slot = (kt / KPW) % SSLOTS;
        const int shift = (KPW == 2) ? ((kt & 1) << 4) : 0;
        const uint8_t* sa_p = SAs + slot * SA_BYTES;
        const uint8_t* sb_p = SBs + slot * SB_BYTES;
        uint32_t sa[MI], sb[NJ];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) sa[mi] = (*reinterpret_cast<const uint32_t*>(sa_p + sa_off[mi])) >> shift;
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) sb[nj] = (*reinterpret_cast<const uint32_t*>(sb_p + sb_off[nj])) >> shift;
        uint32_t af[KS][MI][4], bfr[KS][NJ][2];
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++) ldsm_x4(a_base + a_ld[ks][mi], af[ks][mi][0], af[ks][mi][1], af[ks][mi][2], af[ks][mi][3]);
#pragma unroll
            for (int j2 = 0; j2 < NJ / 2; j2++) ldsm_x4(w_base + w_ld[ks][j2], bfr[ks][2 * j2][0], bfr[ks][2 * j2][1], bfr[ks][2 * j2 + 1][0], bfr[ks][2 * j2 + 1][1]);
        }
#define MMA_KS(K_)     _Pragma("unroll") for (int mi = 0; mi < MI; mi++)     _Pragma("unroll") for (int nj = 0; nj < NJ; nj++) mma_mx<K_>(acc[mi][nj], af[K_][mi], bfr[K_][nj][0], bfr[K_][nj][1], sa[mi], sb[nj]);
        MMA_KS(0)
        MMA_KS(1)
#if BKB == 128
        MMA_KS(2)
        MMA_KS(3)
#endif
#undef MMA_KS
    }
    cp_wait<0>();

    // ---- epilogues
    const bool full_m = (m0 + BM) <= p.M;
    if constexpr (EPI == 0) {
        // bias + GELU -> MX8: one 1x32 block = 4 consecutive nj of one row, held by the 4 lanes of a quad
        uint8_t* stq = smem_raw;
        __syncthreads();
        const int rkq = p.N >> 7;
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int rl = wm * WTM + mi * 16 + (lane >> 2) + hh * 8;
                const int row = m0 + rl;
#pragma unroll
                for (int q = 0; q < NJ / 4; q++) {
                    float x[8];
                    uint32_t amax = 0u;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int nj = q * 4 + j;
                        const int col = n0 + blk_of(wn, nj) * 8 + (lane & 3) * 2;
                        float2 b2 = *reinterpret_cast<const float2*>(p.bias + col);
                        float x0 = acc[mi][nj][hh * 2] + b2.x, x1 = acc[mi][nj][hh * 2 + 1] + b2.y;
                        x0 = x0 * 0.5f * (1.0f + erff(x0 * 0.7071067811865476f));
                        x1 = x1 * 0.5f * (1.0f + erff(x1 * 0.7071067811865476f));
                        x[2 * j] = x0; x[2 * j + 1] = x1;
                        amax = max(amax, max(__float_as_uint(fabsf(x0)), __float_as_uint(fabsf(x1))));
                    }
                    amax = max(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                    amax = max(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                    const uint32_t s = rceil_scale_byte(amax);
                    const float inv = inv_scale_of(s);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int nj = q * 4 + j;
                        uint32_t v2 = (s >= 255u) ? 0x7f7fu : cvt_e4m3x2(x[2 * j] * inv, x[2 * j + 1] * inv);
                        *reinterpret_cast<uint16_t*>(stq + rl * STQ_LD + blk_of(wn, nj) * 8 + (lane & 3) * 2) = (uint16_t)v2;
                    }
                    if ((lane & 3) == 0 && row < p.M) {
                        const int kb = (n0 + blk_of(wn, q * 4) * 8) >> 5;
                        p.Csf[((size_t)((row >> 7) * rkq + (kb >> 2)) << 9) + ((row & 31) << 4) + (((row >> 5) & 3) << 2) + (kb & 3)] = (uint8_t)s;
                    }
                }
            }
        }
        __syncthreads();
        constexpr int CH = BN / 16;
        for (int idx = tid; idx < BM * CH; idx += NT) {
            const int r = idx / CH, ch = idx % CH;
            const int grow = m0 + r;
            if (grow >= p.M) continue;
            *reinterpret_cast<uint4*>(p.Cq + (size_t)grow * p.N + n0 + ch * 16) = *reinterpret_cast<const uint4*>(stq + r * STQ_LD + ch * 16);
        }
        return;
    }
    __half* stg = reinterpret_cast<__half*>(smem_raw);
    if (STG) __syncthreads();
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int rl = wm * WTM + mi * 16 + (lane >> 2) + hh * 8;
            const int row = m0 + rl;
            if (!full_m && row >= p.M) continue;
            if (EPI == 1) {
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    int col = n0 + blk_of(wn, nj) * 8 + (lane & 3) * 2;
                    if (!NFULL && col >= p.N) continue;
                    float2 b2 = *reinterpret_cast<const float2*>(p.bias + col);
                    float2 g2 = *reinterpret_cast<const float2*>(p.gamma + col);
                    float x0 = (acc[mi][nj][hh * 2] + b2.x) * g2.x;
                    float x1 = (acc[mi][nj][hh * 2 + 1] + b2.y) * g2.y;
                    size_t o = (size_t)row * p.N + col;
                    float2 r2 = *reinterpret_cast<const float2*>(p.R + o);
                    *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + o) = make_float2(r2.x + x0, r2.y + x1);
                }
            } else if (EPI == 4) {
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    int col = n0 + blk_of(wn, nj) * 8 + (lane & 3) * 2;
                    if (!NFULL && col >= p.N) continue;
                    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<bf16*>(p.C) + (size_t)row * p.N + col) =
                        __floats2bfloat162_rn(acc[mi][nj][hh * 2], acc[mi][nj][hh * 2 + 1]);
                }
            } else if (EPI == 2) {  // qkv + rope + bnhd layout, fp16 out
                const int b_i = row / p.Ntok, n_i = row % p.Ntok, pp = n_i - p.prefix;
                const int which = n0 / p.d;
                const bool do_rope = (which < 2) && (pp >= 0);
                const int pc = pp > 0 ? pp : 0;
                constexpr int HALF = NJ / 2;
#pragma unroll
                for (int nj = 0; nj < HALF; nj++) {
                    int blo = blk_of(wn, nj), bhi = blk_of(wn, nj + HALF);
                    int col_lo = n0 + blo * 8 + (lane & 3) * 2, col_hi = n0 + bhi * 8 + (lane & 3) * 2;
                    float2 bl = *reinterpret_cast<const float2*>(p.bias + col_lo);
                    float2 bh = *reinterpret_cast<const float2*>(p.bias + col_hi);
                    float l0 = acc[mi][nj][hh * 2] + bl.x, l1 = acc[mi][nj][hh * 2 + 1] + bl.y;
                    float h0 = acc[mi][nj + HALF][hh * 2] + bh.x, h1 = acc[mi][nj + HALF][hh * 2 + 1] + bh.y;
                    if (do_rope) {
                        int cl = (col_lo % p.d) % 64;  // 0..31 (even)
                        float2 cs_l = *reinterpret_cast<const float2*>(p.cos_t + (size_t)pc * 64 + cl);
                        float2 sn_l = *reinterpret_cast<const float2*>(p.sin_t + (size_t)pc * 64 + cl);
                        float2 cs_h = *reinterpret_cast<const float2*>(p.cos_t + (size_t)pc * 64 + cl + 32);
                        float2 sn_h = *reinterpret_cast<const float2*>(p.sin_t + (size_t)pc * 64 + cl + 32);
                        float nl0 = l0 * cs_l.x - h0 * sn_l.x, nl1 = l1 * cs_l.y - h1 * sn_l.y;
                        float nh0 = h0 * cs_h.x + l0 * sn_h.x, nh1 = h1 * cs_h.y + l1 * sn_h.y;
                        l0 = nl0; l1 = nl1; h0 = nh0; h1 = nh1;
                    }
                    if (STG) {
                        *reinterpret_cast<__half2*>(stg + rl * STG_LD + blo * 8 + (lane & 3) * 2) = __floats2half2_rn(l0, l1);
                        *reinterpret_cast<__half2*>(stg + rl * STG_LD + bhi * 8 + (lane & 3) * 2) = __floats2half2_rn(h0, h1);
                    } else {
                        int hd = (col_lo % p.d) / 64, cl2 = (col_lo % p.d) % 64;
                        size_t o = ((((size_t)which * (p.M / p.Ntok) + b_i) * p.Ntok + n_i) * p.H + hd) * 64 + cl2;
                        __half* Out = reinterpret_cast<__half*>(p.C);
                        *reinterpret_cast<__half2*>(Out + o) = __floats2half2_rn(l0, l1);
                        *reinterpret_cast<__half2*>(Out + o + 32) = __floats2half2_rn(h0, h1);
                    }
                }
            }
        }
    }
    if (STG) {
        __syncthreads();
        constexpr int CH = BN / 8;
        for (int idx = tid; idx < BM * CH; idx += NT) {
            const int r = idx / CH, ch = idx % CH;
            const int grow = m0 + r, gcol = n0 + ch * 8;
            if (grow >= p.M || (!NFULL && gcol >= p.N)) continue;
            uint4 v = *reinterpret_cast<const uint4*>(stg + r * STG_LD + ch * 8);
            const int b_i = grow / p.Ntok, n_i = grow % p.Ntok;
            const int which = n0 / p.d;
            const int hd = (n0 % p.d) / 64 + ch / 8;
            size_t o = ((((size_t)which * (p.M / p.Ntok) + b_i) * p.Ntok + n_i) * p.H + hd) * 64 + (ch % 8) * 8;
            *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(p.C) + o) = v;
        }
    }
}
