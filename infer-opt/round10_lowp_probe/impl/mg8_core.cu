// Round 10 PROTOTYPE `mg_i8` (derived from the `mg_f16acc` prototype, itself a copy of round5_gemm_sm120/impl/mg_core.cu):
//   int8 W8A8 GEMM, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32, int32 accumulators (exact), A [M,K] / W [N,K] int8 K-contiguous.
//   BK counts bf16-equivalent elements (bytes / 2): BK = 32 -> a k-tile of 64 int8 elements (2 MMA k32 steps), BK = 64 -> 128.  cp.async / ldmatrix / swizzle code is
//   byte-identical to the 16-bit kernels (an m16n8k32 s8 A/B fragment has the same register layout as m16n8k16 f16).
//   Dequant in every epilogue: acc * sx[row] * sw[col]   (sx = per-token activation scale [M] -> Params.stats_in, sw = per-output-channel weight scale [N] -> Params.cs),
//   then the R5 epilogues unchanged: bias (+GELU) / bias + RoPE + [3,B,N,H,64] layout / gamma + fp32 residual / plain.  No split-K (SPLITK = 1).
// Original header of the f16acc prototype follows.
// Round 10 PROTOTYPE `mg_f16acc`: copy of round5_gemm_sm120/impl/mg_core.cu (read-only source) with
//   * A / W operands fp16 (the cp.async / ldmatrix / swizzle code is type agnostic),
//   * the MMA replaced by mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 (full rate on the GeForce, fp32-accumulate HMMA is half rate),
//     fp16 accumulators are zeroed and, every PKT k-tiles (default 1), added into the fp32 accumulator registers,
//   * Params gets `oscale` (power-of-two un-scaling of the pre-scaled fp16 operands, applied to acc in every epilogue),
//   * EPI 0 / 2 can write fp16 outputs (-DOUTF16=1) so that the next GEMM / attention consumes fp16 directly.
//   Epilogues (RoPE + [3,B,N,H,64] layout, GELU, gamma + fp32 residual, split-K) are otherwise unchanged; LNF / STATS paths are not used here.
// Original header follows.
// Round 5: hand-written sm_120 bf16 GEMM (mma.sync m16n8k16 + cp.async multi-stage + ldmatrix, XOR-swizzled smem)
// with fused epilogues, split-K (last-CTA reduce, deterministic), LayerNorm fusion and PDL.
// Compiled to a cubin per config (macros; see mgb.py):  BM BN BK ST WM WN MINB EPI RF32 PDL_TRIG SPLITK LNF STATS
//   C[M,N] = A[M,K] @ W[N,K]^T   (A, W bf16 row-major, K contiguous)
// EPI 0: GELU(erf(acc + bias)) -> bf16                          (fc1)
//     1: out = R + gamma*(acc + bias)   (fp32 out when RF32)    (proj / fc2)   [STATS=1: also bf16 copy + row-stat partials]
//     2: bias + RoPE(fp32) + [3,B,Ntok,H,64] bf16 layout        (qkv)          [BN multiple of 64, heads whole]
//     3: fp32 [SPLITK,M,N] plain partials                       (tests)
// LNF=1 (EPI 0/2): A is the *raw* bf16 residual x (no LayerNorm applied); LN(x)@W^T is obtained algebraically:
//     out_pre = rstd[m] * (acc[m,n] - mean[m] * cs[n]) + bias32[n]    with W' = ln_w * W folded on the host,
//     cs[n] = sum_k W'[n,k], bias32 = bias + ln_b @ W^T.  mean/rstd come from row-stat partials (sum, M2) written by the
//     producer epilogue (STATS=1) and are combined here (Chan) while the first cp.async stages are in flight.
// PDL: weights are prefetched before griddepcontrol.wait; the CTA triggers launch_dependents right after its prologue.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef BM
#define BM 64
#define BN 64
#define BK 64
#define ST 2
#define WM 2
#define WN 2
#define MINB 3
#define EPI 0
#define RF32 1
#define PDL_TRIG 1
#endif
#ifndef SPLITK
#define SPLITK 1
#endif
#ifndef LNF
#define LNF 0
#endif
#ifndef ORDER
#define ORDER 0   // 0: wait -> sync -> issue next -> compute ; 1: sync -> issue next -> wait -> sync -> compute (TileLang-like)
#endif
#ifndef WPRED
#define WPRED 1   // predicate W-row loads on N (needed when N % BN != 0)
#endif
#ifndef UNR
#define UNR 1
#endif
#ifndef NFULL
#define NFULL 0   // 1: N % BN == 0 guaranteed (no column guards)
#endif
#ifndef STG
#define STG 0     // 1: stage bf16 outputs (EPI 0/2) through smem and write 16B-coalesced rows
#endif
#ifndef PDLWAIT
#define PDLWAIT 1
#endif
#ifndef FRAGALL
#define FRAGALL 0
#endif
#ifndef STATS
#define STATS 0
#endif
#ifndef F16ACC
#define F16ACC 0      // 1: fp16-accumulate MMA + periodic promotion to fp32, 0: original bf16 MMA (sanity check of the copy)
#endif
#ifndef OUTF16
#define OUTF16 0      // EPI 0/2 output dtype: 1 = fp16, 0 = bf16
#endif
#ifndef PKT
#define PKT 1             // k-tiles accumulated in fp16 between promotions to fp32 (1 = every k-tile = BK k-elements; 2 = every other k-tile)
#endif

typedef __nv_bfloat16 bf16;
static_assert(BK == 32 || BK == 64, "BK");
static_assert(!LNF || EPI == 0 || EPI == 2, "LNF only for consumers");
static_assert(!STATS || EPI == 1, "STATS only for producers");

struct Params {
    const int8_t* A; const int8_t* W; const void* bias; const float* gamma; const void* R; void* C;
    const float* sin_t; const float* cos_t;
    float* ws; int* counters;
    const float* cs; const float* stats_in; float* stats_out; bf16* Cb;
    int M, N, K, Ntok, prefix, d, H, pdl;
    int np_in, pcols_in;
    float eps;
    float oscale;                  // round10: acc *= oscale in every epilogue (power of two)
    const char* pf0; const char* pf1; int pf0_bytes, pf1_bytes;   // optional L2 prefetch regions (next kernels' weights)
};

constexpr int NT = WM * WN * 32;
constexpr int BKE = BK * 2;   // int8 elements per k-tile
static_assert(SPLITK == 1, "mg8: no split-K");
#define DEQ(a_, row_, col_) ((float)(a_) * __ldg(p.stats_in + (row_)) * __ldg(p.cs + (col_)))
constexpr int WTM = BM / WM, WTN = BN / WN;
constexpr int MI = WTM / 16, NJ = WTN / 8;
static_assert(NJ % 2 == 0 && MI >= 1, "warp tile");
constexpr int A_CH = BM * BK / 8, W_CH = BN * BK / 8;  // 16B chunks per tile

constexpr int CPR = BK / 8;  // chunks per row
constexpr int STG_LD = BN + 8;  // staging row stride (elements)
static_assert(!STG || (EPI == 0 || EPI == 2), "STG only for bf16-output epilogues");
static_assert(!STG || ST * (BM + BN) * BK * 2 >= BM * STG_LD * 2, "staging buffer does not fit in the pipeline smem");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {   // = m16n8k32 s8 (32 bytes of k per step, like k16 for 16-bit)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816h(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void promote_h(float* c, uint32_t* h) {   // c[4] += fp16 accumulator pair (rows g, g+8), then zero it
    float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&h[0]));
    float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&h[1]));
    c[0] += f0.x; c[1] += f0.y; c[2] += f1.x; c[3] += f1.y;
    h[0] = 0u; h[1] = 0u;
}
#if OUTF16
typedef __half2 out2_t;
__device__ __forceinline__ out2_t pk2(float a, float b) { return __floats2half2_rn(a, b); }
#else
typedef __nv_bfloat162 out2_t;
__device__ __forceinline__ out2_t pk2(float a, float b) { return __floats2bfloat162_rn(a, b); }
#endif
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

// swizzled 16B-unit index of (row, chunk) inside a [rows][BK] bf16 tile
__device__ __forceinline__ int swz_unit(int r, int c) {
    int u = r * CPR + c;
    return u ^ ((u >> 3) & 7);
}

static_assert(EPI != 2 || (BN % 64 == 0 && WN % (BN / 64 > 0 ? BN / 64 : 1) == 0 &&
                           (NJ / 2) * (WN / (BN / 64 > 0 ? BN / 64 : 1)) == 4), "rope warp layout");
// which 8-col block of the CTA tile does warp column wn's nj-th fragment cover?
__device__ __forceinline__ int blk_of(int wn, int nj) {
    if constexpr (EPI == 2) {  // keep RoPE partners (block b and b+4 inside a 64-col head) in the same thread
        constexpr int WPH = WN / (BN / 64 > 0 ? BN / 64 : 1);   // warps per head
        int head = wn / WPH, lw = wn % WPH;
        int pi = lw * (NJ / 2) + (nj % (NJ / 2));
        return head * 8 + pi + 4 * (nj / (NJ / 2));
    }
    return wn * NJ + nj;
}

extern "C" __global__ void __launch_bounds__(NT, MINB) gemm_kernel(const Params p) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    bf16* As = reinterpret_cast<bf16*>(smem_raw);
    bf16* Ws = As + ST * BM * BK;
    __shared__ float2 s_st[LNF ? BM : 1];
    __shared__ float s_red[STATS ? 2 * BM * WN : 1];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
#ifdef KDIM
    static_assert(KDIM % BK == 0, "KDIM % BK");
    const int K = KDIM;     // compile-time K: loop bounds / predicates fold away
#else
    const int K = p.K;
#endif
    const int KT_ALL = K / BKE;
    int kt0 = 0, KT = KT_ALL;
    if (SPLITK > 1) { kt0 = (KT_ALL * (int)blockIdx.z) / SPLITK; KT = (KT_ALL * ((int)blockIdx.z + 1)) / SPLITK - kt0; }
    int acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;

#if F16ACC
    uint32_t ah[MI][NJ][2];     // fp16 accumulators (promoted into acc every PKT k-tiles, then zeroed)
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) { ah[i][j][0] = 0u; ah[i][j][1] = 0u; }
#endif
    const uint32_t As_u = smem_u32(As), Ws_u = smem_u32(Ws);
    // per-thread loader state (hoisted out of the k loop)
    constexpr int A_LD = (A_CH + NT - 1) / NT, W_LD = (W_CH + NT - 1) / NT;
    constexpr bool A_DIV = (A_CH % NT == 0), W_DIV = (W_CH % NT == 0);
    const int8_t* a_src[A_LD]; uint32_t a_dst[A_LD]; int a_sz[A_LD];
    const int8_t* w_src[W_LD]; uint32_t w_dst[W_LD]; int w_sz[W_LD];
#pragma unroll
    for (int i = 0; i < A_LD; i++) {
        int u = tid + i * NT;
        if (!A_DIV && u >= A_CH) u = 0;
        int r = u / CPR, c = u % CPR;
        int gr = m0 + r;
        bool ok = gr < p.M;
        a_src[i] = p.A + (size_t)(ok ? gr : 0) * K + (size_t)kt0 * BKE + c * 16;
        a_dst[i] = As_u + (uint32_t)(swz_unit(r, c) * 16);
        a_sz[i] = ok ? 16 : 0;
    }
#pragma unroll
    for (int i = 0; i < W_LD; i++) {
        int u = tid + i * NT;
        if (!W_DIV && u >= W_CH) u = 0;
        int r = u / CPR, c = u % CPR;
        bool ok = WPRED ? ((n0 + r) < p.N) : true;
        w_src[i] = p.W + (size_t)(ok ? (n0 + r) : 0) * K + (size_t)kt0 * BKE + c * 16;
        w_dst[i] = Ws_u + (uint32_t)(swz_unit(r, c) * 16);
        w_sz[i] = ok ? 16 : 0;
    }
    // ldmatrix offsets (bytes, relative to the stage base), hoisted
    uint32_t a_ld[BK / 16][MI], w_ld[BK / 16][NJ / 2];
#pragma unroll
    for (int ks = 0; ks < BK / 16; ks++) {
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
    auto load_A = [&](int st, int kt) {
#pragma unroll
        for (int i = 0; i < A_LD; i++)
            if (A_DIV || tid + i * NT < A_CH)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a_dst[i] + st * (BM * BK * 2)),
                             "l"(a_src[i] + (size_t)kt * BKE), "r"(a_sz[i]));
    };
    auto load_W = [&](int st, int kt) {
#pragma unroll
        for (int i = 0; i < W_LD; i++)
            if (W_DIV || tid + i * NT < W_CH)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(w_dst[i] + st * (BN * BK * 2)),
                             "l"(w_src[i] + (size_t)kt * BKE), "r"(w_sz[i]));
    };

    // ---- L2 prefetch of weights needed by later kernels (independent of the predecessor -> before griddepcontrol.wait)
    if (p.pf0_bytes | p.pf1_bytes) {
        const int cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        const int ncta = gridDim.x * gridDim.y * gridDim.z;
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const char* base = r ? p.pf1 : p.pf0;
            const int nlines = (r ? p.pf1_bytes : p.pf0_bytes) >> 7;
            for (int i = cta * NT + tid; i < nlines; i += ncta * NT)
                asm volatile("prefetch.global.L2 [%0];\n" ::"l"(base + (size_t)i * 128));
        }
    }
    // ---- prologue: weights first (independent of the predecessor kernel), then wait, then activations
#pragma unroll
    for (int s = 0; s < ST - 1; s++) if (s < KT) load_W(s, s);
    if (PDLWAIT) pdl_wait();
#pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < KT) load_A(s, s);
        cp_commit();
    }
    if (PDL_TRIG && PDLWAIT) pdl_trigger();

    if (LNF) {  // per-row mean / rstd from the producer's partials (Chan combine) while the cp.async stages fly
        const float* sp = p.stats_in;
        for (int rr = tid; rr < BM; rr += NT) {
            const int row = m0 + rr;
            float mean = 0.f, rstd = 0.f;
            if (row < p.M) {
                const float2* ps = reinterpret_cast<const float2*>(sp) + (size_t)row * p.np_in;
                float2 v[32];
                float tot = 0.f;
#pragma unroll
                for (int i = 0; i < 32; i++) if (i < p.np_in) { v[i] = ps[i]; tot += v[i].x; }
                mean = tot / (float)K;
                float m2 = 0.f;
#pragma unroll
                for (int i = 0; i < 32; i++) if (i < p.np_in) {
                    float c = (float)min(p.pcols_in, K - i * p.pcols_in);
                    float dm = v[i].x / c - mean;
                    m2 += v[i].y + c * dm * dm;
                }
                rstd = rsqrtf(m2 / (float)K + p.eps);
            }
            s_st[rr] = make_float2(mean, rstd);
        }
    }

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
            if (nk < KT) { load_W(nk % ST, nk); load_A(nk % ST, nk); }
            cp_commit();
        } else {
            __syncthreads();
            int nk = kt + ST - 1;
            if (nk < KT) { load_W(nk % ST, nk); load_A(nk % ST, nk); }
            cp_commit();
            cp_wait<ST - 1>();
            __syncthreads();
        }
        const int st = kt % ST;
        const uint32_t a_base = As_u + st * BM * BK * 2, w_base = Ws_u + st * BN * BK * 2;
#if FRAGALL
        uint32_t af[BK / 16][MI][4], bfr[BK / 16][NJ][2];
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                ldsm_x4(a_base + a_ld[ks][mi], af[ks][mi][0], af[ks][mi][1], af[ks][mi][2], af[ks][mi][3]);
            }
#pragma unroll
            for (int j2 = 0; j2 < NJ / 2; j2++) {
                ldsm_x4(w_base + w_ld[ks][j2], bfr[ks][2 * j2][0], bfr[ks][2 * j2][1], bfr[ks][2 * j2 + 1][0], bfr[ks][2 * j2 + 1][1]);
            }
        }
#if F16ACC
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816h(ah[mi][nj], af[ks][mi], bfr[ks][nj][0], bfr[ks][nj][1]);
        }
        if (PKT == 1 || ((kt + 1) % PKT) == 0) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) promote_h(acc[mi][nj], ah[mi][nj]);
        }
#else
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++)
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816(acc[mi][nj], af[ks][mi], bfr[ks][nj][0], bfr[ks][nj][1]);
#endif
#else
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            uint32_t af[MI][4], bfr[NJ][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                ldsm_x4(a_base + a_ld[ks][mi], af[mi][0], af[mi][1], af[mi][2], af[mi][3]);
            }
#pragma unroll
            for (int j2 = 0; j2 < NJ / 2; j2++) {
                ldsm_x4(w_base + w_ld[ks][j2], bfr[2 * j2][0], bfr[2 * j2][1], bfr[2 * j2 + 1][0], bfr[2 * j2 + 1][1]);
            }
#if F16ACC
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816h(ah[mi][nj], af[mi], bfr[nj][0], bfr[nj][1]);
#else
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816(acc[mi][nj], af[mi], bfr[nj][0], bfr[nj][1]);
#endif
        }
#if F16ACC
        if (PKT == 1 || ((kt + 1) % PKT) == 0) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) promote_h(acc[mi][nj], ah[mi][nj]);
        }
#endif
#endif
    }
    cp_wait<0>();
#if F16ACC
    if (PKT > 1) {      // flush a partial promotion group (no-op when KT % PKT == 0: ah is zero)
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) promote_h(acc[mi][nj], ah[mi][nj]);
    }
#endif

    // ---- split-K: write fp32 partial, last-arriving CTA of the tile sums (fixed order -> deterministic) and runs the epilogue
    if (SPLITK > 1 && EPI != 3) {
        __shared__ int s_last;
        const int tile = blockIdx.y * gridDim.x + blockIdx.x;
        float* wsl = p.ws + (size_t)tile * SPLITK * (BM * BN);
        float* mine = wsl + (size_t)blockIdx.z * (BM * BN);
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int nj = 0; nj < NJ; nj++)
                __stcg(reinterpret_cast<float4*>(mine + ((mi * NJ + nj) * NT + tid) * 4),
                       make_float4(acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3]));
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            int old = atomicAdd(p.counters + tile, 1);
            s_last = (old == SPLITK - 1);
        }
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        float tot[MI][NJ][4];
#pragma unroll
        for (int sidx = 0; sidx < SPLITK; sidx++) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    float4 v;
                    if (sidx == (int)blockIdx.z) v = make_float4(acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3]);
                    else v = __ldcg(reinterpret_cast<const float4*>(wsl + (size_t)sidx * (BM * BN) + ((mi * NJ + nj) * NT + tid) * 4));
                    if (sidx == 0) { tot[mi][nj][0] = v.x; tot[mi][nj][1] = v.y; tot[mi][nj][2] = v.z; tot[mi][nj][3] = v.w; }
                    else { tot[mi][nj][0] += v.x; tot[mi][nj][1] += v.y; tot[mi][nj][2] += v.z; tot[mi][nj][3] += v.w; }
                }
        }
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[mi][nj][e] = tot[mi][nj][e];
        if (tid == 0) p.counters[tile] = 0;
    }

    // ---- epilogues
    if (EPI == 1 && STATS) {
        // out = R + gamma*(acc+bias); write fp32 x, bf16 copy, and per-row (sum, M2) partial over this CTA's columns
        float xv[MI][NJ][4];
        float rsum[MI][2];
        const int ncols = min(BN, p.N - n0);
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int row = m0 + wm * WTM + mi * 16 + (lane >> 2) + hh * 8;
                float s = 0.f;
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    int col = n0 + blk_of(wn, nj) * 8 + (lane & 3) * 2;
                    float v0 = 0.f, v1 = 0.f;
                    if (col < p.N && row < p.M) {
                        __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(p.bias) + col);
                        float2 g2 = *reinterpret_cast<const float2*>(p.gamma + col);
                        size_t o = (size_t)row * p.N + col;
                        float2 r2 = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(p.R) + o);
                        v0 = r2.x + (acc[mi][nj][hh * 2] + __bfloat162float(b2.x)) * g2.x;
                        v1 = r2.y + (acc[mi][nj][hh * 2 + 1] + __bfloat162float(b2.y)) * g2.y;
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + o) = make_float2(v0, v1);
                        *reinterpret_cast<__nv_bfloat162*>(p.Cb + o) = __floats2bfloat162_rn(v0, v1);
                    }
                    xv[mi][nj][hh * 2] = v0; xv[mi][nj][hh * 2 + 1] = v1;
                    s += v0 + v1;
                }
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                rsum[mi][hh] = s;
                if ((lane & 3) == 0) s_red[(wm * WTM + mi * 16 + (lane >> 2) + hh * 8) * WN + wn] = s;
            }
        }
        __syncthreads();
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int rl = wm * WTM + mi * 16 + (lane >> 2) + hh * 8;
                float tot = 0.f;
#pragma unroll
                for (int w = 0; w < WN; w++) tot += s_red[rl * WN + w];
                rsum[mi][hh] = tot;
                const float mean_t = tot / (float)ncols;
                float q = 0.f;
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    int col = n0 + blk_of(wn, nj) * 8 + (lane & 3) * 2;
                    if (col < p.N) {
                        float a0 = xv[mi][nj][hh * 2] - mean_t, a1 = xv[mi][nj][hh * 2 + 1] - mean_t;
                        q += a0 * a0 + a1 * a1;
                    }
                }
                q += __shfl_xor_sync(0xffffffffu, q, 1);
                q += __shfl_xor_sync(0xffffffffu, q, 2);
                if ((lane & 3) == 0) s_red[BM * WN + rl * WN + wn] = q;
            }
        }
        __syncthreads();
        if (wn == 0 && (lane & 3) == 0) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    const int rl = wm * WTM + mi * 16 + (lane >> 2) + hh * 8;
                    const int row = m0 + rl;
                    if (row < p.M) {
                        float m2 = 0.f;
#pragma unroll
                        for (int w = 0; w < WN; w++) m2 += s_red[BM * WN + rl * WN + w];
                        reinterpret_cast<float2*>(p.stats_out)[(size_t)row * gridDim.x + blockIdx.x] = make_float2(rsum[mi][hh], m2);
                    }
                }
        }
        return;
    }
    const bool full_m = (m0 + BM) <= p.M;
    bf16* stg = reinterpret_cast<bf16*>(smem_raw);
    if (STG) __syncthreads();
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int rl = wm * WTM + mi * 16 + (lane >> 2) + hh * 8;
            const int row = m0 + rl;
            if (!full_m && row >= p.M) continue;
            float mean = 0.f, rstd = 1.f;
            if (LNF) { float2 st = s_st[rl]; mean = st.x; rstd = st.y; }
            if (EPI == 0) {
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    int col = n0 + blk_of(wn, nj) * 8 + (lane & 3) * 2;
                    if (!NFULL && col >= p.N) continue;
                    float x0, x1;
                    if (LNF) {
                        float2 b2 = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(p.bias) + col);
                        float2 c2 = *reinterpret_cast<const float2*>(p.cs + col);
                        x0 = rstd * (acc[mi][nj][hh * 2] - mean * c2.x) + b2.x;
                        x1 = rstd * (acc[mi][nj][hh * 2 + 1] - mean * c2.y) + b2.y;
                    } else {
                        __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(p.bias) + col);
                        x0 = DEQ(acc[mi][nj][hh * 2], row, col) + __bfloat162float(b2.x);
                        x1 = DEQ(acc[mi][nj][hh * 2 + 1], row, col + 1) + __bfloat162float(b2.y);
                    }
                    x0 = x0 * 0.5f * (1.0f + erff(x0 * 0.7071067811865476f));
                    x1 = x1 * 0.5f * (1.0f + erff(x1 * 0.7071067811865476f));
                    if (STG) *reinterpret_cast<out2_t*>(stg + rl * STG_LD + blk_of(wn, nj) * 8 + (lane & 3) * 2) = pk2(x0, x1);
                    else *reinterpret_cast<out2_t*>(reinterpret_cast<bf16*>(p.C) + (size_t)row * p.N + col) =
                        pk2(x0, x1);
                }
            } else if (EPI == 1) {
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    int col = n0 + blk_of(wn, nj) * 8 + (lane & 3) * 2;
                    if (!NFULL && col >= p.N) continue;
                    __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(p.bias) + col);
                    float2 g2 = *reinterpret_cast<const float2*>(p.gamma + col);
                    float x0 = (DEQ(acc[mi][nj][hh * 2], row, col) + __bfloat162float(b2.x)) * g2.x;
                    float x1 = (DEQ(acc[mi][nj][hh * 2 + 1], row, col + 1) + __bfloat162float(b2.y)) * g2.y;
                    size_t o = (size_t)row * p.N + col;
                    if (RF32) {
                        float2 r2 = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(p.R) + o);
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + o) = make_float2(r2.x + x0, r2.y + x1);
                    } else {
                        __nv_bfloat162 r2 = *reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(p.R) + o);
                        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<bf16*>(p.C) + o) =
                            __floats2bfloat162_rn(__bfloat162float(r2.x) + x0, __bfloat162float(r2.y) + x1);
                    }
                }
            } else if (EPI == 4) {   // round10: plain GEMM, out = acc * oscale -> fp16/bf16 [M,N] (head ConvT GEMMs, patch-embed, mask einsum shapes)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    int col = n0 + blk_of(wn, nj) * 8 + (lane & 3) * 2;
                    if (!NFULL && col >= p.N) continue;
                    *reinterpret_cast<out2_t*>(reinterpret_cast<bf16*>(p.C) + (size_t)row * p.N + col) =
                        pk2(DEQ(acc[mi][nj][hh * 2], row, col), DEQ(acc[mi][nj][hh * 2 + 1], row, col + 1));
                }
            } else if (EPI == 3) {
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    int col = n0 + blk_of(wn, nj) * 8 + (lane & 3) * 2;
                    if (!NFULL && col >= p.N) continue;
                    *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + ((size_t)blockIdx.z * p.M + row) * p.N + col) =
                        make_float2(DEQ(acc[mi][nj][hh * 2], row, col), DEQ(acc[mi][nj][hh * 2 + 1], row, col + 1));
                }
            } else {  // EPI == 2: qkv + rope + bnhd layout
                const int b_i = row / p.Ntok, n_i = row % p.Ntok, pp = n_i - p.prefix;
                const int which = n0 / p.d;
                const bool do_rope = (which < 2) && (pp >= 0);
                const int pc = pp > 0 ? pp : 0;
                constexpr int HALF = NJ / 2;
#pragma unroll
                for (int nj = 0; nj < HALF; nj++) {
                    // partner blocks: nj (cols c) and nj+HALF (cols c+32)
                    int blo = blk_of(wn, nj), bhi = blk_of(wn, nj + HALF);
                    int col_lo = n0 + blo * 8 + (lane & 3) * 2, col_hi = n0 + bhi * 8 + (lane & 3) * 2;
                    float l0, l1, h0, h1;
                    if (LNF) {
                        float2 bl = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(p.bias) + col_lo);
                        float2 bh = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(p.bias) + col_hi);
                        float2 cl2 = *reinterpret_cast<const float2*>(p.cs + col_lo);
                        float2 ch2 = *reinterpret_cast<const float2*>(p.cs + col_hi);
                        l0 = rstd * (acc[mi][nj][hh * 2] - mean * cl2.x) + bl.x;
                        l1 = rstd * (acc[mi][nj][hh * 2 + 1] - mean * cl2.y) + bl.y;
                        h0 = rstd * (acc[mi][nj + HALF][hh * 2] - mean * ch2.x) + bh.x;
                        h1 = rstd * (acc[mi][nj + HALF][hh * 2 + 1] - mean * ch2.y) + bh.y;
                    } else {
                        __nv_bfloat162 bl = *reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(p.bias) + col_lo);
                        __nv_bfloat162 bh = *reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const bf16*>(p.bias) + col_hi);
                        l0 = DEQ(acc[mi][nj][hh * 2], row, col_lo) + __bfloat162float(bl.x); l1 = DEQ(acc[mi][nj][hh * 2 + 1], row, col_lo + 1) + __bfloat162float(bl.y);
                        h0 = DEQ(acc[mi][nj + HALF][hh * 2], row, col_hi) + __bfloat162float(bh.x); h1 = DEQ(acc[mi][nj + HALF][hh * 2 + 1], row, col_hi + 1) + __bfloat162float(bh.y);
                    }
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
                        *reinterpret_cast<out2_t*>(stg + rl * STG_LD + blo * 8 + (lane & 3) * 2) = pk2(l0, l1);
                        *reinterpret_cast<out2_t*>(stg + rl * STG_LD + bhi * 8 + (lane & 3) * 2) = pk2(h0, h1);
                    } else {
                        int hd = (col_lo % p.d) / 64, cl2 = (col_lo % p.d) % 64;
                        size_t o = ((((size_t)which * (p.M / p.Ntok) + b_i) * p.Ntok + n_i) * p.H + hd) * 64 + cl2;
                        bf16* Out = reinterpret_cast<bf16*>(p.C);
                        *reinterpret_cast<out2_t*>(Out + o) = pk2(l0, l1);
                        *reinterpret_cast<out2_t*>(Out + o + 32) = pk2(h0, h1);
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
            if (EPI == 0) {
                *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(p.C) + (size_t)grow * p.N + gcol) = v;
            } else {
                const int b_i = grow / p.Ntok, n_i = grow % p.Ntok;
                const int which = n0 / p.d;
                const int hd = (n0 % p.d) / 64 + ch / 8;
                size_t o = ((((size_t)which * (p.M / p.Ntok) + b_i) * p.Ntok + n_i) * p.H + hd) * 64 + (ch % 8) * 8;
                *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(p.C) + o) = v;
            }
        }
    }
}
