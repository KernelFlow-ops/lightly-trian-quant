"""One-off: derive impl/mg16_core.cu from round5's mg_core.cu (src0/mg_core.cu)."""
s = open("src0/mg_core.cu").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:60], s.count(old))
    s = s.replace(old, new)


# header
rep("// Round 5: hand-written sm_120 bf16 GEMM", """// Round 10 PROTOTYPE `mg_f16acc`: copy of round5_gemm_sm120/impl/mg_core.cu (read-only source) with
//   * A / W operands fp16 (the cp.async / ldmatrix / swizzle code is type agnostic),
//   * the MMA replaced by mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 (full rate on the GeForce, fp32-accumulate HMMA is half rate),
//     fp16 accumulators are zeroed and, every PROMO k16-steps (default: every k-tile), added into the fp32 accumulator registers,
//   * Params gets `oscale` (power-of-two un-scaling of the pre-scaled fp16 operands, applied to acc in every epilogue),
//   * EPI 0 / 2 can write fp16 outputs (-DOUTF16=1) so that the next GEMM / attention consumes fp16 directly.
//   Epilogues (RoPE + [3,B,N,H,64] layout, GELU, gamma + fp32 residual, split-K) are otherwise unchanged; LNF / STATS paths are not used here.
// Original header follows.
// Round 5: hand-written sm_120 bf16 GEMM""")
# params
rep("    float eps;\n    const char* pf0;", "    float eps;\n    float oscale;                  // round10: acc *= oscale in every epilogue (power of two)\n    const char* pf0;")
# defaults
rep("#ifndef STATS\n#define STATS 0\n#endif", """#ifndef STATS
#define STATS 0
#endif
#ifndef F16ACC
#define F16ACC 1      // 1: fp16-accumulate MMA + periodic promotion to fp32, 0: original bf16 MMA (sanity check of the copy)
#endif
#ifndef OUTF16
#define OUTF16 1      // EPI 0/2 output dtype: 1 = fp16, 0 = bf16
#endif
#ifndef PROMO
#define PROMO (BK / 16)   // k16 steps accumulated in fp16 between promotions
#endif""")
# mma
rep("__device__ __forceinline__ void pdl_wait()", """__device__ __forceinline__ void mma16816h(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\\n"
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
__device__ __forceinline__ void pdl_wait()""", 1)
rep("#include <cuda_bf16.h>", "#include <cuda_bf16.h>\n#include <cuda_fp16.h>")

# main loop FRAGALL path
old_fa = """#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++)
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816(acc[mi][nj], af[ks][mi], bfr[ks][nj][0], bfr[ks][nj][1]);
"""
new_fa = """#if F16ACC
        uint32_t ah[MI][NJ][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) { ah[mi][nj][0] = 0u; ah[mi][nj][1] = 0u; }
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816h(ah[mi][nj], af[ks][mi], bfr[ks][nj][0], bfr[ks][nj][1]);
            if ((ks + 1) % PROMO == 0) {
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++) promote_h(acc[mi][nj], ah[mi][nj]);
            }
        }
#else
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++)
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816(acc[mi][nj], af[ks][mi], bfr[ks][nj][0], bfr[ks][nj][1]);
#endif
"""
rep(old_fa, new_fa)
# non-FRAGALL path
old_nf = """#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            uint32_t af[MI][4], bfr[NJ][2];"""
new_nf = """#if F16ACC
        uint32_t ah[MI][NJ][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int nj = 0; nj < NJ; nj++) { ah[mi][nj][0] = 0u; ah[mi][nj][1] = 0u; }
#endif
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            uint32_t af[MI][4], bfr[NJ][2];"""
rep(old_nf, new_nf)
old_nf2 = """#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816(acc[mi][nj], af[mi], bfr[nj][0], bfr[nj][1]);
        }
#endif"""
new_nf2 = """#if F16ACC
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816h(ah[mi][nj], af[mi], bfr[nj][0], bfr[nj][1]);
            if ((ks + 1) % PROMO == 0) {
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++) promote_h(acc[mi][nj], ah[mi][nj]);
            }
#else
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) mma16816(acc[mi][nj], af[mi], bfr[nj][0], bfr[nj][1]);
#endif
        }
#endif"""
rep(old_nf2, new_nf2)

# epilogue oscale + out types
rep("""                        x0 = acc[mi][nj][hh * 2] + __bfloat162float(b2.x);
                        x1 = acc[mi][nj][hh * 2 + 1] + __bfloat162float(b2.y);""",
    """                        x0 = acc[mi][nj][hh * 2] * p.oscale + __bfloat162float(b2.x);
                        x1 = acc[mi][nj][hh * 2 + 1] * p.oscale + __bfloat162float(b2.y);""")
rep("""                    if (STG) *reinterpret_cast<__nv_bfloat162*>(stg + rl * STG_LD + blk_of(wn, nj) * 8 + (lane & 3) * 2) = __floats2bfloat162_rn(x0, x1);
                    else *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<bf16*>(p.C) + (size_t)row * p.N + col) =
                        __floats2bfloat162_rn(x0, x1);""",
    """                    if (STG) *reinterpret_cast<out2_t*>(stg + rl * STG_LD + blk_of(wn, nj) * 8 + (lane & 3) * 2) = pk2(x0, x1);
                    else *reinterpret_cast<out2_t*>(reinterpret_cast<bf16*>(p.C) + (size_t)row * p.N + col) =
                        pk2(x0, x1);""")
rep("""                    float x0 = (acc[mi][nj][hh * 2] + __bfloat162float(b2.x)) * g2.x;
                    float x1 = (acc[mi][nj][hh * 2 + 1] + __bfloat162float(b2.y)) * g2.y;
                    size_t o = (size_t)row * p.N + col;
                    if (RF32) {""",
    """                    float x0 = (acc[mi][nj][hh * 2] * p.oscale + __bfloat162float(b2.x)) * g2.x;
                    float x1 = (acc[mi][nj][hh * 2 + 1] * p.oscale + __bfloat162float(b2.y)) * g2.y;
                    size_t o = (size_t)row * p.N + col;
                    if (RF32) {""")
rep("""                        l0 = acc[mi][nj][hh * 2] + __bfloat162float(bl.x); l1 = acc[mi][nj][hh * 2 + 1] + __bfloat162float(bl.y);
                        h0 = acc[mi][nj + HALF][hh * 2] + __bfloat162float(bh.x); h1 = acc[mi][nj + HALF][hh * 2 + 1] + __bfloat162float(bh.y);""",
    """                        l0 = acc[mi][nj][hh * 2] * p.oscale + __bfloat162float(bl.x); l1 = acc[mi][nj][hh * 2 + 1] * p.oscale + __bfloat162float(bl.y);
                        h0 = acc[mi][nj + HALF][hh * 2] * p.oscale + __bfloat162float(bh.x); h1 = acc[mi][nj + HALF][hh * 2 + 1] * p.oscale + __bfloat162float(bh.y);""")
rep("""                        *reinterpret_cast<__nv_bfloat162*>(stg + rl * STG_LD + blo * 8 + (lane & 3) * 2) = __floats2bfloat162_rn(l0, l1);
                        *reinterpret_cast<__nv_bfloat162*>(stg + rl * STG_LD + bhi * 8 + (lane & 3) * 2) = __floats2bfloat162_rn(h0, h1);""",
    """                        *reinterpret_cast<out2_t*>(stg + rl * STG_LD + blo * 8 + (lane & 3) * 2) = pk2(l0, l1);
                        *reinterpret_cast<out2_t*>(stg + rl * STG_LD + bhi * 8 + (lane & 3) * 2) = pk2(h0, h1);""")
rep("""                        *reinterpret_cast<__nv_bfloat162*>(Out + o) = __floats2bfloat162_rn(l0, l1);
                        *reinterpret_cast<__nv_bfloat162*>(Out + o + 32) = __floats2bfloat162_rn(h0, h1);""",
    """                        *reinterpret_cast<out2_t*>(Out + o) = pk2(l0, l1);
                        *reinterpret_cast<out2_t*>(Out + o + 32) = pk2(h0, h1);""")
rep("""                    *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + ((size_t)blockIdx.z * p.M + row) * p.N + col) =
                        make_float2(acc[mi][nj][hh * 2], acc[mi][nj][hh * 2 + 1]);""",
    """                    *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.C) + ((size_t)blockIdx.z * p.M + row) * p.N + col) =
                        make_float2(acc[mi][nj][hh * 2] * p.oscale, acc[mi][nj][hh * 2 + 1] * p.oscale);""")
open("impl/mg16_core.cu", "w").write(s)
print("ok", len(s))
