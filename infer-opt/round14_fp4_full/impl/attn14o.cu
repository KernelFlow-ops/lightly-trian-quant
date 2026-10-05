// Round 14: copy of round12_lowp_stack/impl/attn16o.cu (fp16 in / fp16 out FA2, kind 4) with namespace a16 renamed to a14o, so that it can live in the same process as the r12 module (see NOTES sec 5).
// Round 12: copy of round11_attn_tier0/impl/attn16.cu with the output O written as fp16 (proj GEMM operand) instead of bf16.
// Round 11 prototype: FP16-accumulate FlashAttention-2 forward for sm_120 (RTX 5090), D = 64, no mask, RoPE already applied.
//
// Base: train-opt round12_attention_sm120/impl/attn_fwd.cu (hand-written TMA + mbarrier FA2, itself adapted from
//       spark-kernels attention.cu, MIT, commit a401bfa9e363).  Changes:
//   * input layout = the r5 qkv GEMM epilogue output [3][B][N][H][64] (q, k, v are three [B,N,H,64] tensors; TMA box per head).
//   * no RoPE rotator warp, no query bitmask, no lse / qk_rot outputs.
//   * fp16 or bf16 inputs.  KIND selects the MMA flavour:
//       0: bf16 in, f32 accumulate (S and PV)      -- the numerically-flash baseline (same instruction mix as torch flash)
//       1: fp16 in, f32 accumulate
//       2: fp16 in, S accumulated in f16 (4 chained k16 MMAs), PV f32
//       3: fp16 in, S f32, PV accumulated in f16 per 64-key tile, promoted into the f32 O after every tile
//       4: fp16 in, S f16 and PV f16 (promotion per tile)           -- full "FP16-accumulate FA2" prototype
//     The GeForce parts run f16-accumulate m16n8k16 HMMA at 2x the f32-accumulate rate.  Softmax statistics (running max, row sum, alpha)
//     and the O accumulator stay in fp32; P is rounded to fp16 (11 mantissa bits, better than the bf16 P of flash).
//     Q is pre-scaled by 1/sqrt(64) = 2^-3 in registers (exact) when S is accumulated in f16, so partial sums stay far from 65504.
//   * output O [B][N][H*64] bf16 (the proj GEMM input of the r5 block).
//   * key loop covers N exactly (tail tile predicated), query tiles of CW*16 rows, warps past N retire.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef CW
#define CW 8
#endif
#ifndef STAGES
#define STAGES 4
#endif
#ifndef MINB
#define MINB 1
#endif

namespace a14o {

#define A16_CHECK(expr)                                                                          \
    do {                                                                                         \
        cudaError_t _e = (expr);                                                                 \
        if (_e != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e) + " at line " + std::to_string(__LINE__)); \
    } while (0)

constexpr unsigned kFullMask = 0xffffffffu;
constexpr float kLog2e = 1.4426950408889634f;
__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16_f32(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16_f32(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16_f16(unsigned (&d)[2], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory"); }
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, unsigned bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nA16_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@p bra A16_DONE;\nbra A16_WAIT;\nA16_DONE:\n}\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_4d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2, int c3) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n" ::"r"(smem_u32(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap* map) { asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory"); }
__device__ __forceinline__ int swz_off(int row, int ch) { return row * 64 + ((ch ^ (row & 7)) << 3); }   // elements, 128B-swizzled [rows][64]

using TensorMapEncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                       const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
inline TensorMapEncodeFn tensor_map_encoder() {
    static TensorMapEncodeFn fn = nullptr;
    if (fn == nullptr) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
        A16_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
        if (p == nullptr || q != cudaDriverEntryPointSuccess) throw std::runtime_error("cuTensorMapEncodeTiled unavailable");
        fn = reinterpret_cast<TensorMapEncodeFn>(p);
    }
    return fn;
}
// [B][N][H][64] 16-bit tensor, 4-D map {d=64, h, n, b}, box {64, 1, rows, 1}, 128B swizzle; OOB rows (n >= N) are zero filled.
inline CUtensorMap make_map(const void* base, bool fp16, int B, int N, int H, int box_rows) {
    CUtensorMap map;
    const cuuint64_t dims[4] = {64, (cuuint64_t)H, (cuuint64_t)N, (cuuint64_t)B};
    const cuuint64_t strides[3] = {128, (cuuint64_t)H * 128, (cuuint64_t)N * H * 128};
    const cuuint32_t box[4] = {64, 1, (cuuint32_t)box_rows, 1};
    const cuuint32_t es[4] = {1, 1, 1, 1};
    const CUresult r = tensor_map_encoder()(&map, fp16 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base), dims, strides,
                                            box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed: " + std::to_string((int)r));
    return map;
}

constexpr int BM = CW * 16, BN = 64;
constexpr int THREADS = CW * 32 + 32;
constexpr int Q_BYTES = BM * 128, KV_BYTES = BN * 128, STAGE_BYTES = 2 * KV_BYTES;
constexpr int SMEM_BYTES = Q_BYTES + STAGES * STAGE_BYTES + 128;

struct Args { __nv_bfloat16* O; int N, H, BH; float pre; };   // pre: power-of-two Q pre-scale for the f16-accumulated S (overflow headroom), compensated in the softmax scale

__device__ __forceinline__ unsigned pack2(float lo, float hi, bool fp16) {
    if (fp16) { __half2 h = __floats2half2_rn(lo, hi); return *reinterpret_cast<unsigned*>(&h); }
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<unsigned*>(&h);
}

template <int KIND>
__global__ void __launch_bounds__(THREADS, MINB) fwd_kernel(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
                                                           const __grid_constant__ CUtensorMap tmV, Args a) {
    constexpr bool FP16 = KIND != 0;
    constexpr bool S16 = KIND == 2 || KIND == 4;
    constexpr bool PV16 = KIND == 3 || KIND == 4;
    extern __shared__ __align__(1024) unsigned char smem[];
    unsigned char* Qs = smem;
    unsigned char* stages = smem + Q_BYTES;
    uint64_t* bars = reinterpret_cast<uint64_t*>(stages + STAGES * STAGE_BYTES);
    uint64_t* q_full = bars;
    uint64_t* full = bars + 1;
    uint64_t* empty = full + STAGES;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int bh = blockIdx.x % a.BH, qt = blockIdx.x / a.BH;
    const int b = bh / a.H, h = bh - b * a.H;
    const int q0 = qt * BM;
    const int N = a.N;
    const int T = cdiv(N, BN);
    const int kv_tail = N - BN * (T - 1);
    const int n_active = min(CW, cdiv(N - q0, 16));

    if (tid == 0) {
        mbar_init(q_full, 1);
#pragma unroll
        for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], n_active); }
        fence_mbar_init();
    }
    __syncthreads();

    if (warp == CW) {  // ---- TMA producer
        if (lane == 0) {
            prefetch_tensormap(&tmQ); prefetch_tensormap(&tmK); prefetch_tensormap(&tmV);
            mbar_arrive_expect_tx(q_full, Q_BYTES);
            tma_load_4d(Qs, &tmQ, q_full, 0, h, q0, b);
            for (int u = 0; u < T; ++u) {
                const int s = u % STAGES;
                const unsigned use = u / STAGES;
                if (use > 0) mbar_wait(&empty[s], (use - 1) & 1);
                mbar_arrive_expect_tx(&full[s], STAGE_BYTES);
                unsigned char* dst = stages + s * STAGE_BYTES;
                tma_load_4d(dst, &tmK, &full[s], 0, h, u * BN, b);
                tma_load_4d(dst + KV_BYTES, &tmV, &full[s], 0, h, u * BN, b);
            }
        }
        return;
    }
    if (warp >= n_active) return;

    unsigned qf[4][4];
    mbar_wait(q_full, 0);
    {
        const int row = warp * 16 + (lane & 15);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) ldmatrix_x4(qf[kk], Qs + 2 * swz_off(row, 2 * kk + (lane >> 4)));
    }
    if constexpr (S16) {  // fold 1/sqrt(64) into Q: exact power of two
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                __half2 v = *reinterpret_cast<__half2*>(&qf[kk][e]);
                v = __hmul2(v, __floats2half2_rn(0.125f * a.pre, 0.125f * a.pre));
                qf[kk][e] = *reinterpret_cast<unsigned*>(&v);
            }
    }
    const float SCALE = S16 ? kLog2e / a.pre : 0.125f * kLog2e;

    float o[8][4];
#pragma unroll
    for (int dj = 0; dj < 8; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float sacc[8][4];
    unsigned pa[4][4];
    const int row0 = q0 + warp * 16 + g;

    auto qk = [&](const unsigned char* ks, bool tail) {
        if constexpr (S16) {
            unsigned s16[8][2];
#pragma unroll
            for (int nj = 0; nj < 8; ++nj) { s16[nj][0] = 0u; s16[nj][1] = 0u; }
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int nj = 0; nj < 8; nj += 2) {
                    if (tail && nj * 8 >= kv_tail) continue;
                    const int row = nj * 8 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4(r, ks + 2 * swz_off(row, 2 * kk + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[2]};
                    const unsigned b1[2] = {r[1], r[3]};
                    mma_f16_f16(s16[nj], qf[kk], b0);
                    if (!tail || (nj + 1) * 8 < kv_tail) mma_f16_f16(s16[nj + 1], qf[kk], b1);
                }
            }
#pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&s16[nj][0]));
                float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&s16[nj][1]));
                sacc[nj][0] = f0.x; sacc[nj][1] = f0.y; sacc[nj][2] = f1.x; sacc[nj][3] = f1.y;
            }
        } else {
#pragma unroll
            for (int nj = 0; nj < 8; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e) sacc[nj][e] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int nj = 0; nj < 8; nj += 2) {
                    if (tail && nj * 8 >= kv_tail) continue;
                    const int row = nj * 8 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4(r, ks + 2 * swz_off(row, 2 * kk + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[2]};
                    const unsigned b1[2] = {r[1], r[3]};
                    if constexpr (FP16) {
                        mma_f16_f32(sacc[nj], qf[kk], b0);
                        if (!tail || (nj + 1) * 8 < kv_tail) mma_f16_f32(sacc[nj + 1], qf[kk], b1);
                    } else {
                        mma_bf16_f32(sacc[nj], qf[kk], b0);
                        if (!tail || (nj + 1) * 8 < kv_tail) mma_bf16_f32(sacc[nj + 1], qf[kk], b1);
                    }
                }
            }
        }
    };
    auto softmax = [&](bool tail) {
        if (tail) {
#pragma unroll
            for (int nj = 0; nj < 8; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (nj * 8 + c2 + (e & 1) >= kv_tail) sacc[nj][e] = -INFINITY;
        }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx = fmaxf(sacc[0][2 * r], sacc[0][2 * r + 1]);
#pragma unroll
            for (int nj = 1; nj < 8; ++nj) mx = fmaxf(mx, fmaxf(sacc[nj][2 * r], sacc[nj][2 * r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            const float m_new = fmaxf(m[r], mx * SCALE);   // running max: never reset per tile
            const float m_use = m_new == -INFINITY ? 0.f : m_new;
            alpha[r] = ex2(m[r] - m_use);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                const float p0 = ex2(fmaf(sacc[nj][2 * r], SCALE, -m_use));
                const float p1 = ex2(fmaf(sacc[nj][2 * r + 1], SCALE, -m_use));
                sacc[nj][2 * r] = p0;
                sacc[nj][2 * r + 1] = p1;
                rs += p0 + p1;
            }
            l[r] = fmaf(l[r], alpha[r], rs);
            m[r] = m_new;
        }
#pragma unroll
        for (int dj = 0; dj < 8; ++dj) {
            o[dj][0] *= alpha[0]; o[dj][1] *= alpha[0]; o[dj][2] *= alpha[1]; o[dj][3] *= alpha[1];
        }
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            pa[kt][0] = pack2(sacc[2 * kt][0], sacc[2 * kt][1], FP16);
            pa[kt][1] = pack2(sacc[2 * kt][2], sacc[2 * kt][3], FP16);
            pa[kt][2] = pack2(sacc[2 * kt + 1][0], sacc[2 * kt + 1][1], FP16);
            pa[kt][3] = pack2(sacc[2 * kt + 1][2], sacc[2 * kt + 1][3], FP16);
        }
    };
    auto pv = [&](const unsigned char* vs, bool tail) {
        if constexpr (PV16) {
            unsigned o16[8][2];
#pragma unroll
            for (int dj = 0; dj < 8; ++dj) { o16[dj][0] = 0u; o16[dj][1] = 0u; }
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                if (tail && kt * 16 >= kv_tail) continue;
#pragma unroll
                for (int dj = 0; dj < 8; dj += 2) {
                    const int row = kt * 16 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4_trans(r, vs + 2 * swz_off(row, dj + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[1]};
                    const unsigned b1[2] = {r[2], r[3]};
                    mma_f16_f16(o16[dj], pa[kt], b0);
                    mma_f16_f16(o16[dj + 1], pa[kt], b1);
                }
            }
#pragma unroll
            for (int dj = 0; dj < 8; ++dj) {   // promote the per-tile f16 partial sums into the f32 O
                float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&o16[dj][0]));
                float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&o16[dj][1]));
                o[dj][0] += f0.x; o[dj][1] += f0.y; o[dj][2] += f1.x; o[dj][3] += f1.y;
            }
        } else {
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                if (tail && kt * 16 >= kv_tail) continue;
#pragma unroll
                for (int dj = 0; dj < 8; dj += 2) {
                    const int row = kt * 16 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4_trans(r, vs + 2 * swz_off(row, dj + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[1]};
                    const unsigned b1[2] = {r[2], r[3]};
                    if constexpr (FP16) { mma_f16_f32(o[dj], pa[kt], b0); mma_f16_f32(o[dj + 1], pa[kt], b1); }
                    else { mma_bf16_f32(o[dj], pa[kt], b0); mma_bf16_f32(o[dj + 1], pa[kt], b1); }
                }
            }
        }
    };

    for (int u = 0; u < T; ++u) {
        const int s = u % STAGES;
        mbar_wait(&full[s], (u / STAGES) & 1);
        const unsigned char* ks = stages + s * STAGE_BYTES;
        const unsigned char* vs = ks + KV_BYTES;
        if (u + 1 < T || kv_tail == BN) { qk(ks, false); softmax(false); pv(vs, false); }
        else { qk(ks, true); softmax(true); pv(vs, true); }
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float ls = l[r];
        ls += __shfl_xor_sync(kFullMask, ls, 1);
        ls += __shfl_xor_sync(kFullMask, ls, 2);
        const int row = row0 + 8 * r;
        if (row >= N) continue;
        const float inv = 1.f / ls;
        __nv_bfloat16* out = a.O + (((size_t)b * N + row) * a.H + h) * 64 + c2;
#pragma unroll
        for (int dj = 0; dj < 8; ++dj)
            *reinterpret_cast<__half2*>(out + dj * 8) = __floats2half2_rn(o[dj][2 * r] * inv, o[dj][2 * r + 1] * inv);
    }
}

template <int KIND>
void launch(const at::Tensor& qkv, at::Tensor& O, cudaStream_t stream, float pre) {
    const int B = qkv.size(1), N = qkv.size(2), H = qkv.size(3);
    static bool attr = false;
    if (!attr) { A16_CHECK(cudaFuncSetAttribute(fwd_kernel<KIND>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES)); attr = true; }
    const size_t plane = (size_t)B * N * H * 64 * 2;   // bytes
    const char* base = reinterpret_cast<const char*>(qkv.data_ptr());
    const bool fp16 = KIND != 0;
    const CUtensorMap tmQ = make_map(base, fp16, B, N, H, BM);
    const CUtensorMap tmK = make_map(base + plane, fp16, B, N, H, BN);
    const CUtensorMap tmV = make_map(base + 2 * plane, fp16, B, N, H, BN);
    Args a{reinterpret_cast<__nv_bfloat16*>(O.data_ptr()), N, H, B * H, pre};
    fwd_kernel<KIND><<<B * H * cdiv(N, BM), THREADS, SMEM_BYTES, stream>>>(tmQ, tmK, tmV, a);
    A16_CHECK(cudaGetLastError());
}

}  // namespace a14o

// qkv: [3,B,N,H,64] contiguous, bf16 (kind 0) or fp16 (kinds 1-4).  Returns O [B,N,H*64] bf16.
at::Tensor attn16_fwd(const at::Tensor& qkv, int64_t kind, double pre) {
    TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous() && qkv.dim() == 5 && qkv.size(0) == 3 && qkv.size(4) == 64, "qkv must be contiguous [3,B,N,H,64]");
    TORCH_CHECK(qkv.scalar_type() == (kind == 0 ? at::kBFloat16 : at::kHalf), "dtype must be bf16 for kind 0, fp16 otherwise");
    const c10::cuda::CUDAGuard guard(qkv.device());
    at::Tensor O = at::empty({qkv.size(1), qkv.size(2), qkv.size(3) * 64}, qkv.options().dtype(at::kHalf));
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    switch (kind) {
        case 0: a14o::launch<0>(qkv, O, st, (float)pre); break;
        case 1: a14o::launch<1>(qkv, O, st, (float)pre); break;
        case 2: a14o::launch<2>(qkv, O, st, (float)pre); break;
        case 3: a14o::launch<3>(qkv, O, st, (float)pre); break;
        case 4: a14o::launch<4>(qkv, O, st, (float)pre); break;
        default: TORCH_CHECK(false, "kind in 0..4");
    }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn16_fwd", &attn16_fwd, py::arg("qkv"), py::arg("kind"), py::arg("pre") = 1.0); }
