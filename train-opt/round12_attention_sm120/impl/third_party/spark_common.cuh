// Device helpers for sm_120 mma.sync / ldmatrix / mbarrier / TMA.
//
// Adapted from spark-kernels `include/spark/common.cuh`
//   source : https://github.com/aidxhxr/5090-blackwell-kernels
//   commit : a401bfa9e363 (see /root/autodl-tmp/train-opt-tmp/third_party/SOURCES.txt)
//   license: MIT, Copyright (c) 2026 Amirkhan Aidarkhan (full text in LICENSE.spark)
// Only the primitives used by round12 are kept; the 4-D tensor-map encoder and the 4-D TMA load are
// new (packed [B,N,3,H,64] QKV layout).
#pragma once

#include <cuda.h>  // CUtensorMap and the cuTensorMapEncodeTiled types (header only, no libcuda link)
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace a12 {

using bf16 = __nv_bfloat16;

#define A12_CUDA_CHECK(expr)                                                                      \
    do {                                                                                          \
        cudaError_t _err = (expr);                                                                \
        if (_err != cudaSuccess) {                                                                \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) +     \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                                         \
    } while (0)

constexpr unsigned kFullMask = 0xffffffffu;

__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// 2^x on the MUFU pipe. ex2.approx takes -inf to +0, which is what a masked score needs.
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ float bf16lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }

// ldmatrix: four 8x8 b16 matrices from shared memory into one register per matrix per lane.
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned (&r)[4], const void* smem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
// D[16x8] (+)= A[16x16] * B[16x8], bf16 inputs, fp32 accumulate.
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "A12_MBAR_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra A12_MBAR_DONE;\n"
        "bra A12_MBAR_WAIT;\n"
        "A12_MBAR_DONE:\n"
        "}\n" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}
// 4-D TMA load: coordinates (c0 innermost .. c3 outermost), in elements.
__device__ __forceinline__ void tma_load_4d(void* smem_dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2, int c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, %6}], [%2];\n" ::"r"(smem_u32(smem_dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
        : "memory");
}
__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
// 1-D bulk copy global -> shared through the copy engine (no tensor map), completion counted on `bar` like a TMA box.
// `bytes` a multiple of 16, both addresses 16-byte aligned.
__device__ __forceinline__ void bulk_load(void* smem_dst, const void* gmem_src, unsigned bytes, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(smem_u32(smem_dst)),
        "l"(gmem_src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}
// fp32 vector atomic add (RED.E.ADD.F32x2 on sm_90+; the return value is unused).
__device__ __forceinline__ void red_add2(float* p, float a, float b) { atomicAdd(reinterpret_cast<float2*>(p), make_float2(a, b)); }

__device__ __forceinline__ void named_barrier_sync(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(nthreads) : "memory");
}

// ---- host side: tensor maps ------------------------------------------------------------------
using TensorMapEncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                       const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                       CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
inline TensorMapEncodeFn tensor_map_encoder() {
    static TensorMapEncodeFn fn = nullptr;
    if (fn == nullptr) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
        A12_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
        if (p == nullptr || q != cudaDriverEntryPointSuccess) throw std::runtime_error("cuTensorMapEncodeTiled is not available");
        fn = reinterpret_cast<TensorMapEncodeFn>(p);
    }
    return fn;
}

// Packed activations [B][N][S][64] bf16, S = number of 64-wide slots per token (3*H for packed qkv,
// H for O / dO): a 4-D map {d=64, slot, n, b} with a {64, 1, box_rows, 1} box and the 128-byte swizzle.
// Rows past N (inside one batch) are zero-filled by the copy engine.
inline CUtensorMap make_slot_map(const void* base, int B, int N, int S, int box_rows) {
    CUtensorMap map;
    const cuuint64_t dims[4] = {64, (cuuint64_t)S, (cuuint64_t)N, (cuuint64_t)B};
    const cuuint64_t strides[3] = {128, (cuuint64_t)S * 128, (cuuint64_t)N * S * 128};
    const cuuint32_t box[4] = {64, 1, (cuuint32_t)box_rows, 1};
    const cuuint32_t es[4] = {1, 1, 1, 1};
    const CUresult r = tensor_map_encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base), dims, strides, box, es,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled(4d) failed: " + std::to_string((int)r));
    return map;
}

}  // namespace a12
