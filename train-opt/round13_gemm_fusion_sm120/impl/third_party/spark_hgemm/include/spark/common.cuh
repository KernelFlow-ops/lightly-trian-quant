// Copied unmodified from aidxhxr/5090-blackwell-kernels (Spark kernels) @ a401bfa9e363, MIT License (see ../../LICENSE).
// Shared helpers for every kernel in spark-kernels.
// Primary target: NVIDIA RTX 5090 (GB202), compute capability 12.0 (sm_120), CUDA 13.x.
// Secondary target: NVIDIA GB10 (DGX Spark), compute capability 12.1 (sm_121).
#pragma once

#include <cuda.h>  // CUtensorMap and the cuTensorMapEncodeTiled types (header only, no libcuda link)
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace spark {

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------
#define SPARK_CUDA_CHECK(expr)                                                                \
    do {                                                                                      \
        cudaError_t _err = (expr);                                                            \
        if (_err != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
        }                                                                                     \
    } while (0)

// Check the most recent kernel launch (call right after <<<>>>).
#define SPARK_CHECK_LAUNCH() SPARK_CUDA_CHECK(cudaGetLastError())

#define SPARK_REQUIRE(cond, msg)                                    \
    do {                                                            \
        if (!(cond)) throw std::invalid_argument(std::string(msg)); \
    } while (0)

// ---------------------------------------------------------------------------
// Small host/device utilities
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) {
    return (a + b - 1) / b;
}
__host__ __device__ __forceinline__ constexpr int64_t cdiv64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// SM count of the current device; the grid-stride kernels size their grids from it. Cached
// after the first call. If the attribute query reports nothing, assume the primary target.
constexpr int kFallbackSMs = 170;  // RTX 5090 (the GB10 has 48)
inline int num_sms() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        SPARK_CUDA_CHECK(cudaGetDevice(&dev));
        SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        if (sms <= 0) sms = kFallbackSMs;
    }
    return sms;
}

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// True if `p` can be read or written as one 128-bit vector. cudaMalloc returns 256-byte
// aligned buffers, but a pointer into the middle of one (a PyTorch storage offset, say) may
// not be; the vectorized kernels must refuse those instead of faulting with a misaligned
// address, which poisons the CUDA context for the rest of the process.
inline bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) % 16) == 0;
}

// ---------------------------------------------------------------------------
// Warp-level reductions (all lanes receive the result)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(kFullMask, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(kFullMask, v, offset));
    }
    return v;
}

// Block-level reductions. `smem` must hold at least 32 floats. All threads in
// the block receive the result. Requires blockDim.x to be a multiple of 32.
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nwarps) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    v = smem[0];
    __syncthreads();  // allow smem reuse by the caller
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nwarps) ? smem[lane] : -INFINITY;
    if (wid == 0) v = warp_reduce_max(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    v = smem[0];
    __syncthreads();
    return v;
}

// ---------------------------------------------------------------------------
// bf16 <-> f32 conversion helpers (uniform names for both element types)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float to_f32(float v) {
    return v;
}
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ float from_f32<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// Vector-of-8 loads for bf16 (16 bytes) and vector-of-4 for f32 (16 bytes).
// Both are "one 128-bit transaction per thread" — the ideal width on Blackwell sm_12x.
struct __align__(16) bf16x8 {
    __nv_bfloat162 h[4];
};
struct __align__(16) f32x4 {
    float v[4];
};

// ---------------------------------------------------------------------------
// cp.async (Ampere+; available on sm_120 / sm_121). 16-byte global->shared copy that
// bypasses registers. Used by the double-buffered GEMM pipelines.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr) : "memory");
}
// Same, but with `valid == false` nothing is read and the 16 smem bytes are zero-filled
// (src-size 0): the way a tile row past the end of a matrix is loaded without a branch.
__device__ __forceinline__ void cp_async_16_zfill(void* smem_ptr, const void* gmem_ptr,
                                                  bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem_ptr),
                 "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
// The "memory" clobber keeps the compiler from hoisting shared-memory reads above the wait.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ---------------------------------------------------------------------------
// Raw tensor-core primitives (sm_80+; used by the hgemm mma.sync variants).
// ---------------------------------------------------------------------------
// ldmatrix: four 8x8 b16 matrices from shared memory into one register per matrix per lane.
// Lanes 8i..8i+7 supply the row addresses of matrix i (16 bytes per row).
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
// Same, transposing each 8x8 matrix on the way: turns a k-major (row = k) tile of B into the
// "col" operand layout mma.sync wants.
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned (&r)[4], const void* smem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
// D[16x8] (+)= A[16x16] * B[16x8], bf16 inputs, fp32 accumulate. Fragment layouts are the
// PTX ISA ones for m16n8k16: a[4] = (rows g / g+8) x (k 0-7 / 8-15), b[2] = k 0-7 / 8-15,
// d[4] = (row g, cols 2c..2c+1), (row g+8, same cols) with g = lane/4, c = lane%4.
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const unsigned (&a)[4],
                                               const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// fp32 -> tf32. A tf32 operand is an fp32 word whose low 13 mantissa bits the tensor core
// ignores, so any fp32 register can be handed to the mma as it is (truncation, a biased
// 2^-10 relative error). Rounding to the nearest tf32 first halves the bound to 2^-11 and
// removes the bias. cvt.rn (nearest even, sm_90+) is one SASS instruction on sm_120
// (F2FP.TF32.F32); cvt.rna (ties away, the sm_80 form) is emulated there with three (a
// NaN test, an add of 0x1000 and a mask), which at 24 conversions per 16 mma set the clock
// of the tf32 GEMM (docs/design/sgemm.md).
__device__ __forceinline__ unsigned f32_to_tf32(float x) {
    unsigned r;
    asm("cvt.rn.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}
// D[16x8] (+)= A[16x8] * B[8x8], tf32 inputs, fp32 accumulate (sgemm variants 6 and 7).
// Fragment layouts are the PTX ISA ones for m16n8k8 with 32-bit types, one element per
// register: a[4] = (row g, k c), (row g+8, k c), (row g, k c+4), (row g+8, k c+4);
// b[2] = (k c, col g), (k c+4, col g); d[4] as for m16n8k16, with g = lane/4, c = lane%4.
// 2,048 FLOP per instruction, half of m16n8k16's 4,096, and the RTX 5090 issues it at half
// the bf16 rate on top (bench_peak: 130 vs 259 TFLOPS).
__device__ __forceinline__ void mma_tf32_1688(float (&d)[4], const unsigned (&a)[4],
                                              const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// D[16x8] (+)= A[16x32] * B[32x8], e4m3 inputs, fp32 accumulate (the fp8gemm variants).
// Fragment layouts are the PTX ISA ones for m16n8k32 with 8-bit types, four elements per
// register with the lowest k in the low byte: a[4] = (row g / g+8) x (k 4c..4c+3 / 16+4c..
// 16+4c+3), b[2] = (k 4c..4c+3 / 16+4c..16+4c+3) x col g, d[4] as for m16n8k16. Read as
// 16-bit pairs of adjacent k, a[] is the m16n8k16 A fragment of the same bytes and b[] the
// m16n8k16 B fragment, which is what lets ldmatrix (a b16 instruction) load both.
//
// Two instructions compute this. The plain one (mma.sync...f32.e4m3.e4m3.f32, sm_89+) runs
// at half the fp16-accumulate rate on the RTX 5090, 517 TFLOPS measured, as on the GeForce
// Ada parts. The block-scaled one (.kind::mxf8f6f4, the MXFP8 instruction, sm_120a) scales
// each row of A and column of B by a ue8m0 factor (a power of two) before the same fp32
// accumulation, and with every factor 2^0 it computes the same bits at the full rate, 1,027
// TFLOPS measured; it is the instruction cuBLASLt's fp8 kernels issue on this card
// (QMMA.SF...E8 in the SASS). It is only exposed on the architecture-specific targets, so
// the fp8 kernels are compiled for sm_120a / sm_121a and fall back to the plain instruction
// on a plain sm_120 build (docs/design/fp8gemm.md).
__device__ __forceinline__ void mma_e4m3_16832_plain(float (&d)[4], const unsigned (&a)[4],
                                                     const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define SPARK_HAS_MX_MMA 1
// The scale operands: a .b32 register of four ue8m0 bytes per lane, and {byte, thread}
// immediates that pick which byte of which lanes' registers the hardware reads for each row
// and column. With 0x7F (2^0) in every byte of every lane the selection does not matter.
__device__ __forceinline__ void mma_e4m3_16832(float (&d)[4], const unsigned (&a)[4],
                                               const unsigned (&b)[2]) {
    const unsigned one = 0x7F7F7F7Fu;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %10, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(one));
}
// The same instruction with real scales (the MX mode of fp8gemm): `sa` and `sb` are the
// lane's four ue8m0 bytes for four consecutive 32-wide k-blocks of its row of A and its
// column of B, and BID (an immediate) picks the byte for this k32 step. With thread-id 0 the
// hardware reads, for a lane in quad g (g = lane / 4): row g of A from lane 4g, row g+8 from
// lane 4g+1, column g of B from lane 4g; the other lanes' registers are ignored (thread-id 1
// would read lanes 4g+2 / 4g+3 for A and 4g+1 for B instead). Measured with a probe of
// distinct bytes in every lane, docs/design/fp8gemm.md.
template <int BID>
__device__ __forceinline__ void mma_e4m3_16832_mx(float (&d)[4], const unsigned (&a)[4],
                                                  const unsigned (&b)[2], unsigned sa,
                                                  unsigned sb) {
    static_assert(BID >= 0 && BID < 4, "byte-id selects one of the four ue8m0 bytes");
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, 0}, %11, {%12, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb),
          "n"(BID));
}
#else
#define SPARK_HAS_MX_MMA 0
__device__ __forceinline__ void mma_e4m3_16832(float (&d)[4], const unsigned (&a)[4],
                                               const unsigned (&b)[2]) {
    mma_e4m3_16832_plain(d, a, b);
}
// No block-scaled instruction on this target: the host refuses MX mode (fp8gemm_mx_available)
// and this body is never reached.
template <int BID>
__device__ __forceinline__ void mma_e4m3_16832_mx(float (&d)[4], const unsigned (&a)[4],
                                                  const unsigned (&b)[2], unsigned, unsigned) {
    mma_e4m3_16832_plain(d, a, b);
}
#endif

// D[16x8] (+)= A[16x64] * B[64x8] in e2m1 (fp4) with block scales, fp32 accumulate: the
// kind::mxf4nvf4 instruction of sm_120a, the fp4gemm kernels' only mma. The operands are the
// m16n8k32 fp8 fragments with every byte holding two e2m1 values, lowest k in the low nibble:
// a[4] = (row g / g+8) x (k 8c..8c+7 / 32+8c..32+8c+7), b[2] = (k 8c..8c+7 / 32+8c..
// 32+8c+7) x col g, g = lane / 4, c = lane % 4, so the byte-level loads of the fp8 GEMM feed
// it unchanged (measured with a probe, docs/design/fp4gemm.md).
//
// NVFP4 (scale_vec::4X, ue4m3): each lane's scale register holds four e4m3 scale bytes, byte
// i scaling k 16i..16i+15 of the lane's row of A or column of B. Which lanes the hardware
// reads is set by the thread-id immediate: TA = 0 reads row g from lane 4g and row g+8 from
// lane 4g+1, TA = 1 reads them from lanes 4g+2 and 4g+3; TB = 0 reads column g from lane 4g,
// TB = 1 from lane 4g+1. Byte-id must be 0. So one register serves two m16 tiles of A (the
// quad's lanes 0-1 hold one tile's rows, lanes 2-3 the other's) or two n8 tiles of B.
//
// MXFP4 (scale_vec::2X, ue8m0): two scale bytes per lane per k64, byte BID + i scaling k
// 32i..32i+31, BID 0 or 2; the same lane selection.
#if SPARK_HAS_MX_MMA
template <int TA, int TB>
__device__ __forceinline__ void mma_e2m1_16864_nvf4(float (&d)[4], const unsigned (&a)[4],
                                                    const unsigned (&b)[2], unsigned sa,
                                                    unsigned sb) {
    static_assert((TA == 0 || TA == 1) && (TB == 0 || TB == 1), "thread-id is 0 or 1");
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, %12}, %11, {0, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb),
          "n"(TA), "n"(TB));
}
template <int BID, int TA, int TB>
__device__ __forceinline__ void mma_e2m1_16864_mxf4(float (&d)[4], const unsigned (&a)[4],
                                                    const unsigned (&b)[2], unsigned sa,
                                                    unsigned sb) {
    static_assert(BID == 0 || BID == 2, "byte-id selects the low or the high byte pair");
    static_assert((TA == 0 || TA == 1) && (TB == 0 || TB == 1), "thread-id is 0 or 1");
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, "
        "{%12, %14};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb),
          "n"(BID), "n"(TA), "n"(TB));
}
#else
// No fp4 instruction on this target: the host refuses fp4gemm (fp4gemm_available) and these
// bodies are never reached.
template <int TA, int TB>
__device__ __forceinline__ void mma_e2m1_16864_nvf4(float (&)[4], const unsigned (&)[4],
                                                    const unsigned (&)[2], unsigned, unsigned) {
    __trap();
}
template <int BID, int TA, int TB>
__device__ __forceinline__ void mma_e2m1_16864_mxf4(float (&)[4], const unsigned (&)[4],
                                                    const unsigned (&)[2], unsigned, unsigned) {
    __trap();
}
#endif

// ---------------------------------------------------------------------------
// TMA / mbarrier (sm_90+; available on sm_120 / sm_121). Used by the hgemm TMA variant.
//
// A TMA load (cp.async.bulk.tensor) is issued by one thread: it names a tensor map (a 128-byte
// descriptor of a global tensor and a box shape, encoded on the host), a box coordinate and a
// shared-memory destination, and the copy engine writes the whole box into smem, applying the
// map's swizzle. Completion is signalled on an mbarrier: the issuing thread first tells the
// barrier how many bytes to expect (arrive.expect_tx), the copy engine counts them down as
// they land (complete_tx), and the phase completes when the arrival count and the byte count
// are both satisfied. Consumers wait on the phase parity: a barrier alternates between phase 0
// and phase 1 every time it completes, and try_wait.parity P returns once the phase with
// parity P has completed, so the n-th use of a barrier waits on parity n & 1.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
// Initialize a barrier that completes a phase after `count` arrivals (plus any expected bytes).
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count)
                 : "memory");
}
// Make the initialized barriers visible to the async proxy (the TMA unit) before any
// complete_tx can target them. Call once after mbar_init, before the __syncthreads.
__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
// Orders this thread's generic-proxy shared-memory accesses against later async-proxy ones
// (a TMA reading smem that threads wrote, or a TMA overwriting smem that threads read).
__device__ __forceinline__ void fence_proxy_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
// One arrival that also registers `bytes` of transactions the phase must see before it completes.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
                 "r"(bytes)
                 : "memory");
}
// Plain arrival (release semantics: this thread's earlier smem reads are ordered before it).
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
// Spin until the phase with parity `parity` has completed (acquire semantics).
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "SPARK_MBAR_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra SPARK_MBAR_DONE;\n"
        "bra SPARK_MBAR_WAIT;\n"
        "SPARK_MBAR_DONE:\n"
        "}\n" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}
// 2-D TMA load: the box of `map` at coordinates (c0 along the innermost dimension, c1 along the
// outer one), in elements, into `smem_dst`, completion counted on `bar`. `map` must point at a
// kernel parameter declared `const __grid_constant__ CUtensorMap`. The destination must be
// aligned to the swizzle span (1 KB for the 128-byte swizzle, 512 B for the 64-byte one).
__device__ __forceinline__ void tma_load_2d(void* smem_dst, const CUtensorMap* map, uint64_t* bar,
                                            int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n" ::"r"(smem_u32(smem_dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
        : "memory");
}
// 3-D form: coordinates (c0, c1, c2) with c0 along the innermost dimension. Used by the
// attention kernel, whose tensors are [B*H][S][D] and whose tiles must not run from one head
// into the next: a box that hangs off the end of the S dimension is zero-filled instead.
__device__ __forceinline__ void tma_load_3d(void* smem_dst, const CUtensorMap* map, uint64_t* bar,
                                            int c0, int c1, int c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];\n" ::"r"(smem_u32(smem_dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2)
        : "memory");
}
// 1-D bulk copy global -> shared through the copy engine (no tensor map), completion counted
// on `bar` like a TMA box. `bytes` a multiple of 16, both addresses 16-byte aligned.
__device__ __forceinline__ void bulk_load(void* smem_dst, const void* gmem_src, unsigned bytes,
                                          uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, "
        "[%3];\n" ::"r"(smem_u32(smem_dst)),
        "l"(gmem_src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}
// Pull the descriptor into the TMA unit's cache ahead of the first load.
__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
// bar.sync on a named barrier for `nthreads` threads: lets a subset of the block (the consumer
// warps of a warp-specialized kernel) synchronize without the rest of it.
__device__ __forceinline__ void named_barrier_sync(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(nthreads) : "memory");
}
// bar.arrive: counts toward the same barrier without waiting on it. A barrier of `nthreads`
// completes once syncs plus arrives reach that count, so 128 threads that arrive and 128 that
// sync on a 256-thread barrier release the 128 that sync: the hand-off between the two
// consumer groups of the ping-pong attention kernel.
__device__ __forceinline__ void named_barrier_arrive(int id, int nthreads) {
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(nthreads) : "memory");
}

// Host side: encode a tensor map for a row-major 2-D bf16 matrix (rows x cols, cols
// contiguous) with a box of box_rows x box_cols. The driver entry point is fetched through
// the runtime, so nothing links against libcuda. Out-of-range boxes are zero-filled by the
// copy engine and still count their full byte size toward the barrier.
using TensorMapEncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                       const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                       const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                       CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
inline TensorMapEncodeFn tensor_map_encoder() {
    static TensorMapEncodeFn fn = nullptr;
    if (fn == nullptr) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
        SPARK_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000,
                                                          cudaEnableDefault, &q));
        if (p == nullptr || q != cudaDriverEntryPointSuccess) {
            throw std::runtime_error("cuTensorMapEncodeTiled is not available in this driver");
        }
        fn = reinterpret_cast<TensorMapEncodeFn>(p);
    }
    return fn;
}
inline CUtensorMap make_tensor_map_2d(CUtensorMapDataType type, size_t elem_bytes, const void* base,
                                      uint64_t rows, uint64_t cols, uint32_t box_rows,
                                      uint32_t box_cols, CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    const cuuint64_t dims[2] = {cols, rows};
    const cuuint64_t strides[1] = {cols * elem_bytes};  // bytes, outer dim only
    const cuuint32_t box[2] = {box_cols, box_rows};
    const cuuint32_t elem_strides[2] = {1, 1};
    const CUresult r =
        tensor_map_encoder()(&map, type, 2, const_cast<void*>(base), dims, strides, box,
                             elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        throw std::runtime_error("cuTensorMapEncodeTiled failed with CUresult " +
                                 std::to_string(static_cast<int>(r)));
    }
    return map;
}
inline CUtensorMap make_tensor_map_2d_bf16(const void* base, uint64_t rows, uint64_t cols,
                                           uint32_t box_rows, uint32_t box_cols,
                                           CUtensorMapSwizzle swizzle) {
    return make_tensor_map_2d(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, sizeof(__nv_bfloat16), base, rows,
                              cols, box_rows, box_cols, swizzle);
}
// Same for a matrix of 8-bit elements (the fp8gemm TMA variant moves e4m3 as raw bytes).
inline CUtensorMap make_tensor_map_2d_u8(const void* base, uint64_t rows, uint64_t cols,
                                         uint32_t box_rows, uint32_t box_cols,
                                         CUtensorMapSwizzle swizzle) {
    return make_tensor_map_2d(CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, base, rows, cols, box_rows,
                              box_cols, swizzle);
}

// 3-D map for a [d2][d1][d0] tensor (d0 contiguous) with a box of box1 x box0 in the two
// inner dimensions and 1 in the outer one, so a box never crosses from one d2 slice into the
// next and rows past d1 are zero-filled.
inline CUtensorMap make_tensor_map_3d(CUtensorMapDataType type, size_t elem_bytes, const void* base,
                                      uint64_t d0, uint64_t d1, uint64_t d2, uint32_t box0,
                                      uint32_t box1, CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    const cuuint64_t dims[3] = {d0, d1, d2};
    const cuuint64_t strides[2] = {d0 * elem_bytes, d1 * d0 * elem_bytes};
    const cuuint32_t box[3] = {box0, box1, 1};
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    const CUresult r =
        tensor_map_encoder()(&map, type, 3, const_cast<void*>(base), dims, strides, box,
                             elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        throw std::runtime_error("cuTensorMapEncodeTiled (3-D) failed with CUresult " +
                                 std::to_string(static_cast<int>(r)));
    }
    return map;
}
inline CUtensorMap make_tensor_map_3d_bf16(const void* base, uint64_t d0, uint64_t d1, uint64_t d2,
                                           uint32_t box0, uint32_t box1,
                                           CUtensorMapSwizzle swizzle) {
    return make_tensor_map_3d(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, sizeof(__nv_bfloat16), base, d0, d1,
                              d2, box0, box1, swizzle);
}
// Same for 8-bit elements (the fp8 attention kernel moves e4m3 as raw bytes).
inline CUtensorMap make_tensor_map_3d_u8(const void* base, uint64_t d0, uint64_t d1, uint64_t d2,
                                         uint32_t box0, uint32_t box1, CUtensorMapSwizzle swizzle) {
    return make_tensor_map_3d(CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, base, d0, d1, d2, box0, box1,
                              swizzle);
}

}  // namespace spark
