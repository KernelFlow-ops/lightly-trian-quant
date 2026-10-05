// Round 14 step 2: FP4 activation producers.
//   lnq   : LayerNorm of the fp32 residual stream (round12 ln16.cu arithmetic: one warp per row, same statistics / order) fused with the FP4 block quantiser
//           -> packed e2m1 q [M, D/2] + blocked scales sf [ceil(M/128), D/(4 vec), 32, 4, 4]  (the A operand of the own GEMM kernel)
//   quantx: stand-alone row quantiser for fp16 / bf16 / fp32 input of any K % 256 == 0 (attention output -> proj, head ConvT inputs)
// Formats / rounding are the fp4_ref.py spec (bit-exact, see tests/t_quant14.py):
//   nv4: sf = e4m3_RN(min(((amax_blk * (1/6)) / ts), 448)), v = x / (e4m3(sf) * ts) (IEEE division; block with sf == 0 -> data 0), ts = static device scalar
//   mx4: byte = min(exp(amax_blk * (1/6)) + (mantissa != 0), 254) (RCEIL), v = x * 2^(127 - byte), zero block -> byte 0 / data 0
// e2m1: cvt.rn.satfinite.e2m1x2.f32 (RN-even, saturating at 6).  The per-tensor scale is static (calibrated on the first eager call), so no row-global pass is needed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr float RCP6 = 1.0f / 6.0f;

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
// 8 floats -> 8 packed e2m1 (v[0] in the low nibble of byte 0)
__device__ __forceinline__ uint32_t pack8(const float* v) {
    uint32_t out;
    asm("{\n.reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n}"
        : "=r"(out)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return out;
}
__device__ __forceinline__ uint32_t sf_blocked_off(int row, int kb, int kt_tiles) {
    return ((uint32_t)(row >> 7) * kt_tiles + (kb >> 2)) * 512u + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (kb & 3);
}

// quantise 8 values whose block amax (over the whole 16 / 32 block) is `amax`; returns the packed word, `sfbyte` = scale byte
template <int FMT>
__device__ __forceinline__ uint32_t quant8(const float* v, float amax, float ts, uint32_t& sfbyte) {
    float s[8];
    if (FMT == 0) {
        float t = __fdiv_rn(amax * RCP6, ts);
        t = fminf(t, 448.f);
        sfbyte = f32_to_e4m3(t);
        const float denom = e4m3_to_f32(sfbyte) * ts;
#pragma unroll
        for (int j = 0; j < 8; j++) s[j] = denom > 0.f ? __fdiv_rn(v[j], denom) : 0.f;
    } else {
        const uint32_t r = __float_as_uint(amax * RCP6);
        const uint32_t e = (r >> 23) & 0xffu, fr = (r & 0x7fffffu) != 0 ? 1u : 0u;
        sfbyte = min(e + fr, 254u);
        const float inv = __uint_as_float((254u - sfbyte) << 23);
#pragma unroll
        for (int j = 0; j < 8; j++) s[j] = v[j] * inv;
    }
    return pack8(s);
}

__device__ __forceinline__ void load8(const float* p, float* o) {
    float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ void load8(const __half* p, float* o) {
    uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* o) {
    uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}

// ---- LN + quant.  D = NV * 256 elements per row, one warp per row, lane holds elements (i * 32 + lane) * 8 .. +7 of iteration i
template <int NV, int FMT>
__global__ void __launch_bounds__(128)
lnq_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, const float* __restrict__ tsp,
           uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int M, float eps) {
    constexpr int D = NV * 256;
    constexpr int KT = FMT == 0 ? D / 64 : D / 128;     // scale tiles along K
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= M) return;
    float v[NV][8];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < NV; i++) {
        load8(x + (size_t)row * D + (i * 32 + lane) * 8, v[i]);
#pragma unroll
        for (int j = 0; j < 8; j++) sum += v[i][j];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    const float mean = sum * (1.f / D);
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < NV; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) { float t = v[i][j] - mean; sq += t * t; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, o);
    const float rstd = rsqrtf(sq * (1.f / D) + eps);
    const float ts = FMT == 0 ? __ldg(tsp) : 1.f;
#pragma unroll
    for (int i = 0; i < NV; i++) {
        float ww[8], bb[8], y[8];
        load8(w + (i * 32 + lane) * 8, ww);
        load8(b + (i * 32 + lane) * 8, bb);
        float am = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) { y[j] = (v[i][j] - mean) * rstd * ww[j] + bb[j]; am = fmaxf(am, fabsf(y[j])); }
        am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 1));
        if (FMT == 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 2));
        uint32_t sfb;
        const uint32_t qw = quant8<FMT>(y, am, ts, sfb);
        const int e0 = (i * 32 + lane) * 8;
        *reinterpret_cast<uint32_t*>(q + (size_t)row * (D / 2) + e0 / 2) = qw;
        if ((lane & (FMT == 0 ? 1 : 3)) == 0) sf[sf_blocked_off(row, e0 / (FMT == 0 ? 16 : 32), KT)] = (uint8_t)sfb;
    }
}

// ---- stand-alone quantiser: any K % 256 == 0 (one warp per row, 8 elements per lane and iteration)
template <typename T, int FMT>
__global__ void __launch_bounds__(128)
quantx_kernel(const T* __restrict__ x, const float* __restrict__ tsp, uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int M, int K) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= M) return;
    const int kt = FMT == 0 ? K / 64 : K / 128;
    const float ts = FMT == 0 ? __ldg(tsp) : 1.f;
    for (int e0 = lane * 8; e0 < K; e0 += 256) {
        float v[8];
        load8(x + (size_t)row * K + e0, v);
        float am = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) am = fmaxf(am, fabsf(v[j]));
        am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 1));
        if (FMT == 1) am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 2));
        uint32_t sfb;
        const uint32_t qw = quant8<FMT>(v, am, ts, sfb);
        *reinterpret_cast<uint32_t*>(q + (size_t)row * (K / 2) + e0 / 2) = qw;
        if ((lane & (FMT == 0 ? 1 : 3)) == 0) sf[sf_blocked_off(row, e0 / (FMT == 0 ? 16 : 32), kt)] = (uint8_t)sfb;
    }
}

static const float* optf(const c10::optional<at::Tensor>& t) { return t.has_value() ? t->data_ptr<float>() : nullptr; }

void lnq(at::Tensor x, at::Tensor w, at::Tensor b, double eps, c10::optional<at::Tensor> ts, at::Tensor q, at::Tensor sf, int64_t fmt) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == at::kFloat, "x: contiguous fp32 cuda");
    const int D = x.size(-1), M = x.numel() / D;
    TORCH_CHECK(D == 1024 || D == 768, "D");
    TORCH_CHECK(fmt == 1 || ts.has_value(), "nv4 needs ts");
    auto st = at::cuda::getCurrentCUDAStream();
    const dim3 grid((M + 3) / 4);
#define L(NV, F) lnq_kernel<NV, F><<<grid, 128, 0, st>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), optf(ts), q.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(), M, (float)eps)
    if (D == 1024) { if (fmt == 0) L(4, 0); else L(4, 1); }
    else { if (fmt == 0) L(3, 0); else L(3, 1); }
#undef L
}

void quantx(at::Tensor x, c10::optional<at::Tensor> ts, at::Tensor q, at::Tensor sf, int64_t fmt) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(1) % 256 == 0, "x: contiguous [M, K%256==0]");
    TORCH_CHECK(fmt == 1 || ts.has_value(), "nv4 needs ts");
    const int M = x.size(0), K = x.size(1);
    auto st = at::cuda::getCurrentCUDAStream();
    const dim3 grid((M + 3) / 4);
#define Q(T, F) quantx_kernel<T, F><<<grid, 128, 0, st>>>((const T*)x.data_ptr(), optf(ts), q.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(), M, K)
    if (x.scalar_type() == at::kHalf) { if (fmt == 0) Q(__half, 0); else Q(__half, 1); }
    else if (x.scalar_type() == at::kBFloat16) { if (fmt == 0) Q(__nv_bfloat16, 0); else Q(__nv_bfloat16, 1); }
    else if (x.scalar_type() == at::kFloat) { if (fmt == 0) Q(float, 0); else Q(float, 1); }
    else TORCH_CHECK(false, "dtype");
#undef Q
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lnq", &lnq);
    m.def("quantx", &quantx);
}
