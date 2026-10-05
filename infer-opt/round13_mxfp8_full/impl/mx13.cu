// Round 13: MXFP8 (E4M3 + UE8M0 per 1x32 along K, RCEIL) helper kernels for the whole-model variant r13_mx8a (cuBLASLt `_scaled_mm` GEMMs).
//
// Scale layout = cuBLAS / CUTLASS blocked layout (the one of train-opt r17 `quant_dual`, round-6 `mx_ext.cu`): scales of a [R, K] operand live in
//   sf[(R/128), (K/128), 32, 4, 4] u8, byte offset = ((row/128)*(K/128) + kb/4)*512 + (row%32)*16 + ((row/32)%4)*4 + kb%4,  kb = k/32.
// Scale rule (bit-identical to r17 / round10 quant_ref.mx8_quantize): byte = clamp(biased_exp(fl32(amax/448)) + (mantissa != 0), 255), data = cvt.rn.satfinite.e4m3(x * 2^-(byte-127)).
//
//   ln_mx8       fp32 residual [M,d] -> LayerNorm -> MX8 (q [M,d], sf)        (one warp per row; d = 768 | 1024)
//   quant_rows   fp16 / bf16 / fp32 [M,K] -> MX8                                (one warp per row; K % 256 == 0)
//   gelu_quant   bf16 [M,K] (fc1 GEMM output) + fp32 bias -> GELU_erf -> MX8     (fc2 operand)
//   epi_rope     bf16 y [M,3d] + bias + RoPE -> fp16 [3,B,N,H,64]                (qkv epilogue pass, layout of r12 attention)
//   res_ln_mx8   epi_res + the next LayerNorm + quant in one pass (writes the fp32 residual and the MX8 operand)
//   epi_res      bf16 y [M,N] -> fp32 out = res + gamma * (y + bias)             (proj / fc2 epilogue pass)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu

// ---- programmatic dependent launch: every kernel of this file triggers its dependents at entry and waits for its producer before touching data
//      (both instructions are no-ops when the kernel was launched without the attribute)
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ void load8(const float* p, float* o) {
    float4 a = __ldg(reinterpret_cast<const float4*>(p));
    float4 b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* o) {
    uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void load8(const __half* p, float* o) {
    uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void store8h(__half* p, const float* v) {
    uint4 u;
    __half2* q = reinterpret_cast<__half2*>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) q[j] = __floats2half2_rn(v[2 * j], v[2 * j + 1]);
    *reinterpret_cast<uint4*>(p) = u;
}
__device__ __forceinline__ void store8f(float* p, const float* v) {
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// ---- MX8 scalar pieces (same arithmetic as train-opt r17 mx8_kernels.cu)
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
__device__ __forceinline__ uint2 quant8(const float* v, uint32_t s) {
    uint2 o;
    if (s >= 255u) { o.x = 0x7f7f7f7fu; o.y = 0x7f7f7f7fu; return o; }
    const float inv = inv_scale_of(s);
    o.x = cvt_e4m3x2(v[0] * inv, v[1] * inv) | (cvt_e4m3x2(v[2] * inv, v[3] * inv) << 16);
    o.y = cvt_e4m3x2(v[4] * inv, v[5] * inv) | (cvt_e4m3x2(v[6] * inv, v[7] * inv) << 16);
    return o;
}
__device__ __forceinline__ size_t sf_off(int row, int kb, int rk) {
    return ((size_t)((row >> 7) * rk + (kb >> 2)) << 9) + ((row & 31) << 4) + (((row >> 5) & 3) << 2) + (kb & 3);
}
// 8 values per lane, 4 consecutive lanes = one 32-element block: amax over the quad, scale byte, packed e4m3 bytes
__device__ __forceinline__ uint2 mx8_block(const float* v, uint32_t& s) {
    uint32_t a = 0u;
#pragma unroll
    for (int j = 0; j < 8; j++) a = max(a, __float_as_uint(fabsf(v[j])));
    a = max(a, __shfl_xor_sync(FULL_MASK, a, 1));
    a = max(a, __shfl_xor_sync(FULL_MASK, a, 2));
    s = rceil_scale_byte(a);
    return quant8(v, s);
}

// ------------------------------------------------------------------------------------------------ LN + quant
template <int NV>
__global__ void __launch_bounds__(128)
ln_mx8_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int M, float eps) {
    constexpr int D = NV * 256;
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    pdl_trigger();
    pdl_wait();
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
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(FULL_MASK, sum, o);
    float mean = sum * (1.f / D);
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < NV; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) { float t = v[i][j] - mean; sq += t * t; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(FULL_MASK, sq, o);
    float rstd = rsqrtf(sq * (1.f / D) + eps);
#pragma unroll
    for (int i = 0; i < NV; i++) {
        float ww[8], bb[8], y[8];
        int k0 = (i * 32 + lane) * 8;
        load8(w + k0, ww);
        load8(b + k0, bb);
#pragma unroll
        for (int j = 0; j < 8; j++) y[j] = (v[i][j] - mean) * rstd * ww[j] + bb[j];
        uint32_t s;
        uint2 o = mx8_block(y, s);
        *reinterpret_cast<uint2*>(q + (size_t)row * D + k0) = o;
        if ((lane & 3) == 0) sf[sf_off(row, k0 >> 5, D >> 7)] = (uint8_t)s;
    }
}


// residual add of the previous GEMM + LN + quant in one pass:  x_out = res + gamma*(y + bias);  q, sf = MX8(LN(x_out))   (one warp per row; d = 768 | 1024)
template <int NV>
__global__ void __launch_bounds__(128)
res_ln_mx8_kernel(const __nv_bfloat16* __restrict__ y, const float* __restrict__ res, const float* __restrict__ gamma, const float* __restrict__ bias,
                  const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ x_out, uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int M, float eps) {
    constexpr int D = NV * 256;
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    pdl_trigger();
    pdl_wait();
    if (row >= M) return;
    float v[NV][8];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < NV; i++) {
        int k0 = (i * 32 + lane) * 8;
        float yy[8], rr[8], gg[8], bb[8];
        load8(y + (size_t)row * D + k0, yy);
        load8(res + (size_t)row * D + k0, rr);
        load8(gamma + k0, gg);
        load8(bias + k0, bb);
#pragma unroll
        for (int j = 0; j < 8; j++) { v[i][j] = rr[j] + gg[j] * (yy[j] + bb[j]); sum += v[i][j]; }
        store8f(x_out + (size_t)row * D + k0, v[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(FULL_MASK, sum, o);
    float mean = sum * (1.f / D);
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < NV; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) { float t = v[i][j] - mean; sq += t * t; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(FULL_MASK, sq, o);
    float rstd = rsqrtf(sq * (1.f / D) + eps);
#pragma unroll
    for (int i = 0; i < NV; i++) {
        float ww[8], bb[8], yv[8];
        int k0 = (i * 32 + lane) * 8;
        load8(w + k0, ww);
        load8(b + k0, bb);
#pragma unroll
        for (int j = 0; j < 8; j++) yv[j] = (v[i][j] - mean) * rstd * ww[j] + bb[j];
        uint32_t s;
        uint2 o = mx8_block(yv, s);
        *reinterpret_cast<uint2*>(q + (size_t)row * D + k0) = o;
        if ((lane & 3) == 0) sf[sf_off(row, k0 >> 5, D >> 7)] = (uint8_t)s;
    }
}

// ------------------------------------------------------------------------------------------------ generic row quantiser (+ optional bias+GELU)
template <typename T, bool GELU>
__global__ void __launch_bounds__(128)
quant_rows_k(const T* __restrict__ x, const float* __restrict__ bias, uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int M, int K) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    pdl_trigger();
    pdl_wait();
    if (row >= M) return;
    const T* xr = x + (size_t)row * K;
    const int rk = K >> 7;
#pragma unroll 4
    for (int c = 0; c < (K >> 8); c++) {
        int k0 = (c * 32 + lane) * 8;
        float v[8];
        load8(xr + k0, v);
        if (GELU) {
            float bb[8];
            load8(bias + k0, bb);
#pragma unroll
            for (int j = 0; j < 8; j++) { float t = v[j] + bb[j]; v[j] = t * 0.5f * (1.0f + erff(t * 0.7071067811865476f)); }
        }
        uint32_t s;
        uint2 o = mx8_block(v, s);
        *reinterpret_cast<uint2*>(q + (size_t)row * K + k0) = o;
        if ((lane & 3) == 0) sf[sf_off(row, k0 >> 5, rk)] = (uint8_t)s;
    }
}

// ------------------------------------------------------------------------------------------------ epilogue passes (cuBLASLt writes bf16 y without epilogue)
// one thread per (row, which, head, 8-wide quarter of the low half); partners c and c+32 (RoPE rotates halves)
__global__ void __launch_bounds__(256)
epi_rope_k(const __nv_bfloat16* __restrict__ y, const float* __restrict__ bias, const float* __restrict__ sin_t, const float* __restrict__ cos_t,
           __half* __restrict__ out, int M, int Ntok, int prefix, int d, int H) {
    pdl_trigger();
    pdl_wait();
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)M * 3 * H * 4;
    if (t >= total) return;
    int c = (t & 3) * 8; long r = t >> 2;
    int h = r % H; r /= H;
    int which = r % 3; int m = (int)(r / 3);
    int col = which * d + h * 64 + c;
    const __nv_bfloat16* yp = y + (size_t)m * 3 * d + col;
    float lo[8], hi[8], bl[8], bh[8];
    load8(yp, lo); load8(yp + 32, hi);
    load8(bias + col, bl); load8(bias + col + 32, bh);
#pragma unroll
    for (int j = 0; j < 8; j++) { lo[j] += bl[j]; hi[j] += bh[j]; }
    int n = m % Ntok, pp = n - prefix;
    if (which < 2 && pp >= 0) {
        float cl[8], sl[8], ch[8], sh[8];
        load8(cos_t + (size_t)pp * 64 + c, cl); load8(sin_t + (size_t)pp * 64 + c, sl);
        load8(cos_t + (size_t)pp * 64 + c + 32, ch); load8(sin_t + (size_t)pp * 64 + c + 32, sh);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float nl = lo[j] * cl[j] - hi[j] * sl[j], nh = hi[j] * ch[j] + lo[j] * sh[j];
            lo[j] = nl; hi[j] = nh;
        }
    }
    int B = M / Ntok, b = m / Ntok;
    size_t o = ((((size_t)which * B + b) * Ntok + n) * H + h) * 64 + c;
    store8h(out + o, lo);
    store8h(out + o + 32, hi);
}

__global__ void __launch_bounds__(256)
epi_res_k(const __nv_bfloat16* __restrict__ y, const float* __restrict__ res, const float* __restrict__ gamma, const float* __restrict__ bias,
          float* __restrict__ out, long n8, int N) {
    pdl_trigger();
    pdl_wait();
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n8) return;
    long i = t * 8;
    int n = (int)(i % N);
    float v[8], r[8], g[8], b[8], o[8];
    load8(y + i, v); load8(res + i, r); load8(gamma + n, g); load8(bias + n, b);
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = r[j] + g[j] * (v[j] + b[j]);
    store8f(out + i, o);
}

// ------------------------------------------------------------------------------------------------ host side
#define CHECK_MX(x, K, q, sf) \
    TORCH_CHECK((x).is_cuda() && (x).dim() == 2 && (x).is_contiguous(), "x must be a contiguous CUDA matrix"); \
    TORCH_CHECK((K) % 256 == 0, "K must be a multiple of 256"); \
    TORCH_CHECK((q).scalar_type() == at::kByte && (q).is_contiguous() && (q).numel() >= (x).size(0) * (K), "q buffer"); \
    TORCH_CHECK((sf).scalar_type() == at::kByte && (sf).is_contiguous() && (sf).numel() >= (((x).size(0) + 127) / 128) * ((K) / 128) * 512, "sf buffer")

static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

static bool g_pdl = true;
void set_pdl(bool on) { g_pdl = on; }
template <typename... KArgs, typename... Args>
static void launch_k(void (*k)(KArgs...), dim3 grid, dim3 block, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = g_pdl ? 1 : 0;
    cudaError_t e = cudaLaunchKernelEx(&cfg, k, static_cast<KArgs>(args)...);
    TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
}

void ln_mx8(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor q, at::Tensor sf, double eps) {
    c10::cuda::CUDAGuard g(x.device());
    int d = x.size(-1);
    TORCH_CHECK(x.dtype() == at::kFloat && x.is_contiguous() && w.dtype() == at::kFloat && b.dtype() == at::kFloat, "x, w, b fp32 contiguous");
    TORCH_CHECK(d == 1024 || d == 768, "d must be 1024 or 768");
    auto x2 = x.view({-1, d});
    int M = x2.size(0);
    CHECK_MX(x2, d, q, sf);
    if (d == 1024) launch_k(ln_mx8_kernel<4>, dim3((M + 3) / 4), dim3(128), x2.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), q.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(), M, (float)eps);
    else launch_k(ln_mx8_kernel<3>, dim3((M + 3) / 4), dim3(128), x2.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), q.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(), M, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool GELU>
void quant_impl(at::Tensor x, at::Tensor bias, at::Tensor q, at::Tensor sf) {
    c10::cuda::CUDAGuard g(x.device());
    int K = x.size(-1);
    auto x2 = x.view({-1, K});
    int M = x2.size(0);
    CHECK_MX(x2, K, q, sf);
    uint8_t* pq = q.data_ptr<uint8_t>();
    uint8_t* ps = sf.data_ptr<uint8_t>();
    const float* pb = GELU ? bias.data_ptr<float>() : nullptr;
    dim3 grid((M + 3) / 4);
    if (x.scalar_type() == at::kHalf) launch_k(quant_rows_k<__half, GELU>, dim3(grid), dim3(128), reinterpret_cast<const __half*>(x2.data_ptr()), pb, pq, ps, M, K);
    else if (x.scalar_type() == at::kBFloat16) launch_k(quant_rows_k<__nv_bfloat16, GELU>, dim3(grid), dim3(128), reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr()), pb, pq, ps, M, K);
    else if (x.scalar_type() == at::kFloat) launch_k(quant_rows_k<float, GELU>, dim3(grid), dim3(128), x2.data_ptr<float>(), pb, pq, ps, M, K);
    else TORCH_CHECK(false, "x must be fp16 / bf16 / fp32");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void quant_rows(at::Tensor x, at::Tensor q, at::Tensor sf) { quant_impl<false>(x, at::Tensor(), q, sf); }
void gelu_quant(at::Tensor y, at::Tensor bias, at::Tensor q, at::Tensor sf) { quant_impl<true>(y, bias, q, sf); }

// y bf16 [B*Ntok, 3d] -> fp16 [3, B, Ntok, H, 64]
at::Tensor epi_rope(at::Tensor y, at::Tensor bias, at::Tensor sin_t, at::Tensor cos_t, int64_t Ntok, int64_t prefix, int64_t d, int64_t H) {
    c10::cuda::CUDAGuard g(y.device());
    TORCH_CHECK(y.dtype() == at::kBFloat16 && y.is_contiguous() && y.size(1) == 3 * d && d == H * 64, "y bf16 [M,3d]");
    TORCH_CHECK(bias.dtype() == at::kFloat && sin_t.dtype() == at::kFloat && cos_t.dtype() == at::kFloat, "fp32 bias / sin / cos");
    int M = y.size(0);
    int64_t B = M / Ntok;
    at::Tensor out = at::empty({3, B, Ntok, H, 64}, y.options().dtype(at::kHalf));
    long total = (long)M * 3 * H * 4;
    launch_k(epi_rope_k, dim3((total + 255) / 256), dim3(256), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), bias.data_ptr<float>(), sin_t.data_ptr<float>(), cos_t.data_ptr<float>(),
                                                          reinterpret_cast<__half*>(out.data_ptr()), M, Ntok, prefix, d, H);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor epi_res(at::Tensor y, at::Tensor res, at::Tensor gamma, at::Tensor bias) {
    c10::cuda::CUDAGuard g(y.device());
    TORCH_CHECK(y.dtype() == at::kBFloat16 && y.is_contiguous() && res.dtype() == at::kFloat && res.is_contiguous(), "y bf16, res fp32");
    int N = y.size(1);
    TORCH_CHECK(N % 8 == 0 && res.numel() == y.numel(), "shape");
    at::Tensor out = at::empty_like(res);
    long n8 = y.numel() / 8;
    launch_k(epi_res_k, dim3((n8 + 255) / 256), dim3(256), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), res.data_ptr<float>(), gamma.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), n8, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


at::Tensor res_ln_mx8(at::Tensor y, at::Tensor res, at::Tensor gamma, at::Tensor bias, at::Tensor w, at::Tensor b, at::Tensor q, at::Tensor sf, double eps) {
    c10::cuda::CUDAGuard g(y.device());
    int d = res.size(-1);
    TORCH_CHECK(y.dtype() == at::kBFloat16 && y.is_contiguous() && res.dtype() == at::kFloat && res.is_contiguous() && y.numel() == res.numel(), "y bf16, res fp32");
    TORCH_CHECK(d == 1024 || d == 768, "d must be 1024 or 768");
    auto r2 = res.view({-1, d});
    int M = r2.size(0);
    CHECK_MX(r2, d, q, sf);
    at::Tensor out = at::empty_like(res);
    const __nv_bfloat16* py = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr());
    if (d == 1024) launch_k(res_ln_mx8_kernel<4>, dim3((M + 3) / 4), dim3(128), py, r2.data_ptr<float>(), gamma.data_ptr<float>(), bias.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), q.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(), M, (float)eps);
    else launch_k(res_ln_mx8_kernel<3>, dim3((M + 3) / 4), dim3(128), py, r2.data_ptr<float>(), gamma.data_ptr<float>(), bias.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), q.data_ptr<uint8_t>(), sf.data_ptr<uint8_t>(), M, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("res_ln_mx8", &res_ln_mx8);
    m.def("set_pdl", &set_pdl);
    m.def("ln_mx8", &ln_mx8);
    m.def("quant_rows", &quant_rows);
    m.def("gelu_quant", &gelu_quant);
    m.def("epi_rope", &epi_rope);
    m.def("epi_res", &epi_res);
}
