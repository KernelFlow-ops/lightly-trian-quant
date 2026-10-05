// K1: fused residual-add + LayerNorm.  One warp per row, 128-bit vector loads/stores.
//   s = x + r        -> s_out (new residual stream; stored in x's dtype: bf16 or fp32)
//   y = LN(s)*w + b  -> y_out (bf16; fp32 statistics, eps given)
// x / r may each be bf16 or fp32 (fp32 residual stream = what torch autocast does).
// With bf16 x the sum is rounded to bf16 before the statistics (matches the eager bf16 add).
// NV = d / 256 -> 8 elements per lane per step, coalesced across the warp.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* o) {
    uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(q[j]); o[2 * j] = f.x; o[2 * j + 1] = f.y; }
}
__device__ __forceinline__ void load8(const float* p, float* o) {
    float4 a = __ldg(reinterpret_cast<const float4*>(p));
    float4 b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ void store8(__nv_bfloat16* p, const float* v) {
    uint4 u;
    __nv_bfloat162* q = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) q[j] = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
    *reinterpret_cast<uint4*>(p) = u;
}
__device__ __forceinline__ void store8(float* p, const float* v) {
    reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}
__device__ __forceinline__ float rnd(__nv_bfloat16*, float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float rnd(float*, float v) { return v; }

template <int NV, typename TX, typename TR, bool HAS_R>
__global__ void __launch_bounds__(128)
add_ln_kernel(const TX* __restrict__ x, const TR* __restrict__ r, const float* __restrict__ w,
              const float* __restrict__ b, TX* __restrict__ s_out, __nv_bfloat16* __restrict__ y_out,
              int M, float eps) {
    constexpr int D = NV * 256;
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= M) return;
    float v[NV][8];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < NV; i++) {
        size_t e = (size_t)row * D + (i * 32 + lane) * 8;
        load8(x + e, v[i]);
        if (HAS_R) {
            float t[8];
            load8(r + e, t);
#pragma unroll
            for (int j = 0; j < 8; j++) v[i][j] = rnd((TX*)nullptr, v[i][j] + t[j]);
            store8(s_out + e, v[i]);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) sum += v[i][j];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    float mean = sum * (1.f / D);
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < NV; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) { float t = v[i][j] - mean; sq += t * t; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, o);
    float rstd = rsqrtf(sq * (1.f / D) + eps);
#pragma unroll
    for (int i = 0; i < NV; i++) {
        size_t e = (size_t)row * D + (i * 32 + lane) * 8;
        float ww[8], bb[8], y[8];
        load8(w + (i * 32 + lane) * 8, ww);
        load8(b + (i * 32 + lane) * 8, bb);
#pragma unroll
        for (int j = 0; j < 8; j++) y[j] = (v[i][j] - mean) * rstd * ww[j] + bb[j];
        store8(y_out + e, y);
    }
}

template <int NV, typename TX, typename TR, bool HAS_R>
static void launch(const at::Tensor& x, const c10::optional<at::Tensor>& r, const at::Tensor& w,
                   const at::Tensor& b, at::Tensor& s, at::Tensor& y, double eps) {
    int M = x.size(0);
    dim3 grid((M + 3) / 4), block(128);
    add_ln_kernel<NV, TX, TR, HAS_R><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const TX*>(x.data_ptr()),
        HAS_R ? reinterpret_cast<const TR*>(r->data_ptr()) : nullptr,
        reinterpret_cast<const float*>(w.data_ptr()), reinterpret_cast<const float*>(b.data_ptr()),
        reinterpret_cast<TX*>(s.data_ptr()), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, (float)eps);
}

template <typename TX, typename TR, bool HAS_R>
static void dispatch_d(int d, const at::Tensor& x, const c10::optional<at::Tensor>& r, const at::Tensor& w,
                       const at::Tensor& b, at::Tensor& s, at::Tensor& y, double eps) {
    if (d == 1024) launch<4, TX, TR, HAS_R>(x, r, w, b, s, y, eps);
    else launch<3, TX, TR, HAS_R>(x, r, w, b, s, y, eps);
}

using bf = __nv_bfloat16;

// returns (s, y).  If r is None, s is x itself (no copy) and only LN is computed.
std::vector<at::Tensor> add_layernorm(at::Tensor x, c10::optional<at::Tensor> r, at::Tensor w,
                                      at::Tensor b, double eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous cuda");
    TORCH_CHECK(x.dtype() == at::kBFloat16 || x.dtype() == at::kFloat, "x must be bf16 or fp32");
    int d = x.size(-1);
    TORCH_CHECK(d == 1024 || d == 768, "d must be 1024 or 768");
    TORCH_CHECK(w.dtype() == at::kFloat && b.dtype() == at::kFloat && w.is_contiguous() && b.is_contiguous(), "w,b must be contiguous fp32");
    auto x2 = x.view({-1, d});
    at::Tensor s = r.has_value() ? at::empty_like(x2) : x2;
    at::Tensor y = at::empty(x2.sizes(), x2.options().dtype(at::kBFloat16));
    bool xb = x.dtype() == at::kBFloat16;
    if (r.has_value()) {
        TORCH_CHECK(r->is_contiguous() && r->numel() == x.numel(), "r must be contiguous, same shape");
        bool rb = r->dtype() == at::kBFloat16;
        TORCH_CHECK(rb || r->dtype() == at::kFloat, "r must be bf16 or fp32");
        if (xb && rb) dispatch_d<bf, bf, true>(d, x2, r, w, b, s, y, eps);
        else if (xb && !rb) dispatch_d<bf, float, true>(d, x2, r, w, b, s, y, eps);
        else if (!xb && rb) dispatch_d<float, bf, true>(d, x2, r, w, b, s, y, eps);
        else dispatch_d<float, float, true>(d, x2, r, w, b, s, y, eps);
    } else {
        if (xb) dispatch_d<bf, bf, false>(d, x2, r, w, b, s, y, eps);
        else dispatch_d<float, float, false>(d, x2, r, w, b, s, y, eps);
    }
    return {s.view_as(x), y.view(x.sizes())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add_layernorm", &add_layernorm); }
