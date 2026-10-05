// Round 12: LayerNorm of the fp32 residual stream -> fp16 output (A operand of the fp16-accumulate GEMMs).
// Copy-and-modify of round5_gemm_sm120/impl/ln_kernels.cu (K1, the r == None case): same statistics and arithmetic order,
// one warp per row, 128-bit vector loads/stores; the only change is that y is rounded to fp16 instead of bf16.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void load8(const float* p, float* o) {
    float4 a = __ldg(reinterpret_cast<const float4*>(p));
    float4 b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ void store8h(__half* p, const float* v) {
    uint4 u;
    __half2* q = reinterpret_cast<__half2*>(&u);
#pragma unroll
    for (int j = 0; j < 4; j++) q[j] = __floats2half2_rn(v[2 * j], v[2 * j + 1]);
    *reinterpret_cast<uint4*>(p) = u;
}

template <int NV>
__global__ void __launch_bounds__(128)
ln16_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
            __half* __restrict__ y_out, int M, float eps) {
    constexpr int D = NV * 256;
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
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
        float ww[8], bb[8], y[8];
        load8(w + (i * 32 + lane) * 8, ww);
        load8(b + (i * 32 + lane) * 8, bb);
#pragma unroll
        for (int j = 0; j < 8; j++) y[j] = (v[i][j] - mean) * rstd * ww[j] + bb[j];
        store8h(y_out + (size_t)row * D + (i * 32 + lane) * 8, y);
    }
}

template <int NV>
static void launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, at::Tensor& y, double eps) {
    int M = x.size(0);
    ln16_kernel<NV><<<(M + 3) / 4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), reinterpret_cast<__half*>(y.data_ptr()), M, (float)eps);
}

// x fp32 [..., d] (d = 1024 or 768) -> LN(x) * w + b as fp16, same shape.
at::Tensor ln16(at::Tensor x, at::Tensor w, at::Tensor b, double eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == at::kFloat, "x must be contiguous fp32 cuda");
    int d = x.size(-1);
    TORCH_CHECK(d == 1024 || d == 768, "d must be 1024 or 768");
    TORCH_CHECK(w.dtype() == at::kFloat && b.dtype() == at::kFloat && w.is_contiguous() && b.is_contiguous(), "w,b must be contiguous fp32");
    auto x2 = x.view({-1, d});
    at::Tensor y = at::empty(x2.sizes(), x2.options().dtype(at::kHalf));
    if (d == 1024) launch<4>(x2, w, b, y, eps);
    else launch<3>(x2, w, b, y, eps);
    return y.view(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ln16", &ln16); }
