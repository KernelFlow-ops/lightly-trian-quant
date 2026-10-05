// Round 10: stand-alone versions of the R5 GEMM epilogues, used ONLY to price the missing epilogue of GEMM candidates that write a plain bf16 / int32 output
// (cuBLASLt, Spark, torch._int_mm).  Simple bandwidth-bound kernels (no tuning): they bound what a separate epilogue pass costs.
//   epi_res :  out_f32[m,n] = res[m,n] + gamma[n] * (y[m,n] * sx[m] * sw[n] + bias[n])            (proj / fc2 epilogue)
//   epi_gelu:  out_bf16[m,n] = gelu_erf(y[m,n] * sx[m] * sw[n] + bias[n])                          (fc1 epilogue)
//   epi_rope:  out_bf16[3,B,N,H,64] = RoPE(y[m, which*d + h*64 + c] * sx[m]*sw[col] + bias)        (qkv epilogue: bias, RoPE on q/k rows >= prefix, head layout)
// y is bf16 (scales null) or int32 (scales given).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

template <typename T> __device__ __forceinline__ float ld(const T* p, long i);
template <> __device__ __forceinline__ float ld<__nv_bfloat16>(const __nv_bfloat16* p, long i) { return __bfloat162float(p[i]); }
template <> __device__ __forceinline__ float ld<int>(const int* p, long i) { return (float)p[i]; }

template <typename T>
__global__ void epi_res_k(const T* y, const float* res, const float* gamma, const float* bias, const float* sx, const float* sw, float* out, int M, int N) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * N) return;
    int n = i % N, m = i / N;
    float v = ld<T>(y, i);
    if (sx) v *= sx[m] * sw[n];
    out[i] = res[i] + gamma[n] * (v + bias[n]);
}

template <typename T>
__global__ void epi_gelu_k(const T* y, const float* bias, const float* sx, const float* sw, __nv_bfloat16* out, int M, int N) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)M * N) return;
    int n = i % N, m = i / N;
    float v = ld<T>(y, i);
    if (sx) v *= sx[m] * sw[n];
    v += bias[n];
    out[i] = __float2bfloat16_rn(v * 0.5f * (1.0f + erff(v * 0.7071067811865476f)));
}

// one thread per (row, which, head, c<32): partners c and c+32
template <typename T>
__global__ void epi_rope_k(const T* y, const float* bias, const float* sx, const float* sw, const float* sin_t, const float* cos_t, __nv_bfloat16* out,
                           int M, int Ntok, int prefix, int d, int H) {
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)M * 3 * H * 32;
    if (t >= total) return;
    int c = t % 32; long r = t / 32;
    int h = r % H; r /= H;
    int which = r % 3; int m = r / 3;
    int col = which * d + h * 64 + c;
    long base = (long)m * 3 * d;
    float lo = ld<T>(y, base + col), hi = ld<T>(y, base + col + 32);
    if (sx) { lo *= sx[m] * sw[col]; hi *= sx[m] * sw[col + 32]; }
    lo += bias[col]; hi += bias[col + 32];
    int n = m % Ntok, pp = n - prefix;
    if (which < 2 && pp >= 0) {
        float cl = cos_t[(long)pp * 64 + c], sl = sin_t[(long)pp * 64 + c], ch = cos_t[(long)pp * 64 + c + 32], sh = sin_t[(long)pp * 64 + c + 32];
        float nl = lo * cl - hi * sl, nh = hi * ch + lo * sh;
        lo = nl; hi = nh;
    }
    int B = M / Ntok, b = m / Ntok;
    long o = ((((long)which * B + b) * Ntok + n) * H + h) * 64 + c;
    out[o] = __float2bfloat16_rn(lo);
    out[o + 32] = __float2bfloat16_rn(hi);
}

// per-row (token) symmetric int8 quantiser: s = amax/127, q = clamp(rint(x/s), -127, 127)   (bf16 [M,K] -> int8 [M,K] + fp32 scale [M]); one block per row, 256 threads, K % 2048 == 0 or K % 8 == 0
__global__ void quant_int8_k(const __nv_bfloat16* x, int8_t* q, float* s, int K) {
    __shared__ float red[8];
    const __nv_bfloat16* xr = x + (long)blockIdx.x * K;
    float amax = 0.f;
    for (int i = threadIdx.x * 8; i < K; i += 256 * 8) {
        uint4 v = *reinterpret_cast<const uint4*>(xr + i);
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(__bfloat162float(e[j])));
    }
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < 8; w++) amax = fmaxf(amax, red[w]);
    float sc = amax > 0.f ? amax / 127.f : 1.f;
    float inv = 1.f / sc;
    if (threadIdx.x == 0) s[blockIdx.x] = sc;
    for (int i = threadIdx.x * 8; i < K; i += 256 * 8) {
        uint4 v = *reinterpret_cast<const uint4*>(xr + i);
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&v);
        int8_t o[8];
#pragma unroll
        for (int j = 0; j < 8; j++) o[j] = (int8_t)max(-127, min(127, __float2int_rn(__bfloat162float(e[j]) * inv)));
        *reinterpret_cast<uint2*>(q + (long)blockIdx.x * K + i) = *reinterpret_cast<uint2*>(o);
    }
}
void quant_int8(torch::Tensor x, torch::Tensor q, torch::Tensor s) {
    int M = x.size(0), K = x.size(1);
    quant_int8_k<<<M, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)x.data_ptr(), (int8_t*)q.data_ptr(), s.data_ptr<float>(), K);
}

#define OPT(t) (t.numel() > 0 ? t.data_ptr<float>() : nullptr)
void epi_res(torch::Tensor y, torch::Tensor res, torch::Tensor gamma, torch::Tensor bias, torch::Tensor sx, torch::Tensor sw, torch::Tensor out) {
    int M = y.size(0), N = y.size(1); long n = (long)M * N; auto s = at::cuda::getCurrentCUDAStream();
    if (y.scalar_type() == torch::kInt) epi_res_k<int><<<(n + 255) / 256, 256, 0, s>>>(y.data_ptr<int>(), res.data_ptr<float>(), gamma.data_ptr<float>(), bias.data_ptr<float>(), OPT(sx), OPT(sw), out.data_ptr<float>(), M, N);
    else epi_res_k<__nv_bfloat16><<<(n + 255) / 256, 256, 0, s>>>((const __nv_bfloat16*)y.data_ptr(), res.data_ptr<float>(), gamma.data_ptr<float>(), bias.data_ptr<float>(), OPT(sx), OPT(sw), out.data_ptr<float>(), M, N);
}
void epi_gelu(torch::Tensor y, torch::Tensor bias, torch::Tensor sx, torch::Tensor sw, torch::Tensor out) {
    int M = y.size(0), N = y.size(1); long n = (long)M * N; auto s = at::cuda::getCurrentCUDAStream();
    if (y.scalar_type() == torch::kInt) epi_gelu_k<int><<<(n + 255) / 256, 256, 0, s>>>(y.data_ptr<int>(), bias.data_ptr<float>(), OPT(sx), OPT(sw), (__nv_bfloat16*)out.data_ptr(), M, N);
    else epi_gelu_k<__nv_bfloat16><<<(n + 255) / 256, 256, 0, s>>>((const __nv_bfloat16*)y.data_ptr(), bias.data_ptr<float>(), OPT(sx), OPT(sw), (__nv_bfloat16*)out.data_ptr(), M, N);
}
void epi_rope(torch::Tensor y, torch::Tensor bias, torch::Tensor sx, torch::Tensor sw, torch::Tensor sin_t, torch::Tensor cos_t, torch::Tensor out, int64_t Ntok, int64_t prefix, int64_t d, int64_t H) {
    int M = y.size(0); long total = (long)M * 3 * H * 32; auto s = at::cuda::getCurrentCUDAStream();
    if (y.scalar_type() == torch::kInt) epi_rope_k<int><<<(total + 255) / 256, 256, 0, s>>>(y.data_ptr<int>(), bias.data_ptr<float>(), OPT(sx), OPT(sw), sin_t.data_ptr<float>(), cos_t.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), M, Ntok, prefix, d, H);
    else epi_rope_k<__nv_bfloat16><<<(total + 255) / 256, 256, 0, s>>>((const __nv_bfloat16*)y.data_ptr(), bias.data_ptr<float>(), OPT(sx), OPT(sw), sin_t.data_ptr<float>(), cos_t.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), M, Ntok, prefix, d, H);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("epi_res", &epi_res);
    m.def("epi_gelu", &epi_gelu);
    m.def("epi_rope", &epi_rope);
    m.def("quant_int8", &quant_int8);
}
