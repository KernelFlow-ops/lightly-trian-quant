// Round 6: activation quantization kernels for FP8 (e4m3) GEMMs with per-token (rowwise) scales.
//   quant_rows : bf16/fp32 [M,K] -> fp8 [M,K] + scale[M]                (absmax/448)
//   ln_quant   : LayerNorm(x) fused with quantization (x fp32/bf16, K <= 1024)
//   gelu_quant : exact-erf GELU(x bf16) fused with quantization (K up to 4096+)
// One warp per row; each lane owns 8 contiguous elements per iteration (16 B loads for bf16).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define FP8_MAX 448.0f

template <typename T> __device__ __forceinline__ void load8(const T* p, float* f);
template <> __device__ __forceinline__ void load8<__nv_bfloat16>(const __nv_bfloat16* p, float* f) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) { float2 t = __bfloat1622float2(h[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}
template <> __device__ __forceinline__ void load8<float>(const float* p, float* f) {
  float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}

__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f)); }

__device__ __forceinline__ void store8_fp8(__nv_fp8_e4m3* p, const float* f, float inv_s) {
  uint32_t w[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint32_t v = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __nv_fp8_e4m3 q(fminf(fmaxf(f[i * 4 + j] / inv_s, -FP8_MAX), FP8_MAX));  // f / s  (inv_s holds s)
      v |= (uint32_t)q.__x << (8 * j);
    }
    w[i] = v;
  }
  *reinterpret_cast<uint2*>(p) = make_uint2(w[0], w[1]);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// MODE 0: plain quantization of x; MODE 1: gelu(x) then quantization
template <typename T, int MODE>
__global__ void quant_rows_kernel(const T* __restrict__ x, __nv_fp8_e4m3* __restrict__ out,
                                  float* __restrict__ scale, int M, int K) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= M) return;
  const T* xr = x + (size_t)row * K;
  float amax = 0.f;
  for (int k = lane * 8; k < K; k += 256) {
    float f[8];
    load8<T>(xr + k, f);
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(MODE == 1 ? gelu_erf(f[i]) : f[i]));
  }
  amax = warp_max(amax);
  float s = fmaxf(amax / FP8_MAX, 1e-30f);
  for (int k = lane * 8; k < K; k += 256) {
    float f[8];
    load8<T>(xr + k, f);
    if (MODE == 1) {
#pragma unroll
      for (int i = 0; i < 8; ++i) f[i] = gelu_erf(f[i]);
    }
    store8_fp8(out + (size_t)row * K + k, f, s);
  }
  if (lane == 0) scale[row] = s;
}

template <typename T>
__global__ void ln_quant_kernel(const T* __restrict__ x, const float* __restrict__ gamma,
                                const float* __restrict__ beta, __nv_fp8_e4m3* __restrict__ out,
                                float* __restrict__ scale, int M, int K, float eps) {
  constexpr int MAXV = 4;  // K <= 1024
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= M) return;
  const T* xr = x + (size_t)row * K;
  float v[MAXV][8];
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < MAXV; ++j) {
    int k = j * 256 + lane * 8;
    if (k < K) {
      load8<T>(xr + k, v[j]);
#pragma unroll
      for (int i = 0; i < 8; ++i) sum += v[j][i];
    }
  }
  float mean = warp_sum(sum) / (float)K;
  float sq = 0.f;
#pragma unroll
  for (int j = 0; j < MAXV; ++j) {
    int k = j * 256 + lane * 8;
    if (k < K) {
#pragma unroll
      for (int i = 0; i < 8; ++i) { float d = v[j][i] - mean; sq += d * d; }
    }
  }
  float rstd = rsqrtf(warp_sum(sq) / (float)K + eps);
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < MAXV; ++j) {
    int k = j * 256 + lane * 8;
    if (k < K) {
      const float4* gp = reinterpret_cast<const float4*>(gamma + k);
      const float4* bp = reinterpret_cast<const float4*>(beta + k);
      float4 g0 = gp[0], g1 = gp[1], b0 = bp[0], b1 = bp[1];
      float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
      float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        // reference: LN output is fp32 (autocast) and then rounded to bf16 by the linear
        float y = (v[j][i] - mean) * rstd * g[i] + b[i];
        y = __bfloat162float(__float2bfloat16_rn(y));
        v[j][i] = y;
        amax = fmaxf(amax, fabsf(y));
      }
    }
  }
  amax = warp_max(amax);
  float s = fmaxf(amax / FP8_MAX, 1e-30f);
#pragma unroll
  for (int j = 0; j < MAXV; ++j) {
    int k = j * 256 + lane * 8;
    if (k < K) store8_fp8(out + (size_t)row * K + k, v[j], s);
  }
  if (lane == 0) scale[row] = s;
}

static void check(const torch::Tensor& x, const torch::Tensor& out, const torch::Tensor& scale) {
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && scale.is_contiguous(), "contiguous");
  TORCH_CHECK(x.size(1) % 256 == 0, "K must be a multiple of 256");
  TORCH_CHECK(out.scalar_type() == torch::kFloat8_e4m3fn && scale.scalar_type() == torch::kFloat32, "dtypes");
}

template <int MODE>
void quant_rows(torch::Tensor x, torch::Tensor out, torch::Tensor scale) {
  check(x, out, scale);
  int M = (int)x.size(0), K = (int)x.size(1);
  int warps = 8, blocks = (M + warps - 1) / warps;
  auto st = c10::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr());
  if (x.scalar_type() == torch::kBFloat16)
    quant_rows_kernel<__nv_bfloat16, MODE><<<blocks, warps * 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), o, scale.data_ptr<float>(), M, K);
  else if (x.scalar_type() == torch::kFloat32)
    quant_rows_kernel<float, MODE><<<blocks, warps * 32, 0, st>>>(x.data_ptr<float>(), o, scale.data_ptr<float>(), M, K);
  else TORCH_CHECK(false, "dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ln_quant(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor out,
              torch::Tensor scale, double eps) {
  check(x, out, scale);
  int M = (int)x.size(0), K = (int)x.size(1);
  TORCH_CHECK(K <= 1024, "K<=1024");
  int warps = 8, blocks = (M + warps - 1) / warps;
  auto st = c10::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr());
  if (x.scalar_type() == torch::kBFloat16)
    ln_quant_kernel<__nv_bfloat16><<<blocks, warps * 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        o, scale.data_ptr<float>(), M, K, (float)eps);
  else
    ln_quant_kernel<float><<<blocks, warps * 32, 0, st>>>(x.data_ptr<float>(), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), o, scale.data_ptr<float>(), M, K, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_rows", &quant_rows<0>);
  m.def("gelu_quant", &quant_rows<1>);
  m.def("ln_quant", &ln_quant);
}
