// MXFP8 E4M3 values and E8M0 scales for the swizzled layout used by _scaled_mm.
// A warp quantizes a 32-element K block; LayerNorm uses a warp per row so the
// normalization reduction and the eight 32-element scales share one launch.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

constexpr float FP8_MAX = 448.0f;

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d));
  return x;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
  return x;
}

__device__ __forceinline__ float block_scale(float amax) {
  return exp2f(ceilf(log2f(fmaxf(amax / FP8_MAX, 1e-30f))));
}

// Equivalent to mx_test.py::to_blocked for K divisible by 128.
__device__ __forceinline__ int scale_offset(int row, int kblock, int K) {
  int ncols4 = K / 128;
  return (((row / 128) * ncols4 + kblock / 4) * 32 + row % 32) * 16
       + (row % 128 / 32) * 4 + kblock % 4;
}

template <typename T> __device__ __forceinline__ float to_float(T x) { return float(x); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T, bool GELU>
__global__ void quant_kernel(const T* __restrict__ x, __nv_fp8_e4m3* __restrict__ out,
                             unsigned char* __restrict__ scale, int K) {
  int row = blockIdx.x;
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int first_block = blockIdx.y * 32 + warp * 4;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int kb = first_block + j;
    if (kb >= K / 32) continue;
    int off = row * K + kb * 32 + lane;
    float v = to_float(x[off]);
    if (GELU) v = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    float s = block_scale(warp_max(fabsf(v)));
    out[off] = __nv_fp8_e4m3(fminf(fmaxf(v / s, -FP8_MAX), FP8_MAX));
    if (lane == 0) scale[scale_offset(row, kb, K)] = __nv_fp8_e8m0(s).__x;
  }
}

template <typename T>
__global__ void ln_quant_kernel(const T* __restrict__ x, const float* __restrict__ gamma,
                                const float* __restrict__ beta, __nv_fp8_e4m3* __restrict__ out,
                                unsigned char* __restrict__ scale, int M, int K, float eps) {
  int row = blockIdx.x * 8 + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (row >= M) return;
  float v[4][8];
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int k0 = j * 256 + lane * 8;
    if (k0 < K) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v[j][i] = to_float(x[row * K + k0 + i]);
        sum += v[j][i];
      }
    }
  }
  float mean = warp_sum(sum) / K;
  float sq = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int k0 = j * 256 + lane * 8;
    if (k0 < K) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float d = v[j][i] - mean;
        sq += d * d;
      }
    }
  }
  float rstd = rsqrtf(warp_sum(sq) / K + eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int k0 = j * 256 + lane * 8;
    if (k0 < K) {
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float y = (v[j][i] - mean) * rstd * gamma[k0 + i] + beta[k0 + i];
        y = __bfloat162float(__float2bfloat16_rn(y));
        v[j][i] = y;
        amax = fmaxf(amax, fabsf(y));
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      float s = block_scale(amax);
      int kb = k0 / 32;
      if (lane % 4 == 0) scale[scale_offset(row, kb, K)] = __nv_fp8_e8m0(s).__x;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        out[row * K + k0 + i] = __nv_fp8_e4m3(fminf(fmaxf(v[j][i] / s, -FP8_MAX), FP8_MAX));
    }
  }
}

static void check(torch::Tensor x, torch::Tensor out, torch::Tensor scale) {
  TORCH_CHECK(x.is_cuda() && out.is_cuda() && scale.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && out.is_contiguous() && scale.is_contiguous(), "contiguous 2D input");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kBFloat16, "input must be fp32/bf16");
  TORCH_CHECK(x.size(1) % 256 == 0, "K must be divisible by 256");
  TORCH_CHECK(out.sizes() == x.sizes() && out.scalar_type() == torch::kFloat8_e4m3fn, "FP8 output shape/dtype");
  TORCH_CHECK(scale.scalar_type() == c10::ScalarType::Float8_e8m0fnu && scale.numel() ==
      ((x.size(0) + 127) / 128) * 128 * (x.size(1) / 32), "E8M0 scale shape/dtype");
  TORCH_CHECK(x.device() == out.device() && x.device() == scale.device(), "device mismatch");
}

template <bool GELU>
void quant_rows(torch::Tensor x, torch::Tensor out, torch::Tensor scale) {
  check(x, out, scale);
  int M = x.size(0), K = x.size(1);
  auto stream = c10::cuda::getCurrentCUDAStream();
  dim3 grid(M, (K / 32 + 31) / 32);
  auto* q = reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr());
  auto* s = reinterpret_cast<unsigned char*>(scale.data_ptr());
  if (x.scalar_type() == torch::kBFloat16)
    quant_kernel<__nv_bfloat16, GELU><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), q, s, K);
  else
    quant_kernel<float, GELU><<<grid, 256, 0, stream>>>(x.data_ptr<float>(), q, s, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ln_quant(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
              torch::Tensor out, torch::Tensor scale, double eps) {
  check(x, out, scale);
  TORCH_CHECK(x.size(1) <= 1024 && gamma.is_contiguous() && beta.is_contiguous() &&
      gamma.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32 &&
      gamma.numel() == x.size(1) && beta.numel() == x.size(1) &&
      gamma.device() == x.device() && beta.device() == x.device(), "LayerNorm parameters");
  int M = x.size(0), K = x.size(1);
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto* q = reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr());
  auto* s = reinterpret_cast<unsigned char*>(scale.data_ptr());
  if (x.scalar_type() == torch::kBFloat16)
    ln_quant_kernel<__nv_bfloat16><<<(M + 7) / 8, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), q, s, M, K, eps);
  else
    ln_quant_kernel<float><<<(M + 7) / 8, 256, 0, stream>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), q, s, M, K, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_rows", &quant_rows<false>);
  m.def("gelu_quant", &quant_rows<true>);
  m.def("ln_quant", &ln_quant);
}
