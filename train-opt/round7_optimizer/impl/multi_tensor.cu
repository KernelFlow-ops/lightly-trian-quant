#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {
constexpr int THREADS = 256;
constexpr int CHUNK = 4096;
// Seven pointers per tensor: master, grad, m, v, step, lr, optional bf16 shadow.
__device__ float* ptr(const int64_t* table, int t, int field) {
  return reinterpret_cast<float*>(table[t * 7 + field]);
}
__global__ void norm_chunks(const int64_t* table, const int64_t* chunks, float* sums) {
  const int64_t* chunk = chunks + blockIdx.x * 3;
  const float* grad = ptr(table, chunk[0], 1) + chunk[1];
  float sum = 0.f;
  for (int i = threadIdx.x; i < chunk[2]; i += THREADS) sum += grad[i] * grad[i];
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
  __shared__ float warps[8];
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warps[threadIdx.x] : 0.f;
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) sums[blockIdx.x] = sum;
  }
}
__global__ void norm_controls(const int64_t* table, const float* hyper, const float* sums,
                             float* controls, float* norm, int nt, int nc, float max_norm) {
  double sum = 0.;
  for (int i = threadIdx.x; i < nc; i += THREADS) sum += sums[i];
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
  __shared__ double warps[8];
  __shared__ float clip;
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warps[threadIdx.x] : 0.;
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) {
      *norm = sqrtf(static_cast<float>(sum));
      clip = max_norm > 0.f ? fminf(1.f, max_norm / (*norm + 1.e-6f)) : 1.f;
    }
  }
  __syncthreads();
  for (int t = threadIdx.x; t < nt; t += THREADS) {
    float* step = ptr(table, t, 4);
    float s = *step + 1.f;
    *step = s;
    float b1 = hyper[t * 4], b2 = hyper[t * 4 + 1], wd = hyper[t * 4 + 2];
    float lr = *ptr(table, t, 5);
    controls[t * 4] = clip;
    controls[t * 4 + 1] = lr / (1.f - powf(b1, s));
    controls[t * 4 + 2] = sqrtf(1.f - powf(b2, s));
    controls[t * 4 + 3] = 1.f - lr * wd;
  }
}
__global__ void update_chunks(const int64_t* table, const int64_t* chunks, const float* hyper,
                              const float* controls) {
  const int64_t* chunk = chunks + blockIdx.x * 3;
  int t = chunk[0];
  int64_t offset = chunk[1];
  float* p = ptr(table, t, 0) + offset;
  float* grad = ptr(table, t, 1) + offset;
  float* m = ptr(table, t, 2) + offset;
  float* v = ptr(table, t, 3) + offset;
  auto* shadow = reinterpret_cast<__nv_bfloat16*>(table[t * 7 + 6]);
  float b1 = hyper[t * 4], b2 = hyper[t * 4 + 1], eps = hyper[t * 4 + 3];
  float clip = controls[t * 4], rate = controls[t * 4 + 1];
  float bc2_sqrt = controls[t * 4 + 2], decay = controls[t * 4 + 3];
  for (int i = threadIdx.x; i < chunk[2]; i += THREADS) {
    float g = grad[i] * clip;
    float mi = m[i] + (g - m[i]) * (1.f - b1);
    float vi = b2 * v[i] + (1.f - b2) * g * g;
    m[i] = mi; v[i] = vi;
    float updated = p[i] * decay - rate * mi / (sqrtf(vi) / bc2_sqrt + eps);
    p[i] = updated;
    grad[i] = 0.f;
    if (shadow != nullptr) shadow[offset + i] = __float2bfloat16_rn(updated);
  }
}
}
void step(torch::Tensor table, torch::Tensor chunks, torch::Tensor hyper,
          torch::Tensor partial, torch::Tensor controls, torch::Tensor norm, double max_norm) {
  TORCH_CHECK(table.is_cuda() && chunks.is_cuda() && hyper.is_cuda(), "CUDA metadata required");
  c10::cuda::CUDAGuard guard(table.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int nc = chunks.size(0), nt = table.size(0);
  if (nc == 0) return;
  norm_chunks<<<nc, THREADS, 0, stream>>>(table.data_ptr<int64_t>(), chunks.data_ptr<int64_t>(), partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  norm_controls<<<1, THREADS, 0, stream>>>(table.data_ptr<int64_t>(), hyper.data_ptr<float>(), partial.data_ptr<float>(),
      controls.data_ptr<float>(), norm.data_ptr<float>(), nt, nc, static_cast<float>(max_norm));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  update_chunks<<<nc, THREADS, 0, stream>>>(table.data_ptr<int64_t>(), chunks.data_ptr<int64_t>(), hyper.data_ptr<float>(), controls.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("step", &step, "Three-launch multi-tensor AdamW + global clip + shadow"); }
