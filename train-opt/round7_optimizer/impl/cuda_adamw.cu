#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace trainopt {

template <typename scalar_t>
__global__ void norm_kernel(const scalar_t* grad, int64_t n, float* out) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = static_cast<float>(grad[i]);
    atomicAdd(out, x * x);
  }
}

__global__ void increment_step_kernel(float* step) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *step += 1.0f;
  }
}

template <typename scalar_t>
__global__ void adamw_kernel(
    scalar_t* param, scalar_t* grad, float* exp_avg, float* exp_avg_sq,
    int64_t n, const float* norm, const float* lr, const float* wd,
    const float* step, float max_norm, float beta1, float beta2, float eps,
    at::BFloat16* shadow) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float p = static_cast<float>(param[i]);
  float g = static_cast<float>(grad[i]);
  float total_norm = sqrtf(fmaxf(*norm, 0.0f));
  float coef = fminf(1.0f, max_norm / (total_norm + 1.0e-6f));
  g *= coef;
  float m = beta1 * exp_avg[i] + (1.0f - beta1) * g;
  float v = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
  exp_avg[i] = m;
  exp_avg_sq[i] = v;
  float s = fmaxf(*step, 1.0f);
  float bc1 = 1.0f - powf(beta1, s);
  float bc2 = 1.0f - powf(beta2, s);
  float denom = sqrtf(v / bc2) + eps;
  float learning_rate = *lr;
  p *= (1.0f - learning_rate * (*wd));
  p -= (learning_rate / bc1) * (m / denom);
  param[i] = static_cast<scalar_t>(p);
  if (shadow != nullptr) shadow[i] = static_cast<at::BFloat16>(p);
}

static void check_cuda(const char* where) {
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, where, ": ", cudaGetErrorString(err));
}

torch::Tensor fused_adamw_step(
    std::vector<torch::Tensor> params,
    std::vector<torch::Tensor> grads,
    std::vector<torch::Tensor> exp_avg,
    std::vector<torch::Tensor> exp_avg_sq,
    std::vector<torch::Tensor> steps,
    std::vector<torch::Tensor> lrs,
    std::vector<torch::Tensor> wds,
    std::vector<torch::Tensor> shadows,
    double max_norm, double beta1, double beta2, double eps) {
  TORCH_CHECK(params.size() == grads.size(), "params/grads size mismatch");
  TORCH_CHECK(params.size() == exp_avg.size() && params.size() == exp_avg_sq.size(), "Adam state size mismatch");
  TORCH_CHECK(params.size() == steps.size() && params.size() == lrs.size() && params.size() == wds.size(), "device scalar size mismatch");
  TORCH_CHECK(shadows.empty() || shadows.size() == params.size(), "shadow size mismatch");
  auto options = torch::TensorOptions().device(params[0].device()).dtype(torch::kFloat32);
  auto norm = torch::zeros({}, options);
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  for (size_t j = 0; j < params.size(); ++j) {
    TORCH_CHECK(params[j].is_cuda() && grads[j].is_cuda(), "CUDA tensors required");
    TORCH_CHECK(params[j].is_contiguous() && grads[j].is_contiguous(), "contiguous tensors required");
    int blocks = static_cast<int>((grads[j].numel() + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        grads[j].scalar_type(), "gradient_norm", [&] {
          norm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(grads[j].data_ptr<scalar_t>(), grads[j].numel(), norm.data_ptr<float>());
        });
    check_cuda("norm_kernel");
  }
  for (size_t j = 0; j < params.size(); ++j) {
    increment_step_kernel<<<1, 1, 0, stream>>>(steps[j].data_ptr<float>());
    check_cuda("increment_step_kernel");
    int blocks = static_cast<int>((params[j].numel() + threads - 1) / threads);
    at::BFloat16* shadow_ptr = nullptr;
    if (!shadows.empty()) shadow_ptr = shadows[j].data_ptr<at::BFloat16>();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        params[j].scalar_type(), "adamw_update", [&] {
          adamw_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              params[j].data_ptr<scalar_t>(), grads[j].data_ptr<scalar_t>(),
              exp_avg[j].data_ptr<float>(), exp_avg_sq[j].data_ptr<float>(),
              params[j].numel(), norm.data_ptr<float>(), lrs[j].data_ptr<float>(),
              wds[j].data_ptr<float>(), steps[j].data_ptr<float>(),
              static_cast<float>(max_norm), static_cast<float>(beta1),
              static_cast<float>(beta2), static_cast<float>(eps), shadow_ptr);
        });
    check_cuda("adamw_kernel");
  }
  return norm.sqrt();
}

}  // namespace trainopt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_adamw_step", &trainopt::fused_adamw_step,
        "Correctness-first CUDA fused AdamW + clip + shadow");
}
