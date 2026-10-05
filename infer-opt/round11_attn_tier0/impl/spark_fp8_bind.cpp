// Minimal torch binding for Spark's fp8 attention forward (candidate-matrix use, round 11).
// Adapted from spark-kernels python/csrc/bindings.cpp (MIT, aidxhxr/5090-blackwell-kernels@a401bfa9e363).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include "spark/kernels.h"

using at::Tensor;

static Tensor attention_fp8(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& q_scale, const Tensor& k_scale,
                            const Tensor& v_scale, bool per_head, int variant) {
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn && k.scalar_type() == at::kFloat8_e4m3fn && v.scalar_type() == at::kFloat8_e4m3fn, "e4m3 q,k,v");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && q.dim() == 4, "contiguous [B,H,S,D]");
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor out = at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
    int var = variant < 0 ? spark::attention_fp8_num_variants() - 1 : variant;
    spark::attention_fp8(reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()), reinterpret_cast<const __nv_fp8_e4m3*>(k.data_ptr()),
                         reinterpret_cast<const __nv_fp8_e4m3*>(v.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)q.size(0),
                         (int)q.size(1), (int)k.size(1), (int)q.size(2), (int)k.size(2), (int)q.size(3), q_scale.data_ptr<float>(),
                         k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), per_head, false, var, at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attention_fp8", &attention_fp8); }
