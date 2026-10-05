// Minimal torch binding for the Spark attention kernels (forward with lse, backward) used by the round12
// candidate matrix.  Adapted from spark-kernels python/csrc/bindings.cpp (MIT, aidxhxr/5090-blackwell-kernels@a401bfa9e363).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <tuple>

#include "spark/kernels.h"

using at::Tensor;

static const __nv_bfloat16* bf16_ptr(const Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); }
static __nv_bfloat16* bf16_ptr_mut(Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); }

static std::tuple<Tensor, Tensor> attention_fwd(const Tensor& q, const Tensor& k, const Tensor& v, bool causal, int variant) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "contiguous [B,H,S,D] bf16 expected");
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor out = at::empty_like(q);
    Tensor lse = at::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(at::kFloat));
    int var = variant < 0 ? spark::attention_num_variants() - 1 : variant;
    spark::attention_bf16(bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), bf16_ptr_mut(out), (int)q.size(0), (int)q.size(1), (int)k.size(1),
                          (int)q.size(2), (int)k.size(2), (int)q.size(3), causal, var, at::cuda::getCurrentCUDAStream(), lse.data_ptr<float>());
    return {out, lse};
}

static std::tuple<Tensor, Tensor, Tensor> attention_bwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
                                                        const Tensor& d_out, const Tensor& lse, bool causal, int variant,
                                                        bool deterministic) {
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor dq = at::empty_like(q), dk = at::empty_like(k), dv = at::empty_like(v);
    int var = variant < 0 ? spark::attention_bwd_num_variants() - 1 : variant;
    spark::attention_bwd_bf16(bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), bf16_ptr(out), bf16_ptr(d_out), lse.data_ptr<float>(),
                              bf16_ptr_mut(dq), bf16_ptr_mut(dk), bf16_ptr_mut(dv), (int)q.size(0), (int)q.size(1), (int)k.size(1),
                              (int)q.size(2), (int)k.size(2), (int)q.size(3), causal, deterministic, var,
                              at::cuda::getCurrentCUDAStream());
    return {dq, dk, dv};
}

static int fwd_variants() { return spark::attention_num_variants(); }
static int bwd_variants() { return spark::attention_bwd_num_variants(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_fwd", &attention_fwd);
    m.def("attention_bwd", &attention_bwd);
    m.def("fwd_variants", &fwd_variants);
    m.def("bwd_variants", &bwd_variants);
}
