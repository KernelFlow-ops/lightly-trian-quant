"""JIT build of the Spark bf16 hgemm (third_party/spark_hgemm) with a minimal torch binding.

Candidate only (row-major NN, bf16 out, tanh GELU); see NOTES for why it is not the chosen path.
"""
from __future__ import annotations

import os
from pathlib import Path

_EXT = None
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / 'third_party' / 'spark_hgemm'
_BUILD = Path('/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/build/spark_hgemm')

BINDING = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "spark/kernels.h"

static const __nv_bfloat16* P(const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); }
static __nv_bfloat16* PM(at::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); }

// out[M,N] = act(a[M,K] @ b[K,N] + bias) + residual ; all bf16 row-major contiguous
void hgemm(at::Tensor a, at::Tensor b, at::Tensor out, int64_t variant, c10::optional<at::Tensor> bias, int64_t act, c10::optional<at::Tensor> residual) {
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(), "contiguous bf16 expected");
    c10::cuda::CUDAGuard guard(a.device());
    spark::HgemmEpilogue ep;
    ep.act = static_cast<int>(act);
    if (bias.has_value()) ep.bias = P(*bias);
    if (residual.has_value()) ep.residual = P(*residual);
    int M = a.size(0), K = a.size(1), N = b.size(1);
    spark::hgemm_bf16(P(a), P(b), PM(out), M, N, K, static_cast<int>(variant), at::cuda::getCurrentCUDAStream(), ep);
}
int64_t num_variants() { return spark::hgemm_num_variants(); }
bool supports(int64_t M, int64_t N, int64_t K, int64_t v) { return spark::hgemm_supports(M, N, K, v); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hgemm", &hgemm);
    m.def("num_variants", &num_variants);
    m.def("supports", &supports);
}
'''

def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault('MAX_JOBS', '8')
        _BUILD.mkdir(parents=True, exist_ok=True)
        binding = _BUILD / 'spark_hgemm_binding.cpp'
        binding.write_text(BINDING)
        srcs = [str(binding)] + sorted(str(p) for p in (_SRC / 'src' / 'kernels').glob('*.cu'))
        _EXT = load(name='trainopt_spark_hgemm', sources=srcs, build_directory=str(_BUILD),
                    extra_include_paths=[str(_SRC / 'include'), str(_SRC / 'src' / 'kernels')],
                    extra_cflags=['-O3', '-std=c++20'],
                    extra_cuda_cflags=['-O3', '-lineinfo', '--expt-relaxed-constexpr', '-std=c++20',
                                       '-gencode', 'arch=compute_120a,code=sm_120a'],
                    verbose=False)
    return _EXT
