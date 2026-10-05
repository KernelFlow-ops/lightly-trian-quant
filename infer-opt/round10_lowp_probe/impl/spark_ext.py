"""Round 10 candidate backend: JIT build of Spark (aidxhxr/5090-blackwell-kernels, MIT, commit a401bfa9e363) fp8gemm (MX mode) / fp4gemm / quantizers with a
minimal torch binding (adapted from train-opt round20_fp4_probe/impl/spark_fp4_ext.py, plus fp8gemm_mx and fp8_quantize).  Sources are compiled in place
from /root/autodl-tmp/train-opt-tmp/third_party/spark (read-only); the build dir is under infer-opt-tmp/round10_lowp_probe.

  python spark_ext.py --build-only     # compile without touching the GPU (no lock needed)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path("/root/autodl-tmp/train-opt-tmp/third_party/spark")
_BUILD = Path("/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/build/spark")
_EXT = None

BINDING = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "spark/kernels.h"

using at::Tensor;
// ---- FP8 MX: out [M,N] bf16 = sum_k 2^(sfa-127) A[m,k] 2^(sfb-127) Bt[n,k]; A [M,K] / Bt [N,K] e4m3 bytes (uint8 or float8 view), sfa [M,K/32] / sfb [N,K/32] uint8
void fp8gemm_mx(const Tensor& a, const Tensor& bt, Tensor& out, const Tensor& sfa, const Tensor& sfb, int64_t variant) {
    c10::cuda::CUDAGuard guard(a.device());
    int M = a.size(0), N = bt.size(0), K = a.size(1);
    spark::fp8gemm_mx((const __nv_fp8_e4m3*)a.data_ptr(), (const __nv_fp8_e4m3*)bt.data_ptr(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K,
                      nullptr, nullptr, (const unsigned char*)sfa.data_ptr(), (const unsigned char*)sfb.data_ptr(), (int)variant, at::cuda::getCurrentCUDAStream());
}
bool fp8_supports(int64_t M, int64_t N, int64_t K, int64_t v) { return spark::fp8gemm_mx_supports(M, N, K, v); }
// x [rows,K] bf16 -> q [rows,K] e4m3 bytes, scale [rows,K/32] uint8 (MX recipe)
void fp8_quantize_mx(const Tensor& x, Tensor& q, Tensor& scale) {
    c10::cuda::CUDAGuard guard(x.device());
    spark::fp8_quantize(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (unsigned char*)q.data_ptr(), scale.data_ptr(), nullptr, x.size(0), x.size(1),
                        spark::FP8Q_MX, at::cuda::getCurrentCUDAStream());
}
// ---- FP4
void fp4gemm(const Tensor& a, const Tensor& bt, Tensor& out, const Tensor& sfa, const Tensor& sfb,
             const std::optional<Tensor>& sa, const std::optional<Tensor>& sb, int64_t fmt, int64_t variant) {
    c10::cuda::CUDAGuard guard(a.device());
    int M = a.size(0), N = bt.size(0), K = 2 * a.size(1);
    spark::fp4gemm((const unsigned char*)a.data_ptr(), (const unsigned char*)bt.data_ptr(),
                   reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K,
                   (const unsigned char*)sfa.data_ptr(), (const unsigned char*)sfb.data_ptr(),
                   sa ? sa->data_ptr<float>() : nullptr, sb ? sb->data_ptr<float>() : nullptr, (int)fmt, (int)variant,
                   at::cuda::getCurrentCUDAStream());
}
bool fp4_supports(int64_t M, int64_t N, int64_t K, int64_t v) { return spark::fp4gemm_supports(M, N, K, v); }
int64_t scale_bytes(int64_t rows, int64_t K, int64_t fmt) { return spark::fp4_scale_bytes(rows, K, (int)fmt); }
void fp4quant(const Tensor& x, Tensor& q, Tensor& sf, const std::optional<Tensor>& scale, int64_t fmt,
              const std::optional<Tensor>& work, const std::optional<Tensor>& scale_out) {
    c10::cuda::CUDAGuard guard(x.device());
    spark::fp4_quantize(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (unsigned char*)q.data_ptr(), (unsigned char*)sf.data_ptr(),
                        x.size(0), x.size(1), scale ? scale->data_ptr<float>() : nullptr, (int)fmt, at::cuda::getCurrentCUDAStream(),
                        work ? (unsigned*)work->data_ptr() : nullptr, scale_out ? scale_out->data_ptr<float>() : nullptr);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8gemm_mx", &fp8gemm_mx);
    m.def("fp8_supports", &fp8_supports);
    m.def("fp8_quantize_mx", &fp8_quantize_mx);
    m.def("fp4gemm", &fp4gemm);
    m.def("fp4_supports", &fp4_supports);
    m.def("scale_bytes", &scale_bytes);
    m.def("fp4quant", &fp4quant);
}
'''


def _args():
    srcs_cu = [str(_SRC / "src" / "kernels" / n) for n in ("fp8gemm.cu", "fp8gemm_tma.cu", "fp8quant.cu", "fp4gemm.cu", "fp4gemm_tma.cu", "fp4quant.cu")]
    return dict(
        name="r10_spark",
        sources=[str(_BUILD / "binding.cpp")] + srcs_cu,
        extra_include_paths=[str(_SRC / "include"), str(_SRC / "src" / "kernels")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-std=c++20", "-gencode", "arch=compute_120a,code=sm_120a"],
        build_directory=str(_BUILD),
    )


def build_only():
    from torch.utils import cpp_extension as ce

    os.environ.setdefault("MAX_JOBS", "16")
    _BUILD.mkdir(parents=True, exist_ok=True)
    (_BUILD / "binding.cpp").write_text(BINDING)
    a = _args()
    ce._write_ninja_file_and_build_library(
        name=a["name"], sources=a["sources"], extra_cflags=a["extra_cflags"], extra_cuda_cflags=a["extra_cuda_cflags"],
        extra_ldflags=[], extra_include_paths=a["extra_include_paths"], build_directory=a["build_directory"], verbose=True, with_cuda=True,
        extra_sycl_cflags=None, with_sycl=False)


def extension():
    """Import the built module (build first with --build-only; call under the GPU lock)."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("MAX_JOBS", "16")
        _BUILD.mkdir(parents=True, exist_ok=True)
        (_BUILD / "binding.cpp").write_text(BINDING)
        _EXT = load(verbose=False, **_args())
    return _EXT


if __name__ == "__main__":
    if "--build-only" in sys.argv:
        build_only()
        print("built")
