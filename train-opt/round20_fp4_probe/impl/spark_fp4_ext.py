"""Candidate backend only: JIT build of Spark's fp4gemm / fp4quant (MIT, aidxhxr/5090-blackwell-kernels a401bfa9e363) with a minimal
torch binding that writes into caller-provided buffers (graph friendly).  Sources are compiled in place from
/root/autodl-tmp/train-opt-tmp/third_party/spark (read-only); nothing is copied.  Build dir is under train-opt-tmp.

  python spark_fp4_ext.py --build-only     # compile without touching the GPU (no lock needed)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path("/root/autodl-tmp/train-opt-tmp/third_party/spark")
_BUILD = Path("/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/build/spark_fp4")
_EXT = None

BINDING = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "spark/kernels.h"

using at::Tensor;
// out [M,N] bf16 = sa*sb*(A[M,K/2] . Bt[N,K/2]^T); sfa/sfb blocked scales (uint8 views); sa/sb fp32 device scalars or None
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
bool supports(int64_t M, int64_t N, int64_t K, int64_t v) { return spark::fp4gemm_supports(M, N, K, v); }
int64_t scale_bytes(int64_t rows, int64_t K, int64_t fmt) { return spark::fp4_scale_bytes(rows, K, (int)fmt); }
// x [rows,K] bf16 -> q [rows,K/2], sf (blocked); scale: device fp32 scalar (nvfp4) or None; work/scale_out: device amax path
void fp4quant(const Tensor& x, Tensor& q, Tensor& sf, const std::optional<Tensor>& scale, int64_t fmt,
              const std::optional<Tensor>& work, const std::optional<Tensor>& scale_out) {
    c10::cuda::CUDAGuard guard(x.device());
    spark::fp4_quantize(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (unsigned char*)q.data_ptr(), (unsigned char*)sf.data_ptr(),
                        x.size(0), x.size(1), scale ? scale->data_ptr<float>() : nullptr, (int)fmt, at::cuda::getCurrentCUDAStream(),
                        work ? (unsigned*)work->data_ptr() : nullptr, scale_out ? scale_out->data_ptr<float>() : nullptr);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp4gemm", &fp4gemm);
    m.def("supports", &supports);
    m.def("scale_bytes", &scale_bytes);
    m.def("fp4quant", &fp4quant);
}
'''


def _args():
    srcs_cu = [str(_SRC / "src" / "kernels" / n) for n in ("fp4gemm.cu", "fp4gemm_tma.cu", "fp4quant.cu", "fp8quant.cu")]
    return dict(
        name="trainopt_r20_spark_fp4",
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
    """Import the built module (call under the GPU lock)."""
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
