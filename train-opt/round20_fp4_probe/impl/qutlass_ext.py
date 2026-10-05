"""Time-boxed (15 min) attempt to build QuTLASS (IST-DASLab/qutlass e74319e3405c, Apache-2.0) as a stable-ABI torch library, in place from third_party,
against the CUTLASS 4.2 headers of cutlass_library (the repo's own submodule is empty).  Candidate backend only (GEMM matmul_nvf4/mxf4_bf16_tn and the
quantizers fusedQuantizeNv/Mx, backward_t_bf16).  Build only, no GPU:   python qutlass_ext.py --build-only ;  load: torch.ops.load_library(LIB)
"""
import os
import sys
from pathlib import Path

SRC = Path("/root/autodl-tmp/train-opt-tmp/third_party/qutlass")
BUILD = Path("/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/build/qutlass")
CUT = Path("/root/miniconda3/lib/python3.12/site-packages/cutlass_library/source")
LIB = BUILD / "_qutlass_C.so"
TV = (2 << 56) | (11 << 48)
SOURCES = ["bindings.cpp", "gemm.cu", "gemm_ada.cu", "fused_quantize_mx.cu", "fused_quantize_mx_mask.cu", "fused_quantize_nv.cu",
           "fused_quantize_mx_sm100.cu", "fused_quantize_nv_sm100.cu", "quartet_bwd_sm120.cu"]


def build_only():
    from torch.utils import cpp_extension as ce

    os.environ.setdefault("MAX_JOBS", "32")
    BUILD.mkdir(parents=True, exist_ok=True)
    srcs = [str(SRC / "qutlass" / "csrc" / n) for n in SOURCES]
    inc = [str(SRC / "qutlass/csrc/include"), str(SRC / "qutlass/csrc/include/cutlass_extensions"), str(CUT / "include"), str(CUT / "tools/util/include")]
    cxx = ["-std=c++17", "-DUSE_CUDA", f"-DTORCH_TARGET_VERSION=0x{TV:016X}ULL", "-DTARGET_CUDA_ARCH=120", "-O3"]
    nvcc = ["-DUSE_CUDA", f"-DTORCH_TARGET_VERSION=0x{TV:016X}ULL", "-DTARGET_CUDA_ARCH=120", "-gencode", "arch=compute_120a,code=sm_120a", "--expt-relaxed-constexpr",
            "--use_fast_math", "-std=c++17", "-O3", "-DNDEBUG", "-Xcompiler", "-funroll-loops"]
    ce._write_ninja_file_and_build_library(name="_qutlass_C", sources=srcs, extra_cflags=cxx, extra_cuda_cflags=nvcc, extra_ldflags=["-lcuda"], extra_include_paths=inc,
                                           build_directory=str(BUILD), verbose=True, with_cuda=True, extra_sycl_cflags=None, with_sycl=False)


if __name__ == "__main__":
    if "--build-only" in sys.argv:
        build_only()
        print("built")
