"""JIT build of Spark's fp8 attention (attention_fp8.cu, MIT) as a torch extension; sources read from third_party (read-only)."""
from pathlib import Path
SPARK = Path("/root/autodl-tmp/train-opt-tmp/third_party/spark")
BUILD = Path("/root/autodl-tmp/infer-opt-tmp/round11_attn_tier0/build_spark_fp8")
_ext = None


def load_spark_fp8():
    global _ext
    if _ext is None:
        import os
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        from torch.utils.cpp_extension import load
        BUILD.mkdir(parents=True, exist_ok=True)
        _ext = load(name="r11_spark_fp8", sources=[str(Path(__file__).resolve().parent / "spark_fp8_bind.cpp"), str(SPARK / "src/kernels/attention_fp8.cu")],
                    extra_include_paths=[str(SPARK / "include"), str(SPARK / "src/kernels")], extra_cflags=["-O3", "-std=c++20"],
                    extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-std=c++20", "-gencode", "arch=compute_120a,code=sm_120a"],
                    build_directory=str(BUILD), verbose=False)
    return _ext
