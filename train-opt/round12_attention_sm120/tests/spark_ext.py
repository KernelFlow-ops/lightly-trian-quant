"""Build/load the Spark attention kernels as a torch extension (candidate-matrix use only)."""
import os
from pathlib import Path

from torch.utils.cpp_extension import load

SPARK = Path("/root/autodl-tmp/train-opt-tmp/third_party/spark")
BUILD = Path("/root/autodl-tmp/train-opt-tmp/round12_attention_sm120/build/spark_attn")


def load_spark():
    BUILD.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    return load(
        name="spark_attn_ext",
        sources=[str(here / "bind_spark_attn.cpp")] + [str(SPARK / "src/kernels" / f) for f in ("attention.cu", "attention_decode.cu", "attention_bwd.cu")],
        extra_include_paths=[str(SPARK / "include")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-std=c++20", "-gencode", "arch=compute_120a,code=sm_120a"],
        build_directory=str(BUILD),
        verbose=True,
    )


if __name__ == "__main__":
    m = load_spark()
    print("fwd variants", m.fwd_variants(), "bwd variants", m.bwd_variants())
