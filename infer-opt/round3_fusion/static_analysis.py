"""Static analysis (ncu substitute): dump the TileLang-generated CUDA for the chosen kernels, compile with
ptxas -v and count SASS instructions (HMMA / LDGSTS / UTMALDG / LDSM ...) via common/kernel_static.sh.
Also analyzes the hand-written LayerNorm kernel. Output: /root/lightly-train/infer-opt/round3_fusion/static/
"""
import subprocess
import sys

sys.path.insert(0, "/root/lightly-train/infer-opt/round3_fusion")
from pathlib import Path

import tilelang
from impl import gemm_tl, ops

OUT = Path("/root/lightly-train/infer-opt/round3_fusion/static")
OUT.mkdir(exist_ok=True)
TL = Path(tilelang.__file__).parent
INC = [f"-I{TL}/src", f"-I{TL}/3rdparty/cutlass/include"]
M, B, N = 2058, 2, 1029
T = ops.TILE_CFG
jobs = {
    "k4_gemm_bias_gelu_vitl": gemm_tl.gemm_bias_gelu_kernel(M, 4096, 1024, **T["gelu_4096_1024"]),
    "k3_fc2_vitl": gemm_tl.gemm_bias_residual_kernel(M, 1024, 4096, res_dtype="float32", **T["res_1024_4096"]),
    "k3_proj_vitl": gemm_tl.gemm_bias_residual_kernel(M, 1024, 1024, res_dtype="float32", **T["res_1024_1024"]),
    "k2_qkv_rope_vitl": gemm_tl.qkv_rope_kernel(B, N, 1024, 16, 5, **T["qkv_1024"]),
}
ln_so = ops._ln().__file__  # hand-written K1: SASS from the built extension
subprocess.run(["bash", "/root/lightly-train/infer-opt/common/kernel_static.sh", "bin", ln_so, str(OUT)],
               capture_output=True, text=True)
ru = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", ln_so], capture_output=True, text=True)
print("===== k1 add_layernorm (extension .so)")
print("\n".join(l for l in ru.stdout.splitlines() if "Function" in l or "REG" in l)[:2500])
print(open(OUT / (Path(ln_so).stem + ".static.txt")).read()[-800:])
for name, k in jobs.items():
    src = OUT / f"{name}.cu"
    src.write_text(k.get_kernel_source())
    cubin = OUT / f"{name}.cubin"
    r = subprocess.run(["/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", "-arch=sm_120a", "-cubin", "-Xptxas", "-v",
                        "-DENABLE_BF16", *INC, "-o", str(cubin), str(src)], capture_output=True, text=True)
    (OUT / f"{name}.ptxas.txt").write_text(r.stderr)
    if r.returncode != 0:
        print(name, "nvcc FAILED", r.stderr[-800:])
        continue
    o = subprocess.run(["bash", "/root/lightly-train/infer-opt/common/kernel_static.sh", "bin", str(cubin), str(OUT)],
                       capture_output=True, text=True)
    print("=====", name)
    print(o.stdout[-1500:], o.stderr[-300:])
