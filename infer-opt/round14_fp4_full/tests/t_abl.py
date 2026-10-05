"""Round 14 step 2 experiment: bottleneck ablations of the FP4 GEMM main loop (fp4g_abl.cu).  Timing only (results are garbage by construction).
  python t_abl.py <model> <fmt> <kind>      under the dual lock, cold weights (>= 192 MB of weight copies)"""
import hashlib, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "impl"))
sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import torch
import fp4gb
import t_gemm as T
from lock2 import gpu_lock2

model, fmt, kind = sys.argv[1], sys.argv[2], sys.argv[3]
fi = 0 if fmt == "nv4" else 1
cfg = dict(bm=128, bn=128, bkb=64, st=2, wm=2, wn=4, minb=2)
if len(sys.argv) > 4:
    cfg.update(eval(sys.argv[4]))
M = 2058
N, K = T.kind_nk2(model, kind)
BUILD = Path(fp4gb.BUILD) / "abl"
BUILD.mkdir(parents=True, exist_ok=True)


def make(defs):
    c = dict(fp4gb.DEFAULT, **cfg, fmt=fi, epi=0)
    src = Path(fp4gb._HERE) / "fp4g_abl.cu"
    h = hashlib.md5(src.read_bytes() + repr(sorted(c.items())).encode() + repr(defs).encode()).hexdigest()[:10]
    out = str(BUILD / f"abl_{h}.cubin")
    if not os.path.exists(out):
        d = [f"-D{k.upper()}={v}" for k, v in c.items()] + [f"-D{x}" for x in defs]
        r = subprocess.run([fp4gb.NVCC, "-arch=sm_120a", "-O3", "-std=c++17", "-cubin", "-o", out, str(src)] + d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
    k = fp4gb.Kernel.__new__(fp4gb.Kernel)
    k.cfg, k.log, k.threads, k.smem = c, "", c["wm"] * c["wn"] * 32, c["st"] * fp4gb.stage_bytes(c)
    k.h = fp4gb.launcher().load(out, "gemm_kernel", k.smem)
    return k


nW = int(min(256, max(8, -(-192e6 // (N * K // 2)))))
X, xq, ws0 = T.operands(fmt, M, N, K, nW=1)
w0 = ws0[0]
ws = [(w0[0].clone(), w0[1].clone()) for _ in range(nW)]
ts = T.tsprod(xq, w0)
out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
variants = {"STG epilogue only": ["ABL_STG", "ABL_NOLOOP"], "STG full": ["ABL_STG"], "empty kernel": ["ABL_EMPTY"], "epilogue only (no k loop)": ["ABL_NOLOOP"], "full": [], "no_scale_lds": ["ABL_NOSF"], "no_ldmatrix": ["ABL_NOLDM"], "no_cp_async": ["ABL_NOCP"], "no_mma": ["ABL_NOMMA"],
            "mma_only(no sf,ldm,cp)": ["ABL_NOSF", "ABL_NOLDM", "ABL_NOCP"], "no_mma_no_cp": ["ABL_NOMMA", "ABL_NOCP"]}
kernels = {k: make(v) for k, v in variants.items()}
with gpu_lock2():
    for name, kern in kernels.items():
        t = T.bench([(lambda w=w: kern(xq[0], w[0], xq[1], w[1], M=M, N=N, K=K, ts=ts, C=out)) for w in ws], reps=6)
        print(f"{model} {fmt} {kind} {cfg} {name:24s}: {t:6.1f} us", flush=True)
