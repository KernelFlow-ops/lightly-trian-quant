"""Activation-quantiser cost matrix (graph replay, interleaved, median of 3): what a standalone quantisation pass costs at our token counts, to be compared with the
producer it would be fused into (R5 `add_ln_kernel` 4.7 us / fc1 GELU epilogue / flash output).

sites:  ln   LN output feeding qkv / fc1 (K = d)   : candidates take the fp32 residual (LN+quant kernels) or the bf16 LN output (pure quantisers)
        o    attention output feeding proj (K = d) : bf16 in
        gelu GELU output feeding fc2 (K = ff)      : bf16 in (gelu_quant takes the fc1 pre-activation)
  python quant_matrix.py [--Ms 2058,2258] [--models vitl16,vitb16]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl"))

import torch  # noqa: E402

import bench10 as b  # noqa: E402
import cands_base as cb  # noqa: E402
import cands_mx8  # noqa: E402,F401

BF = torch.bfloat16
TMP = b.TMP
_M = {}


def r6ext():
    if "r6" not in _M:
        from torch.utils.cpp_extension import load
        bd = TMP / "build" / "r6_mx_ext"
        bd.mkdir(parents=True, exist_ok=True)
        _M["r6"] = load(name="r10_r6_mx_ext", sources=["/root/lightly-train/infer-opt/round6_low_precision/impl/mx_ext.cu"],
                        extra_cuda_cflags=["-O3", "-arch=sm_120a"], build_directory=str(bd), verbose=False)
    return _M["r6"]


def fp4q():
    if "fp4q" not in _M:
        from round24_fp4_quant_opt.impl import fp4q2
        fp4q2._BUILD = TMP / "build" / "r24_fp4q2"
        _M["fp4q"] = fp4q2
    return _M["fp4q"]


def qdata(s):
    M, K = s["M"], s["K"]
    x32 = torch.randn(M, K, device="cuda") * 0.7
    return M, K, x32, x32.to(BF)


def c_ln_add_r5(s):
    M, K, x32, xb = qdata(s)
    o = cb.r5()
    g = torch.rand(K, device="cuda"); bb = torch.randn(K, device="cuda") * 0.1
    x3 = x32.view(2, -1, K)
    return [lambda: o.add_layernorm(x3, None, g, bb, 1e-5)]


def c_ln_quant_r6(s):
    M, K, x32, xb = qdata(s)
    if K > 1024:
        return None
    e = r6ext()
    q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device="cuda")
    sc = torch.empty(((M + 127) // 128) * 128 * (K // 32), dtype=torch.float8_e8m0fnu, device="cuda")
    g = torch.rand(K, device="cuda"); bb = torch.randn(K, device="cuda") * 0.1
    return [lambda: e.ln_quant(x32, g, bb, q, sc, 1e-5)]


def c_quant_rows_r6(s):
    M, K, x32, xb = qdata(s)
    e = r6ext()
    q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device="cuda")
    sc = torch.empty(((M + 127) // 128) * 128 * (K // 32), dtype=torch.float8_e8m0fnu, device="cuda")
    return [lambda: e.quant_rows(xb, q, sc)]


def c_gelu_quant_r6(s):
    M, K, x32, xb = qdata(s)
    e = r6ext()
    q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device="cuda")
    sc = torch.empty(((M + 127) // 128) * 128 * (K // 32), dtype=torch.float8_e8m0fnu, device="cuda")
    return [lambda: e.gelu_quant(xb, q, sc)]


def c_quant_dual_r17(s):
    M, K, x32, xb = qdata(s)
    mx8, _ = cands_mx8.r17()
    return [lambda: mx8.quantize(xb, row=True, col=False)]


def c_spark_fp8mx(s):
    M, K, x32, xb = qdata(s)
    ext = cands_mx8.spark()
    q = torch.empty(M, K, dtype=torch.uint8, device="cuda")
    sc = torch.empty(M, K // 32, dtype=torch.uint8, device="cuda")
    return [lambda: ext.fp8_quantize_mx(xb, q, sc)]


def c_int8_rows(s):
    import cands_epi
    M, K, x32, xb = qdata(s)
    e = cands_epi.ext()
    q = torch.empty(M, K, dtype=torch.int8, device="cuda")
    sc = torch.empty(M, dtype=torch.float32, device="cuda")
    return [lambda: e.quant_int8(xb, q, sc)]


def c_fp4q2(fmt, mode):
    def make(s):
        M, K, x32, xb = qdata(s)
        f = fp4q()
        return [lambda: f.quant_act(xb, fmt, row=True, col=False, amax=mode)]
    return make


def c_spark_fp4q(fmt):
    def make(s):
        M, K, x32, xb = qdata(s)
        ext = cands_mx8.spark()
        vec = 16 if fmt == "nv4" else 32
        nbytes = ext.scale_bytes(M, K, 0 if fmt == "nv4" else 1)
        q = torch.empty(M, K // 2, dtype=torch.uint8, device="cuda")
        sf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        work = torch.zeros(1, dtype=torch.int32, device="cuda")
        so = torch.empty(1, dtype=torch.float32, device="cuda")
        if fmt == "nv4":
            def f():
                work.zero_()
                ext.fp4quant(xb, q, sf, None, 0, work, so)
        else:
            def f():
                ext.fp4quant(xb, q, sf, None, 1, None, None)
        return [f]
    return make


SITES = {
    "ln": ["ln_add_r5", "ln_quant_r6", "int8_rows_r10", "quant_rows_r6", "quant_dual_r17", "spark_fp8mx", "fp4q2_nv4_exact", "fp4q2_mx4", "spark_nv4", "spark_mx4"],
    "o": ["int8_rows_r10", "quant_rows_r6", "quant_dual_r17", "spark_fp8mx", "fp4q2_nv4_exact", "fp4q2_mx4", "spark_nv4", "spark_mx4"],
    "gelu": ["gelu_quant_r6", "int8_rows_r10", "quant_rows_r6", "quant_dual_r17", "spark_fp8mx", "fp4q2_nv4_exact", "fp4q2_mx4", "spark_nv4", "spark_mx4"],
}
MAKERS = {"int8_rows_r10": c_int8_rows, "ln_add_r5": c_ln_add_r5, "ln_quant_r6": c_ln_quant_r6, "quant_rows_r6": c_quant_rows_r6, "gelu_quant_r6": c_gelu_quant_r6,
          "quant_dual_r17": c_quant_dual_r17, "spark_fp8mx": c_spark_fp8mx, "fp4q2_nv4_exact": c_fp4q2("nv4", "exact"), "fp4q2_nv4_delayed": c_fp4q2("nv4", "delayed"),
          "fp4q2_mx4": c_fp4q2("mx4", "exact"), "spark_nv4": c_spark_fp4q("nv4"), "spark_mx4": c_spark_fp4q("mx4")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", default="2058,2258")
    ap.add_argument("--models", default="vitl16,vitb16")
    ap.add_argument("--out", default=str(HERE / "results_quant.json"))
    a = ap.parse_args()
    print("other gpu procs:", b.other_gpu_procs(), flush=True)
    for model in a.models.split(","):
        d, ff = b.MODELS[model]["d"], b.MODELS[model]["ff"]
        for site, K in (("ln", d), ("o", d), ("gelu", ff)):
            for M in [int(x) for x in a.Ms.split(",")]:
                s = dict(model=model, kind=site, M=M, N=K, K=K)
                cands = [b.Cand(n, MAKERS[n], "", repeat=16) for n in SITES[site]]
                t0 = time.time()
                res = b.time_shape(s, cands, rounds=3, verbose=True)
                for n, v in res.items():
                    v.pop("tflops", None)
                b.merge_results(Path(a.out), {f"{model}|{site}|{M}": res})
                print(f"{model}|{site}|{M}: " + "  ".join(f"{n}={v['us']:.1f}" if "us" in v else f"{n}=ERR" for n, v in res.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
