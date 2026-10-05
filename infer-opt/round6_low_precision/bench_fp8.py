"""Stage 2: FP8 performance.

  python bench_fp8.py --gemm                       # GEMM / quant-kernel microbenchmarks (both models' shapes)
  python bench_fp8.py --model vitl16 --units greedy_vitl16_row.json   # whole-model, CUDA graph, real FP8
  python bench_fp8.py --model vitl16 --units preset:all-mlp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "impl"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "round7_head_postprocess"))

import model_factory as mf  # noqa: E402
from bench import env_info, time_fn, write_results  # noqa: E402
import fp8linear as fl  # noqa: E402

OUT = HERE / "results.json"
FP8 = torch.float8_e4m3fn
SHAPES = {  # name: (N, K)
    "vitl16": {"qkv": (3072, 1024), "proj": (1024, 1024), "fc1": (4096, 1024), "fc2": (1024, 4096)},
    "vitb16": {"qkv": (2304, 768), "proj": (768, 768), "fc1": (3072, 768), "fc2": (768, 3072)},
}
REPS = 20


def t_gpu(fn, reps=REPS):
    def run():
        for _ in range(reps):
            fn()
    r = time_fn(run, warmup=5, iters=30, repeats=3)
    return r["gpu_ms_p50"] / reps


def gemm_bench():
    e = fl.ext()
    res = {}
    for model, shapes in SHAPES.items():
        for M in (2058, 2258):
            for name, (N, K) in shapes.items():
                x = torch.randn(M, K, device="cuda").bfloat16()
                w = torch.randn(N, K, device="cuda").bfloat16()
                b = torch.randn(N, device="cuda").bfloat16()
                wq, sw = fl.quant_weight(w)
                qx = torch.empty(M, K, dtype=FP8, device="cuda")
                sx = torch.empty(M, 1, device="cuda")
                e.quant_rows(x, qx, sx)
                flops = 2 * M * N * K
                t_bf = t_gpu(lambda: F.linear(x, w, b))
                t_row = t_gpu(lambda: torch._scaled_mm(qx, wq.t(), scale_a=sx, scale_b=sw, bias=b, out_dtype=torch.bfloat16))
                s1 = torch.ones((), device="cuda")
                t_ten = t_gpu(lambda: torch._scaled_mm(qx, wq.t(), scale_a=s1, scale_b=s1, bias=b, out_dtype=torch.bfloat16))
                t_q = t_gpu(lambda: e.quant_rows(x, qx, sx))
                # unfused torch quantization (absmax, scale, div, clamp, cast)
                def torch_quant():
                    s = (x.float().abs().amax(-1, keepdim=True) / 448.0)
                    return (x.float() / s).clamp(-448, 448).to(FP8), s
                t_tq = t_gpu(torch_quant)
                r = {"M": M, "N": N, "K": K, "bf16_ms": t_bf, "fp8_rowwise_gemm_ms": t_row, "fp8_tensorwise_gemm_ms": t_ten,
                     "quant_kernel_ms": t_q, "torch_quant_ms": t_tq,
                     "bf16_tflops": flops / t_bf / 1e9, "fp8_row_tflops": flops / t_row / 1e9, "fp8_ten_tflops": flops / t_ten / 1e9,
                     "net_speedup_with_fused_quant": t_bf / (t_row + t_q)}
                res[f"{model}/{name}/M{M}"] = r
                print(f"{model:6s} {name:4s} M={M} bf16 {t_bf*1e3:6.1f}us ({r['bf16_tflops']:5.0f}TF) | fp8 row {t_row*1e3:6.1f}us ({r['fp8_row_tflops']:5.0f}TF) "
                      f"ten {t_ten*1e3:6.1f}us | quant {t_q*1e3:5.1f}us (torch {t_tq*1e3:5.1f}us) | net x{r['net_speedup_with_fused_quant']:.2f}", flush=True)
    # fused LN+quant / GELU+quant kernels vs their unfused torch equivalents
    for model, K in (("vitl16", 1024), ("vitb16", 768)):
        M = 2058
        x = torch.randn(M, K, device="cuda")
        g = torch.randn(K, device="cuda")
        bb = torch.randn(K, device="cuda")
        q = torch.empty(M, K, dtype=FP8, device="cuda")
        s = torch.empty(M, 1, device="cuda")
        t_f = t_gpu(lambda: e.ln_quant(x, g, bb, q, s, 1e-5))
        t_ln = t_gpu(lambda: F.layer_norm(x, (K,), g, bb, 1e-5))
        y = torch.randn(M, 4 * K, device="cuda").bfloat16()
        q4 = torch.empty(M, 4 * K, dtype=FP8, device="cuda")
        s4 = torch.empty(M, 1, device="cuda")
        t_g = t_gpu(lambda: e.gelu_quant(y, q4, s4))
        t_gl = t_gpu(lambda: F.gelu(y))
        print(f"{model}: ln_quant {t_f*1e3:.1f}us (torch LN alone {t_ln*1e3:.1f}us) | gelu_quant {t_g*1e3:.1f}us (torch GELU alone {t_gl*1e3:.1f}us)", flush=True)
        res[f"{model}/ln_quant_ms"] = t_f
        res[f"{model}/torch_ln_ms"] = t_ln
        res[f"{model}/gelu_quant_ms"] = t_g
        res[f"{model}/torch_gelu_ms"] = t_gl
    write_results(OUT, {"gemm_microbench": res, "env": env_info()})


def load_units(spec, L):
    if spec.startswith("preset:"):
        p = spec.split(":", 1)[1]
        types = {"all-mlp": ["fc1", "fc2"], "all-proj": ["proj"], "all4": ["qkv", "proj", "fc1", "fc2"],
                 "fc2": ["fc2"], "fc1": ["fc1"], "mlp-half2": ["fc1", "fc2"]}[p]
        layers = range(L // 2, L) if p == "mlp-half2" else range(L)
        return [(i, t) for i in layers for t in types]
    d = json.loads(Path(spec).read_text())
    return [tuple(u) for u in d["chosen_units"]]


def capture(model, batch, precision="bf16-mixed"):
    """CUDA-graph the eager forward (static input); returns (replay_fn, outputs)."""
    def fwd():
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            ml, cl = model.forward_train(batch, return_logits_per_layer=False)
        return ml[-1], cl[-1]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3):
            fwd()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g):
        out = fwd()
    return g.replay, out, fwd


def model_bench(name, units_spec, tag):
    import sens
    import head as hd
    ev = sens.Evaluator(name)
    model = ev.model
    L = ev.L
    units = load_units(units_spec, L)
    batch = ev.batches[0].clone()
    res = {"n_units": len(units), "units_spec": units_spec, "batch_shape": list(batch.shape)}
    variants = {}
    # baseline eager / graph
    replay, out, fwd = capture(model, batch)
    variants["bf16 eager"] = t = time_fn(lambda: fwd(), warmup=20, iters=100)
    variants["bf16 graph"] = time_fn(replay, warmup=20, iters=100)
    hd.enable(model)
    replay_h, _, fwd_h = capture(model, batch)
    variants["bf16+fusedhead graph"] = time_fn(replay_h, warmup=20, iters=100)
    fl.enable(model, units)
    replay_f, _, fwd_f = capture(model, batch)
    variants["fp8+fusedhead graph"] = time_fn(replay_f, warmup=20, iters=100)
    variants["fp8+fusedhead eager"] = time_fn(lambda: fwd_f(), warmup=20, iters=100)
    acc = ev.run()
    hd.disable(model)
    fl.disable(model)
    for k, v in variants.items():
        res[k] = {"wall_p50": v["wall_ms_p50"], "gpu_p50": v["gpu_ms_p50"]}
        print(f"[{name}/{tag}] {k:24s} wall p50 {v['wall_ms_p50']:7.3f} ms | gpu p50 {v['gpu_ms_p50']:7.3f} ms", flush=True)
    print(f"[{name}/{tag}] accuracy (real FP8 + fused head): {acc}", flush=True)
    res["accuracy_8f_real_fp8"] = acc
    write_results(OUT, {name: {tag: {"latency_ms": {"model": {"wall_p50": res["fp8+fusedhead graph"]["wall_p50"],
                                                             "gpu_p50": res["fp8+fusedhead graph"]["gpu_p50"]}},
                                     "bench": res}}, "env": env_info()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--units")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.gemm:
        gemm_bench()
    if args.model:
        model_bench(args.model, args.units, args.tag or f"fp8:{Path(args.units).stem}")


if __name__ == "__main__":
    main()
