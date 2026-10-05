"""R21b: the head's nv4 GEMM configurations on the real ConvT shapes (CUDA-graph replay, candidates interleaved in one process, 3 rounds, median; dual GPU lock).

dgrad  dX [M, C] bf16 = dZ.q [M, 4C/2] . W.qt [C, 4C/2]^T     candidates: QuACK ``plain_ts`` epilogue (device-scalar ts), QuACK BlockScaledOperand (per_tensor_scale), cuBLASLt scaled_mm, bf16 r14 (reference)
wgrad  dW [4C, C] fp32  = dZ.qt [4C, Mp/2] . X.qt [C, Mp/2]^T  same + QuACK split_k (autotuned) + cuBLASLt fp32 out + bf16 r14 (cuBLAS fp32 out)
  python tests/bench_head_gemm.py [--models vitl16,vitb16]     -> profile/bench_head_gemm.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))

import torch  # noqa: E402

from common.bench import assert_gpu_exclusive, gpu_lock  # noqa: E402
from round14_head_sm120.impl import head as h14  # noqa: E402
from round20_fp4_probe.impl import fp4_cands as C  # noqa: E402
from round21_fp4_quant_head.impl import fp4q, head_fp4  # noqa: E402

dev = "cuda"
SHAPES = {"vitl16": (1024, (7168, 28672)), "vitb16": (768, (14336, 57344))}


def make_graph(fn, inner=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    return g


def interleave(cands, rounds=3, reps=10, inner=5):
    graphs, errors = {}, {}
    for n, f in cands.items():
        try:
            graphs[n] = make_graph(f, inner)
        except Exception as e:  # noqa: BLE001
            errors[n] = f"{type(e).__name__}: {str(e)[:200]}"
            torch.cuda.synchronize()
    times = {n: [] for n in graphs}
    order = list(graphs)
    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for r in range(rounds):
        for n in (order if r % 2 == 0 else order[::-1]):
            g = graphs[n]
            for _ in range(2):
                g.replay()
            torch.cuda.synchronize()
            ev[0].record()
            for _ in range(reps):
                g.replay()
            ev[1].record()
            torch.cuda.synchronize()
            times[n].append(ev[0].elapsed_time(ev[1]) * 1000.0 / (reps * inner))
    return {n: round(statistics.median(t), 2) for n, t in times.items()}, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vitl16,vitb16")
    args = ap.parse_args()
    out = {"cases": {}}
    g = torch.Generator(device=dev).manual_seed(0)
    with gpu_lock():
        for _ in range(45):
            try:
                assert_gpu_exclusive()
                break
            except RuntimeError:
                time.sleep(2)
        for model in args.models.split(","):
            Cc, Ms = SHAPES[model]
            N = 4 * Cc
            for ci, M in enumerate(Ms):
                name = f"{model}/convT{ci + 1}"
                dZ = (torch.randn(M, N, device=dev, generator=g) * 1e-3).to(torch.bfloat16)
                X = torch.randn(M, Cc, device=dev, generator=g).to(torch.bfloat16)
                Wm = (torch.randn(N, Cc, device=dev, generator=g) * 0.02).to(torch.bfloat16)
                st = fp4q.FP4State(4)
                tw = fp4q.quant_weight(Wm, "nv4", amax="exact")
                tx = fp4q.quant_act(X, "nv4", row=False, col=True, amax="exact")
                td = fp4q.quant_act(dZ, "nv4", row=True, col=True, sr_row=True, sr_col=True, amax="exact", seed=st.seed)
                dx = torch.empty(M, Cc, device=dev, dtype=torch.bfloat16)
                dw = torch.empty(N, Cc, device=dev, dtype=torch.float32)
                E = head_fp4.fp4_epi.nv4_epilogues()
                from quack.gemm_interface import gemm as qgemm
                ts_d, ts_w = (td.ts * tw.tst).reshape(1), (td.tst * tx.tst).reshape(1)
                a_d, b_d = td.q.view(torch.float4_e2m1fn_x2), tw.qt.view(torch.float4_e2m1fn_x2).t()
                a_w, b_w = td.qt.view(torch.float4_e2m1fn_x2), tx.qt.view(torch.float4_e2m1fn_x2).t()
                kd, kw = C.epi_kw(td.sf, tw.sft, "nv4"), C.epi_kw(td.sft, tx.sft, "nv4")
                dgrad = {
                    "dgrad/epi_plain_ts": lambda: E["plain"](a_d, b_d, out={"D": dx}, ts=ts_d, **kd),
                    "dgrad/operand": lambda: head_fp4.nv4_gemm(td.q, td.sf, td.ts, tw.qt, tw.sft, tw.tst, dx, "operand"),
                    "dgrad/cublaslt": lambda: C.scaled_mm_fp4((td.q, td.sf, td.ts), (tw.qt, tw.sft, tw.tst), "nv4"),
                    "dgrad/bf16_r14": lambda: h14.mm_dgrad(dZ, Wm),
                }
                opA, opB = C.operand((td.qt, td.sft, td.tst), "nv4"), C.operand((tx.qt, tx.sft, tx.tst), "nv4")
                wgrad = {
                    "wgrad/epi_plain_ts": lambda: E["plain"](a_w, b_w, out={"D": dw}, ts=ts_w, **kw),
                    "wgrad/operand": lambda: head_fp4.nv4_gemm(td.qt, td.sft, td.tst, tx.qt, tx.sft, tx.tst, dw, "operand"),
                    "wgrad/operand_splitk": lambda: qgemm(opA, opB.mT, out=dw, split_k=None),
                    "wgrad/cublaslt_bf16out": lambda: C.scaled_mm_fp4((td.qt, td.sft, td.tst), (tx.qt, tx.sft, tx.tst), "nv4", torch.float32),
                    "wgrad/bf16_r14": lambda: h14.mm_wgrad(dZ, X),
                }
                us, errs = interleave({**dgrad, **wgrad})
                flops = 2.0 * M * N * Cc
                out["cases"][name] = {"M": M, "N": N, "K_dgrad": N, "K_wgrad": td.kt, "us": us, "tflops": {k: round(flops / v / 1e6, 1) for k, v in us.items()}, "errors": errs}
                print(name, json.dumps(us), "errors", errs, flush=True)
                del dZ, X, Wm, tw, tx, td, dx, dw
                torch.cuda.empty_cache()
    (ROUND / "profile" / "bench_head_gemm.json").write_text(json.dumps(out, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
