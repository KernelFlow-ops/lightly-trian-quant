"""Round13 GEMM candidate matrix (final, same-machine): 4 Linears x {fwd, dgrad, wgrad}, plain and joint M.

Layouts (row-major torch tensors; X [M,K], W [N,K], dY [M,N]); wgrad writes FP32 (the Parameter-gradient dtype):
  fwd   : Y  [M,N] bf16 = X  @ W^T
  dgrad : dX [M,K] bf16 = dY @ W
  wgrad : dW [N,K] fp32 = dY^T @ X
Candidates: torch (cuBLAS/cuBLASLt via torch.mm), quack (autotuned SM120 GEMM), quack_sk (wgrad with autotuned
split-K, serial/deterministic), spark4/spark6 (Spark hgemm, row-major NN only; fwd uses a pre-transposed W^T copy,
wgrad not supported by the kernel: contraction over M needs a transposed dY).
Timing: CUDA graph of 5 calls, 3 round-robin rounds x 30 calls, median; same process, same lock hold.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

TMP = Path("/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120")
os.environ.setdefault("QUACK_CACHE_DIR", str(TMP / "quack_cache"))
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "impl"))

import torch

from common.bench import gpu_lock
from mb import err_stats, time_us

MODELS = {"vitl16": 1024, "vitb16": 768}
M_PLAIN = {"vitl16": 7 * 1029, "vitb16": 14 * 1029}
M_JOINT = {"vitl16": 7 * 1129, "vitb16": 14 * 1129}
PEAK = 230.0


def candidates(names):
    c = {}
    if "torch" in names:
        c["torch"] = {
            "fwd": lambda X, W, dY, o: torch.mm(X, W.t(), out=o),
            "dgrad": lambda X, W, dY, o: torch.mm(dY, W, out=o),
            "wgrad": lambda X, W, dY, o: torch.mm(dY.t(), X, out_dtype=torch.float32),
        }
    if "quack" in names:
        from quack.gemm_interface import gemm
        c["quack"] = {
            "fwd": lambda X, W, dY, o: gemm(X, W.t(), out=o),
            "dgrad": lambda X, W, dY, o: gemm(dY, W, out=o),
            "wgrad": lambda X, W, dY, o: gemm(dY.t(), X, out=o),
        }
    if "quack_sk" in names:
        from quack.gemm_interface import gemm
        c["quack_sk"] = {"wgrad": lambda X, W, dY, o: gemm(dY.t(), X, out=o, split_k=None)}
    for sv in (4, 6):
        if f"spark{sv}" in names:
            import spark_hgemm_ext
            ext = spark_hgemm_ext.extension()
            cache = {}

            def wt(W, cache=cache):
                key = (W.data_ptr(), tuple(W.shape))
                if key not in cache:
                    cache[key] = W.t().contiguous()
                return cache[key]

            def mk(sv=sv, ext=ext, wt=wt):
                def fwd(X, W, dY, o):
                    ext.hgemm(X, wt(W), o, sv, None, 0, None)
                    return o

                def dgrad(X, W, dY, o):
                    ext.hgemm(dY, W, o, sv, None, 0, None)
                    return o
                return {"fwd": fwd, "dgrad": dgrad}
            c[f"spark{sv}"] = mk()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--m", default="plain")
    ap.add_argument("--cands", default="torch,quack,quack_sk,spark6,spark4")
    ap.add_argument("--ops", default="qkv,proj,fc1,fc2")
    ap.add_argument("--kinds", default="fwd,dgrad,wgrad")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = MODELS[args.model]
    M = {"plain": M_PLAIN[args.model], "joint": M_JOINT[args.model]}.get(args.m) or int(args.m)
    cands = candidates(args.cands.split(","))
    lin = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (4 * d, d), "fc2": (d, 4 * d)}
    rows = []
    torch.manual_seed(0)
    with gpu_lock():
        for lname, (N, K) in lin.items():
            if lname not in args.ops.split(","):
                continue
            X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
            dY = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
            refs = {"fwd": X.float() @ W.float().t(), "dgrad": dY.float() @ W.float(), "wgrad": dY.float().t() @ X.float()}
            shapes = {"fwd": (M, N), "dgrad": (M, K), "wgrad": (N, K)}
            for kind in args.kinds.split(","):
                flops = 2.0 * M * N * K
                odt = torch.float32 if kind == "wgrad" else torch.bfloat16
                outs = {n: torch.empty(shapes[kind], device="cuda", dtype=odt) for n, t in cands.items() if kind in t}
                live = {}
                for cname, table in cands.items():
                    if kind not in table:
                        continue
                    try:
                        res = table[kind](X, W, dY, outs[cname])
                        torch.cuda.synchronize()
                        o = res if (res is not None and res.dtype == odt and res.shape == outs[cname].shape and cname == "torch" and kind == "wgrad") else outs[cname]
                        live[cname] = (table[kind], err_stats(o, refs[kind]))
                    except Exception as exc:
                        rows.append({"model": args.model, "M": M, "linear": lname, "kind": kind, "cand": cname, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                        print(f"{args.model} M={M} {lname:4s} {kind:5s} {cname:9s} ERROR {str(exc)[:120]}", flush=True)
                times = {c: [] for c in live}
                for _ in range(args.rounds):
                    for cname, (fn, _e) in live.items():
                        times[cname].append(time_us(lambda: fn(X, W, dY, outs[cname]), iters=30, repeats=1)["us"])
                for cname, (fn, e) in live.items():
                    us = statistics.median(times[cname])
                    row = {"model": args.model, "M": M, "linear": lname, "kind": kind, "cand": cname, "N": N, "K": K, "us": us,
                           "us_min": min(times[cname]), "tflops": flops / us / 1e6, "pct_230": 100 * flops / us / 1e6 / PEAK, "err": e}
                    rows.append(row)
                best = min((r for r in rows if r.get("linear") == lname and r.get("kind") == kind and "us" in r and r["M"] == M), key=lambda r: r["us"])
                print(f"{args.model} M={M} {lname:4s} {kind:5s} " + " | ".join(
                    f"{r['cand']} {r['us']:.1f}us/{r['tflops']:.0f}TF" for r in rows if r.get("linear") == lname and r.get("kind") == kind and "us" in r and r["M"] == M)
                    + f"  -> best {best['cand']}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
