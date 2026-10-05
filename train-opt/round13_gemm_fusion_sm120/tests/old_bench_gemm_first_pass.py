"""Round13 GEMM candidate matrix: 4 Linears x {fwd, dgrad, wgrad} at the real training shapes.

Usage (always under the GPU lock, taken inside this script):
  python bench_gemm.py --model vitl16 --m 7203 --cands torch,quack --out profile/gemm_vitl16_7203.json

Layouts (row-major torch tensors; X [M,K], W [N,K], dY [M,N]):
  fwd   : Y  [M,N] = X  @ W^T        (B = W^T view, K-major B)
  dgrad : dX [M,K] = dY @ W          (B = W, N-major B)
  wgrad : dW [N,K] = dY^T @ X        (A = dY^T view, M-major A; contraction over M)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/root/lightly-train/train-opt")
TMP = Path("/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120")
os.environ.setdefault("QUACK_CACHE_DIR", str(TMP / "quack_cache"))
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "round13_gemm_fusion_sm120" / "impl"))

import torch

from common.bench import gpu_lock
from mb import err_stats, time_us

MODELS = {"vitl16": 1024, "vitb16": 768}
M_PLAIN = {"vitl16": 7 * 1029, "vitb16": 14 * 1029}
M_JOINT = {"vitl16": 7 * 1129, "vitb16": 14 * 1129}
PEAK_TFLOPS = 230.0


def linears(d: int):
    # name -> (N_out, K_in)
    return {"qkv": (3 * d, d), "proj": (d, d), "fc1": (4 * d, d), "fc2": (d, 4 * d)}


def build_candidates(names):
    cands = {}
    if "torch" in names:
        cands["torch"] = {
            "fwd": lambda X, W, dY, out: torch.mm(X, W.t(), out=out),
            "dgrad": lambda X, W, dY, out: torch.mm(dY, W, out=out),
            "wgrad": lambda X, W, dY, out: torch.mm(dY.t(), X, out=out),
        }
    if "torch_f32out" in names:
        cands["torch_f32out"] = {
            "wgrad": lambda X, W, dY, out: torch.mm(dY.t(), X, out_dtype=torch.float32),
        }
    if "quack" in names:
        from quack.gemm_interface import gemm
        cands["quack"] = {
            "fwd": lambda X, W, dY, out: gemm(X, W.t(), out=out),
            "dgrad": lambda X, W, dY, out: gemm(dY, W, out=out),
            "wgrad": lambda X, W, dY, out: gemm(dY.t(), X, out=out),
        }
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--m", default="plain")
    ap.add_argument("--cands", default="torch,quack")
    ap.add_argument("--ops", default="qkv,proj,fc1,fc2")
    ap.add_argument("--kinds", default="fwd,dgrad,wgrad")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    d = MODELS[args.model]
    M = {"plain": M_PLAIN[args.model], "joint": M_JOINT[args.model]}.get(args.m) or int(args.m)
    names = args.cands.split(",")
    cands = build_candidates(names)
    results = []
    torch.manual_seed(0)
    with gpu_lock():
        for lname, (N, K) in linears(d).items():
            if lname not in args.ops.split(","):
                continue
            X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            W = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05)
            dY = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
            refs = {
                "fwd": (X.float() @ W.float().t()),
                "dgrad": (dY.float() @ W.float()),
                "wgrad": (dY.float().t() @ X.float()),
            }
            outshape = {"fwd": (M, N), "dgrad": (M, K), "wgrad": (N, K)}
            for kind in args.kinds.split(","):
                flops = 2.0 * M * N * K
                for cname, table in cands.items():
                    if kind not in table:
                        continue
                    fn = table[kind]
                    dtype = torch.float32 if cname == "torch_f32out" else torch.bfloat16
                    out = torch.empty(outshape[kind], device="cuda", dtype=dtype)
                    t0 = time.time()
                    try:
                        res = fn(X, W, dY, out)
                        torch.cuda.synchronize()
                        first_s = time.time() - t0
                        res = out if res is None or cname != "torch_f32out" else res
                        e = err_stats(res, refs[kind])
                        tm = time_us(lambda: fn(X, W, dY, out), graph=not args.no_graph)
                        row = {"model": args.model, "M": M, "linear": lname, "kind": kind, "cand": cname,
                               "N": N, "K": K, "us": tm["us"], "tflops": flops / tm["us"] / 1e6,
                               "pct_230": 100.0 * flops / tm["us"] / 1e6 / PEAK_TFLOPS, "mode": tm["mode"],
                               "first_call_s": first_s, "err": e}
                    except Exception as exc:
                        row = {"model": args.model, "M": M, "linear": lname, "kind": kind, "cand": cname,
                               "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                    results.append(row)
                    if "us" in row:
                        print(f"{args.model} M={M} {lname:4s} {kind:5s} {cname:13s} {row['us']:9.1f} us "
                              f"{row['tflops']:6.1f} TF {row['pct_230']:5.1f}% cos={row['err']['cos']:.6f} "
                              f"rel={row['err']['max_rel_to_absmax']:.2e} [{row['mode']}] first={row['first_call_s']:.1f}s",
                              flush=True)
                    else:
                        print(f"{args.model} M={M} {lname:4s} {kind:5s} {cname:13s} ERROR {row['error']}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
