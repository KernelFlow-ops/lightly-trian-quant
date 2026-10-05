"""Explicit QuACK sm_120 GemmConfig sweep for the training GEMMs (one process, under the GPU lock).

For every (linear, kind) it times every config of quack.gemm_config._get_sm120_configs through
quack.gemm_interface._gemm_execute (graph replay) and prints the ranking, so the best config can
be pinned in impl/gemm_cfg.py instead of autotuning at run time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TMP = Path("/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120")
os.environ.setdefault("QUACK_CACHE_DIR", str(TMP / "quack_cache"))
sys.path.insert(0, "/root/lightly-train/train-opt")
sys.path.insert(0, "/root/lightly-train/train-opt/round13_gemm_fusion_sm120/impl")

import torch

from common.bench import gpu_lock
from mb import time_us
from quack.gemm_config import _get_sm120_configs
from quack.gemm_interface import _gemm_execute
from quack.rounding import RoundingMode

MODELS = {"vitl16": 1024, "vitb16": 768}
M_PLAIN = {"vitl16": 7 * 1029, "vitb16": 14 * 1029}
M_JOINT = {"vitl16": 7 * 1129, "vitb16": 14 * 1129}


def run_cfg(A, B, out, cfg):
    return _gemm_execute(
        A, B, out, None, bias=None, alpha=1.0, beta=1.0, cu_seqlens_m=None, cu_seqlens_k=None, A_idx=None,
        batch_idx_permute=None, add_to_output=False, dynamic_scheduler=cfg.is_dynamic_persistent, config=cfg,
        rounding_mode=RoundingMode.RN, sr_seed=0, concat_layout=None, SFA=None, SFB=None, split_k=1,
        split_k_mode=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--m", default="plain")
    ap.add_argument("--ops", default="qkv,proj,fc1,fc2")
    ap.add_argument("--kinds", default="fwd,dgrad,wgrad")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dtype", default="bf16")
    args = ap.parse_args()
    d = MODELS[args.model]
    M = {"plain": M_PLAIN[args.model], "joint": M_JOINT[args.model]}.get(args.m) or int(args.m)
    lin = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (4 * d, d), "fc2": (d, 4 * d)}
    cfgs = _get_sm120_configs()
    odt = torch.float32 if args.out_dtype == "fp32" else torch.bfloat16
    results = []
    torch.manual_seed(0)
    with gpu_lock():
        for lname, (N, K) in lin.items():
            if lname not in args.ops.split(","):
                continue
            X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
            dY = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
            cases = {
                "fwd": (X, W.t(), torch.empty(M, N, device="cuda", dtype=torch.bfloat16)),
                "dgrad": (dY, W, torch.empty(M, K, device="cuda", dtype=torch.bfloat16)),
                "wgrad": (dY.t(), X, torch.empty(N, K, device="cuda", dtype=odt)),
            }
            for kind in args.kinds.split(","):
                A, B, out = cases[kind]
                flops = 2.0 * M * N * K
                rows = []
                for cfg in cfgs:
                    tag = f"{cfg.tile_m}x{cfg.tile_n}{'pp' if cfg.pingpong else 'co'}{'S' if cfg.swap_ab else ''}"
                    try:
                        t0 = time.time()
                        run_cfg(A, B, out, cfg)
                        torch.cuda.synchronize()
                        comp = time.time() - t0
                        tm = time_us(lambda: run_cfg(A, B, out, cfg), iters=30, repeats=2)
                        rows.append({"cfg": tag, "us": tm["us"], "tflops": flops / tm["us"] / 1e6, "compile_s": comp})
                    except Exception as exc:
                        rows.append({"cfg": tag, "error": f"{type(exc).__name__}: {str(exc)[:100]}"})
                good = sorted([r for r in rows if "us" in r], key=lambda r: r["us"])
                print(f"== {args.model} M={M} {lname} {kind} (N={N}, K={K}) best:",
                      ", ".join(f"{r['cfg']}={r['us']:.1f}us/{r['tflops']:.0f}TF" for r in good[:4]),
                      f"| worst {good[-1]['cfg']}={good[-1]['us']:.1f}" if good else "", flush=True)
                for r in rows:
                    if "error" in r:
                        print("   ERR", r["cfg"], r["error"], flush=True)
                results.append({"model": args.model, "M": M, "linear": lname, "kind": kind, "N": N, "K": K, "rows": rows})
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
