"""MXFP8 readiness probe for round17 (data point only; the BF16 block does not use it).

QuACK 0.6.5 block-scaled GEMM on sm_120 at the fc1 / fc2 / qkv / proj training shapes, three layouts
(fwd X @ W^T, dgrad dY @ W, wgrad dY^T @ X), plus an fused-epilogue-quantized-output call (fc1 output quantized to MXFP8 along N,
the next GEMM's contraction axis).  Reports TFLOPS vs the bf16 QuACK GEMM and the error vs the fp32 product.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

TMP = Path("/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120")
os.environ.setdefault("QUACK_CACHE_DIR", str(TMP / "quack_cache"))
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
sys.path.insert(0, "/root/lightly-train/train-opt")
sys.path.insert(0, "/root/lightly-train/train-opt/round13_gemm_fusion_sm120/impl")

import torch

from common.bench import gpu_lock
from mb import err_stats, time_us


def main():
    from quack.blockscaled.operand import BlockScaledOperand
    from quack.gemm_interface import gemm

    d, M = 1024, 7203
    lin = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (4 * d, d), "fc2": (d, 4 * d)}
    rows = []
    torch.manual_seed(0)
    with gpu_lock():
        for name, (N, K) in lin.items():
            X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
            dY = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
            flops = 2.0 * M * N * K
            pad = (-M) % 128  # wgrad contracts over M = 7203 (not a multiple of 32): zero-pad the contraction rows
            dYp = torch.nn.functional.pad(dY, (0, 0, 0, pad))
            Xp = torch.nn.functional.pad(X, (0, 0, 0, pad))
            cases = {
                # kind: (A, B(K,N view), out shape, reference)
                "fwd": (X, W.t(), (M, N), X.float() @ W.float().t()),
                "dgrad": (dY, W, (M, K), dY.float() @ W.float()),
                "wgrad": (dYp.t(), Xp, (N, K), dY.float().t() @ X.float()),
            }
            for kind, (A, B, shp, ref) in cases.items():
                out_bf = torch.empty(shp, device="cuda", dtype=torch.bfloat16)
                out_q = torch.empty(shp, device="cuda", dtype=torch.bfloat16)
                try:
                    # contraction axis must be the quantized axis of each operand:  A (M,K) along K, B (K,N) along K
                    Aq = BlockScaledOperand.quantize(A.contiguous(), "mxfp8_e4m3", dim=-1)
                    Bq = BlockScaledOperand.quantize(B.mT.contiguous(), "mxfp8_e4m3", dim=-1).mT  # (N,K) quantized -> (K,N) view
                    t0 = time.time()
                    gemm(Aq, Bq, out=out_q)
                    torch.cuda.synchronize()
                    first = time.time() - t0
                    e = err_stats(out_q, ref)
                    tq = time_us(lambda: gemm(Aq, Bq, out=out_q), iters=30, repeats=2)["us"]
                    gemm(A, B, out=out_bf)
                    tb = time_us(lambda: gemm(A, B, out=out_bf), iters=30, repeats=2)["us"]
                    row = {"linear": name, "kind": kind, "bf16_us": tb, "mxfp8_us": tq, "bf16_TF": flops / tb / 1e6, "mxfp8_TF": flops / tq / 1e6,
                           "speedup": tb / tq, "err_vs_fp32": e, "first_call_s": first}
                except Exception as exc:
                    row = {"linear": name, "kind": kind, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                rows.append(row)
                print(row, flush=True)
    (Path(__file__).resolve().parents[1] / "profile" / "mxfp8_probe.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
