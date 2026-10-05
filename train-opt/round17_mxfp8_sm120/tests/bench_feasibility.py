"""Round17 feasibility: honest per-block MXFP8 time (real quantization, unfused) vs the round13 bf16 block GEMMs.

Part 1  quantizer matrix (real block tensors, ViT-L plain M=7203): own CUDA dual (row+col), own row-only, own col-only,
        QuACK to_mx compiled (row) + to_mx_dim0 compiled (col) + scale packing, r9 Triton row quantizer.
Part 2  12 GEMMs per block with pre-quantized operands in two operand layouts:
        'k'  : every operand K-major (physically transposed copies; what the round13 probe measured), transposes not timed
        'mn' : native layouts (dgrad B N-major, wgrad A M-major / B N-major; QuACK ldmatrix.trans.b8 path), no transposes needed
Part 3  per-block total: sum(12 GEMMs) + quantization of the 8 activation/gradient tensors (dual) + 4 weights (dual) vs bf16 GEMMs.

  python tests/bench_feasibility.py --model vitl16 --m 7203 [--skip-quack-q]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
sys.path.insert(0, str(ROUND.parent / "round13_gemm_fusion_sm120" / "impl"))

import torch

from common.bench import gpu_lock
import mx8
from mb import err_stats, time_us

DIMS = {"vitl16": 1024, "vitb16": 768}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--skip-quack-q", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = DIMS[args.model]
    M = args.m or (7203 if args.model == "vitl16" else 14406)
    Mp = mx8.pad128(M)
    lin = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (4 * d, d), "fc2": (d, 4 * d)}  # (N, K)
    from quack.gemm_interface import gemm

    res = {"model": args.model, "M": M, "Mpad": Mp, "quant": {}, "gemm": {}, "block": {}}
    torch.manual_seed(0)
    with gpu_lock():
        # ------------------------------------------------------------ Part 1: quantizers
        acts = {  # tensor: (rows, cols) of the block's quantized activations / gradients
            "ln1": (M, d), "o": (M, d), "ln2": (M, d), "h": (M, 4 * d), "g2": (M, d), "dz": (M, 4 * d), "g1": (M, d), "dqkv": (M, 3 * d),
        }
        print("== quantizers (us, CUDA-graph x5; bf16 input)")
        qtot_dual = 0.0
        for name, (r, c) in acts.items():
            x = torch.randn(r, c, device="cuda", dtype=torch.bfloat16)
            row = {}
            row["own_dual"] = time_us(lambda: mx8.quantize(x, mpad=Mp))["us"]
            row["own_row"] = time_us(lambda: mx8.quantize(x, col=False, mpad=Mp))["us"]
            row["own_col"] = time_us(lambda: mx8.quantize(x, row=False, mpad=Mp))["us"]
            gb = r * c * 2 / 1e3
            row["dual_GBps"] = (r * c * 2 + 2 * r * c) / (row["own_dual"] * 1e3)
            if not args.skip_quack_q:
                from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig, to_mx_compiled
                from quack.blockscaled.quantize import to_mx_dim0

                def qk_row():
                    q, s = to_mx_compiled(x, 32, "rceil")
                    return q, pack_scale_2d_to_blocked_contig(s.view(r, c // 32))

                try:
                    row["quack_row_compiled"] = time_us(qk_row)["us"]
                except Exception as exc:
                    row["quack_row_compiled"] = f"err {type(exc).__name__}: {str(exc)[:60]}"
            qtot_dual += row["own_dual"]
            res["quant"][name] = row
            print(f"{name:5s} [{r}x{c}] " + " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        # weights: dual from the bf16 shadow
        wtot = 0.0
        for name, (N, K) in lin.items():
            w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
            wt = time_us(lambda: mx8.quantize(w))["us"]
            wtot += wt
            res["quant"]["W_" + name] = {"own_dual": wt}
            print(f"W_{name} [{N}x{K}] own_dual={wt:.1f}", flush=True)
        res["quant_total_us"] = {"acts_dual": qtot_dual, "weights_dual": wtot}
        print(f"quant total per block: activations {qtot_dual:.0f} us + weights {wtot:.0f} us", flush=True)

        # ------------------------------------------------------------ Part 2: GEMMs
        print("== GEMMs: bf16 QuACK vs MXFP8 QuACK (us); layouts k = all K-major copies, mn = native (ldmatrix.trans)")
        gsum = {"bf16": 0.0, "k": 0.0, "mn": 0.0}
        for name, (N, K) in lin.items():
            X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            W = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
            dY = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
            Xq, Wq, dYq = mx8.quantize(X, mpad=Mp), mx8.quantize(W), mx8.quantize(dY, mpad=Mp)
            flops = 2.0 * M * N * K
            refs = {"fwd": X.float() @ W.float().t(), "dgrad": dY.float() @ W.float(), "wgrad": dY.float().t() @ X.float()}
            # K-major physically transposed copies (probe-style)
            WT = mx8.quantize(W.t().contiguous(), col=False)  # [K, N], SF along N
            dYT = mx8.quantize(torch.nn.functional.pad(dY, (0, 0, 0, Mp - M)).t().contiguous(), col=False)  # [N, Mp], SF along Mp
            XT = mx8.quantize(torch.nn.functional.pad(X, (0, 0, 0, Mp - M)).t().contiguous(), col=False)  # [K, Mp]
            bf = {}
            Wb = W
            for kind in ("fwd", "dgrad", "wgrad"):
                if kind == "fwd":
                    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                    bf[kind] = time_us(lambda: gemm(X, Wb.t(), out=out))["us"]
                elif kind == "dgrad":
                    out = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
                    bf[kind] = time_us(lambda: gemm(dY, Wb, out=out))["us"]
                else:
                    out = torch.empty(N, K, device="cuda", dtype=torch.float32)
                    t_torch = time_us(lambda: torch.mm(dY.t(), X, out_dtype=torch.float32))["us"]
                    t_quack = time_us(lambda: gemm(dY.t(), X, out=out))["us"]
                    bf[kind] = min(t_torch, t_quack)
            cases = {
                "fwd": {"k": (Xq.a_row(), Wq.b_row_t()), "mn": (Xq.a_row(), Wq.b_row_t())},
                "dgrad": {"k": (dYq.a_row(), WT.b_row_t()), "mn": (dYq.a_row(), Wq.b_col())},
                "wgrad": {"k": (dYT.a_row(), XT.b_row_t()), "mn": (dYq.a_col_t(), Xq.b_col())},
            }
            for kind, variants in cases.items():
                shp = {"fwd": (M, N), "dgrad": (M, K), "wgrad": (N, K)}[kind]
                row = {"bf16_us": bf[kind]}
                for lay, (A, B) in variants.items():
                    out = torch.empty(shp, device="cuda", dtype=torch.bfloat16)
                    try:
                        gemm(A, B, out=out)
                        torch.cuda.synchronize()
                        row[f"{lay}_us"] = time_us(lambda: gemm(A, B, out=out))["us"]
                        row[f"{lay}_TF"] = flops / row[f"{lay}_us"] / 1e6
                        row[f"{lay}_relL2"] = err_stats(out, refs[kind])["rel_l2"]
                    except Exception as exc:
                        row[f"{lay}_us"] = None
                        row[f"{lay}_err"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                res["gemm"][f"{name}_{kind}"] = row
                for k2 in ("bf16", "k", "mn"):
                    v = row.get(f"{k2}_us")
                    gsum[k2] += v if v is not None else float("nan")
                print(f"{name:5s} {kind:5s} " + " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        res["gemm_sum_us"] = gsum
        print("GEMM sums (us/block, 12 GEMMs):", {k: round(v, 1) for k, v in gsum.items()}, flush=True)
        res["block"] = {"bf16_gemm_us": gsum["bf16"], "mx8_gemm_us_mn": gsum["mn"], "mx8_gemm_us_k": gsum["k"], "quant_us": qtot_dual + wtot,
                        "mx8_total_us_mn": gsum["mn"] + qtot_dual + wtot}
        print("block (GEMM+quant, unfused):", {k: round(v, 1) for k, v in res["block"].items()}, flush=True)
    out = Path(args.out) if args.out else ROUND / "profile" / f"feasibility_{args.model}_M{M}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
