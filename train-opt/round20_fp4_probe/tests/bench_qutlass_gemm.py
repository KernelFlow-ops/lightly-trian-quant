"""R20a: QuTLASS (IST-DASLab, Apache-2.0, e74319e3405c; built from third_party against CUTLASS 4.2 headers) FP4 GEMM vs QuACK / cuBLASLt on every distinct
(M, N, K) of the training step (TN, bf16 out, plain GEMM, operands from fp4_ref).   -> profile/fp4_gemm_qutlass.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/quack_cache")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
sys.path.insert(0, str(ROUND / "tests"))
import torch

import fp4_cands as C
import fp4_ref as R
from bench_fp4_gemm import interleave, pad
from common.bench import gpu_lock

dev = "cuda"
CFG = {"vitl16": dict(d=1024, Mp=(7203, 7903), head=(7168, 28672)), "vitb16": dict(d=768, Mp=(14406, 15806), head=(14336, 57344))}


def shapes():
    out = {}
    for model, c in CFG.items():
        d = c["d"]
        lin = {"qkv": (3 * d, d), "proj": (d, d), "fc1": (4 * d, d), "fc2": (d, 4 * d)}
        for M in c["Mp"]:
            for n, (N, K) in lin.items():
                out[(M, N, K)] = f"{model} bb{M} {n} fwd"
                out[(M, K, N)] = f"{model} bb{M} {n} dgrad"
                out[(N, K, pad(M, 128))] = f"{model} bb{M} {n} wgrad"
        for M in c["head"]:
            out[(M, 4 * d, d)] = f"{model} head{M} fwd"
            out[(M, d, 4 * d)] = f"{model} head{M} dgrad"
            out[(4 * d, d, M)] = f"{model} head{M} wgrad"
    return out


def main():
    res = {"cases": {}, "notes": []}
    with gpu_lock():
        torch.ops.load_library("/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/build/qutlass/_qutlass_C.so")
        qut = torch.ops._qutlass_C
        g = torch.Generator(device=dev).manual_seed(0)
        checked = False
        for (M, N, K), label in sorted(shapes().items()):
            A = torch.randn(M, K, device=dev, generator=g).bfloat16()
            B = (torch.randn(N, K, device=dev, generator=g) * 0.05).bfloat16()
            ops = {f: (R.quantize_matrix(A, f), R.quantize_matrix(B, f)) for f in ("nv4", "mx4")}
            out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            cands = {}
            for fmt, (a, b) in ops.items():
                ao, bo = C.operand(a, fmt), C.operand(b, fmt)
                cands[f"{fmt}_quack"] = (lambda ao=ao, bo=bo: C.quack_fp4(ao, bo, out))
                sd = torch.float8_e4m3fn if fmt == "nv4" else torch.float8_e8m0fnu
                vec = R.VEC[fmt]
                alpha = (a[2] * b[2]).reshape(1).contiguous() if fmt == "nv4" else torch.ones(1, device=dev)
                fn = qut.matmul_nvf4_bf16_tn if fmt == "nv4" else qut.matmul_mxf4_bf16_tn
                cands[f"{fmt}_qutlass"] = (lambda a=a, b=b, sd=sd, vec=vec, alpha=alpha, fn=fn:
                                          fn(a[0].view(torch.uint8), b[0].view(torch.uint8), a[1].view(sd).reshape(-1, K // vec), b[1].view(sd).reshape(-1, K // vec), alpha))
            cands["nv4_cublaslt"] = lambda a=ops["nv4"][0], b=ops["nv4"][1]: C.scaled_mm_fp4(a, b, "nv4", torch.bfloat16)
            if not checked:  # correctness of the QuTLASS call convention once
                try:
                    fmt = "nv4"
                    a, b = ops[fmt]
                    ref = (R.dequant_copy(a[0], a[1], a[2], fmt, M, K).double() @ R.dequant_copy(b[0], b[1], b[2], fmt, N, K).double().t()).float()
                    o = cands["nv4_qutlass"]()
                    res["correctness_nv4_qutlass_relL2"] = float((o.float() - ref).norm() / ref.norm())
                    fmt = "mx4"
                    a, b = ops[fmt]
                    ref = (R.dequant_copy(a[0], a[1], a[2], fmt, M, K).double() @ R.dequant_copy(b[0], b[1], b[2], fmt, N, K).double().t()).float()
                    o = cands["mx4_qutlass"]()
                    res["correctness_mx4_qutlass_relL2"] = float((o.float() - ref).norm() / ref.norm())
                    checked = True
                except Exception as e:  # noqa: BLE001
                    res["notes"].append(f"correctness check failed: {type(e).__name__}: {str(e)[:300]}")
                    checked = True
            us, errs = interleave(cands, inner=5, reps=10)
            fl = 2.0 * M * N * K
            res["cases"][f"{M}x{N}x{K}"] = {"label": label, "us": us, "tflops": {k: fl / v / 1e6 for k, v in us.items()}, "errors": errs}
            print(f"{label} M={M} N={N} K={K}: " + ", ".join(f"{k}={v:.0f}" for k, v in sorted(us.items(), key=lambda kv: kv[1])) + (f" ERR {list(errs)}" if errs else ""), flush=True)
            if errs:
                print("   ", {k: v[:120] for k, v in errs.items()}, flush=True)
            (ROUND / "profile" / "fp4_gemm_qutlass.json").write_text(json.dumps(res, indent=1))
            del A, B, ops
            torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
