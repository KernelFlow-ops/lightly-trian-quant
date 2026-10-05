"""Per-frame GEMM accounting from the measured matrices (no GPU).  Writes profile/gemm_perframe.json and profile/gemm_summary.md.

per-frame time of a (model, kind) = (L-3) x t(M=2058) + 3 x t(M=2258)   (L = 24 / 12 layers; the last 3 layers are joint with M = 2258)
candidates (kernel time as measured; 'epi_missing' = separately measured bandwidth-bound epilogue pass if the kernel has no fused epilogue;
            'quant' = activation quantisation cost: central = assumed fused cost per call, pessimistic = measured stand-alone quantiser):
  r5_v1            current (fused epilogue)
  mg16 (P1)        fp16-accumulate prototype, fused epilogue; operands/outputs fp16 (no quantisation); best tile from tuned16.json (pkt = 1 variant too)
  int8_mm (P2)     torch._int_mm + epi_i32 pass + int8 per-token quantiser
  mg8 (P2)         int8 prototype with fused dequant epilogue + int8 per-token quantiser
  cublaslt_mx8 / quack_mx8 / spark_mx8_v{1,2}   MXFP8 + (epi pass) + MX8 quantiser
  nv4 / mx4 candidates                          FP4 + (epi pass) + FP4 quantiser
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl"))
MODELS = {"vitl16": dict(L=24), "vitb16": dict(L=12)}
KINDS = ("qkv", "proj", "fc1", "fc2")
EXTRA = ("patch", "ct1", "ct2", "einsum")
CALLS_EXTRA = {"patch": 1, "ct1": 1, "ct2": 1, "einsum": 2}
# assumed incremental cost (us per call) of quantisation fused into the producer; the stand-alone measured kernel is the pessimistic bound
FUSED_Q_US = {"qkv": 1.0, "fc1": 1.0, "proj": 2.0, "fc2": 3.0}
SITE_OF = {"qkv": "ln", "fc1": "ln", "proj": "o", "fc2": "gelu"}


def load(p):
    p = HERE / p
    return json.loads(p.read_text()) if p.exists() else {}


def merged(files):
    out = {}
    for f in files:
        for k, v in load(f).items():
            if k.startswith("_"):
                continue
            out.setdefault(k, {}).update({c: r for c, r in v.items() if isinstance(r, dict) and "us" in r})
    return out


def main():
    mat = merged(["results_gemm_lowp1.json", "results_gemm_epi.json", "results_gemm_quackmx8.json", "results_gemm_fp4a.json", "results_gemm_fp4b.json", "results_gemm_i8.json"])
    base = merged(["results_gemm.json"])
    quant = load("results_quant.json")
    t16 = load("impl/tuned16.json")
    t16p = load("impl/tuned16_pkt1.json")
    t8 = load("impl/tuned8.json")
    res = {}
    for model, mi in MODELS.items():
        L = mi["L"]
        for kind in KINDS:
            rows = {}
            n_norm, n_j = L - 3, 3

            def pf(get):
                a, b = get(2058), get(2258)
                return None if a is None or b is None else (n_norm * a + n_j * b) / 1e3     # ms per frame

            def us(cand, M, src=mat):
                v = src.get(f"{model}|{kind}|{M}", {}).get(cand)
                return None if v is None else v["us"]

            ref = lambda M: us("r5_v1", M)                                                    # noqa: E731
            rows["r5_v1"] = pf(ref)
            rows["cublas_bf16(no epi)"] = pf(lambda M: us("cublas_bf16", M))
            rows["cublas_fp16acc(no epi)"] = pf(lambda M: us("cublas_fp16acc", M))
            for tag, tj in (("mg16_tuned", t16), ("mg16_pkt1", t16p)):
                def g(M, tj=tj):
                    e = tj.get(f"{kind}|{ {'qkv': 3, 'proj': 1, 'fc1': 4, 'fc2': 1}[kind] * 0 + (0) }|0|{M}")   # placeholder, replaced below
                    return None
                # key = kind|N|K|M
                from bench10 import kind_nk
                N, K = kind_nk(model, kind)
                rows[tag] = pf(lambda M, tj=tj: (tj.get(f"{kind}|{N}|{K}|{M}") or {}).get("us"))
                # same-run r5_v1 baseline of the tuning run (3 interleaved rounds)
                rows[tag + "_baseline_r5"] = pf(lambda M, tj=tj: ((tj.get(f"{kind}|{N}|{K}|{M}") or {}).get("baseline_us") or {}).get("r5_v1"))
            from bench10 import kind_nk
            N, K = kind_nk(model, kind)
            rows["mg8_tuned(fused epi, no quant)"] = pf(lambda M: (t8.get(f"{kind}|{N}|{K}|{M}") or {}).get("us"))
            rows["mg8_baseline_r5"] = pf(lambda M: ((t8.get(f"{kind}|{N}|{K}|{M}") or {}).get("baseline_us") or {}).get("r5_v1"))
            for c in ("int8_mm", "cublaslt_mx8", "spark_mx8_v1", "spark_mx8_v2", "quack_mx8", "cublaslt_nv4", "cublaslt_mx4", "qutlass_nv4", "qutlass_mx4",
                      "spark_nv4_v1", "spark_nv4_v2", "spark_mx4_v1", "spark_mx4_v2", "quack_nv4", "quack_mx4"):
                rows[c] = pf(lambda M, c=c: us(c, M))
            rows["epi_i32"] = pf(lambda M: us("epi_i32", M))
            rows["epi_bf16"] = pf(lambda M: us("epi_bf16", M))
            res[f"{model}|{kind}"] = rows
    out = {"per_frame_ms": res, "fused_quant_assumption_us_per_call": FUSED_Q_US}
    (HERE / "profile").mkdir(exist_ok=True)
    (HERE / "profile" / "gemm_perframe.json").write_text(json.dumps(out, indent=1))
    names = sorted({c for r in res.values() for c in r})
    for model in MODELS:
        print(f"\n## {model}: ms per frame (sum over layers)")
        print("| candidate | " + " | ".join(KINDS) + " | sum |")
        print("|---|" + "---:|" * (len(KINDS) + 1))
        for c in names:
            vals = [res[f"{model}|{k}"].get(c) for k in KINDS]
            if all(v is None for v in vals):
                continue
            tot = sum(v for v in vals if v is not None) if all(v is not None for v in vals) else None
            print(f"| {c} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + f" | {'-' if tot is None else f'{tot:.3f}'} |")


if __name__ == "__main__":
    main()
