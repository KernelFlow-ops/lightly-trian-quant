"""Per-call / per-frame GEMM time model built ONLY from the measured matrices (results_gemm*.json, impl/tuned*.json, results_quant.json).

tier names (kernel + whatever is still missing to be a drop-in for the r5_v1 block; quantisation of the activation is added separately):
  r5_v1          measured r5_v1 kernel with its fused epilogue (same-run baseline of each file)
  fp16           mg_f16acc prototype, fused epilogue, tuned tile (impl/tuned16.json);  fp16_pkt1 = restricted to promotion every k-tile
  int8           mg_i8 prototype, fused dequant + epilogue (impl/tuned8.json)
  int8_cublas    torch._int_mm + stand-alone dequant/epilogue pass (epi_i32)
  mx8            min( cuBLASLt MXFP8 + epi pass , QuACK mx8 (+ epi pass for the qkv RoPE/layout) , Spark fp8 MX + epi pass )
  nv4 / mx4      min over the FP4 candidates (+ epi pass)
quantisation cost per call (activation): 'central' = fused into the producer (assumed: LN-fed 1 us, attention-fed 2 us, GELU-fed 3 us; per-token int8 needs the full row -> stand-alone pass for proj / fc2 inputs),
                                         'pess' = measured stand-alone quantiser kernel.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(HERE / "impl"))
from bench10 import kind_nk  # noqa: E402

KINDS = ("qkv", "proj", "fc1", "fc2")
LAYERS = {"vitl16": 24, "vitb16": 12}
NJOINT = 3
FUSED_Q_US = {"qkv": 1.0, "fc1": 1.0, "proj": 2.0, "fc2": 3.0}
SITE = {"qkv": "ln", "fc1": "ln", "proj": "o", "fc2": "gelu"}


def _j(p):
    p = HERE / p
    return json.loads(p.read_text()) if p.exists() else {}


class Speed:
    def __init__(self):
        self.mat = {}
        for f in ("results_gemm_lowp1.json", "results_gemm_epi.json", "results_gemm_quackmx8.json", "results_gemm_fp4a.json", "results_gemm_fp4b.json", "results_gemm_i8.json"):
            for k, v in _j(f).items():
                if k.startswith("_"):
                    continue
                self.mat.setdefault(k, {}).update({c: r for c, r in v.items() if isinstance(r, dict) and "us" in r})
        self.t16, self.t16p, self.t8 = _j("impl/tuned16.json"), _j("impl/tuned16_pkt1.json"), _j("impl/tuned8.json")
        self.quant = _j("results_quant.json")
        self._base_matrix = _j("results_gemm.json")

    def _m(self, model, kind, M, cand):
        v = self.mat.get(f"{model}|{kind}|{M}", {}).get(cand)
        return None if v is None else v["us"]

    def _tuned(self, tj, model, kind, M, field="us"):
        N, K = kind_nk(model, kind)
        e = tj.get(f"{kind}|{N}|{K}|{M}")
        if not e:
            return None
        return e["us"] if field == "us" else (e.get("baseline_us") or {}).get(field)

    def base(self, model, kind, M):
        """r5_v1 baseline = the fastest (most conservative for the savings) r5_v1 time seen in any run: lowp1 matrix, base matrix, and the same-run baselines of the tile sweeps."""
        c = [self._m(model, kind, M, "r5_v1")]
        c.append(self._base_matrix.get(f"{model}|{kind}|{M}", {}).get("r5_v1", {}).get("us"))
        for tj in (self.t16, self.t16p, self.t8):
            c.append(self._tuned(tj, model, kind, M, "r5_v1"))
        c = [x for x in c if x]
        return min(c) if c else None

    def kernel(self, model, kind, tier, M):
        """us per call of the GEMM incl. any separately needed epilogue pass (no activation quantisation)."""
        if tier == "r5_v1":
            return self.base(model, kind, M)
        if tier == "fp16":
            return self._tuned(self.t16, model, kind, M)
        if tier == "fp16_pkt1":
            return self._tuned(self.t16p, model, kind, M)
        if tier == "int8":
            return self._tuned(self.t8, model, kind, M)
        epi = self._m(model, kind, M, "epi_bf16")
        epi32 = self._m(model, kind, M, "epi_i32")
        if tier == "fp16_cublas":      # cuBLAS fp16 with fp16 accumulation (no epilogue) + separate epilogue pass; the qkv epilogue (RoPE + layout) is the expensive one
            g = self._m(model, kind, M, "cublas_fp16acc")
            return None if g is None or epi is None else g + epi
        if tier == "int8_cublas":
            g = self._m(model, kind, M, "int8_mm")
            return None if g is None or epi32 is None else g + epi32
        if tier in ("mx8", "nv4", "mx4"):
            cands = {"mx8": [("cublaslt_mx8", True), ("spark_mx8_v1", True), ("spark_mx8_v2", True), ("quack_mx8", kind == "qkv")],
                     "nv4": [("cublaslt_nv4", True), ("qutlass_nv4", True), ("spark_nv4_v1", True), ("spark_nv4_v2", True), ("quack_nv4", kind == "qkv")],
                     "mx4": [("cublaslt_mx4", True), ("qutlass_mx4", True), ("spark_mx4_v1", True), ("spark_mx4_v2", True), ("quack_mx4", kind == "qkv")]}[tier]
            best = None
            for c, need_epi in cands:
                g = self._m(model, kind, M, c)
                if g is None:
                    continue
                if need_epi:
                    if epi is None:
                        continue
                    g += epi
                best = g if best is None else min(best, g)
            return best
        raise ValueError(tier)

    def quant_us(self, model, kind, fmt, M, mode="central"):
        """activation quantisation cost per call."""
        site = SITE[kind]
        q = self.quant.get(f"{model}|{site}|{M}", {})

        def g(*names):
            v = [q[n]["us"] for n in names if n in q and "us" in q[n]]
            return min(v) if v else None
        if fmt in ("fp16", "bf16"):
            return 0.0
        if fmt in ("int8", "i8"):
            pess = g("int8_rows_r10")
            if mode == "central" and kind in ("qkv", "fc1"):
                return FUSED_Q_US[kind]
            return pess
        pess = {"mx8": g("quant_dual_r17", "spark_fp8mx"), "nv4": g("fp4q2_nv4_exact", "spark_nv4"), "mx4": g("fp4q2_mx4", "spark_mx4")}[fmt]
        return FUSED_Q_US[kind] if mode == "central" else pess

    def pfx_us(self, model, kind, M):
        """extra cost of the bf16 side GEMM on the prefix rows (measured cuBLAS small-M GEMM + 2 us allowance for the row scatter)."""
        v = (_j("results_pfx.json").get(f"{model}|{kind}|{M}") or {}).get("pfx_bf16", {}).get("us")
        return None if v is None else v + 2.0

    def per_layer_ms(self, model, kind, tier, qfmt, layer, mode="central", pfx=False):
        """(baseline, candidate) ms of this layer's GEMM incl. quantisation."""
        L = LAYERS[model]
        M = 2258 if layer >= L - NJOINT else 2058
        b = self.base(model, kind, M)
        k = self.kernel(model, kind, tier, M)
        q = self.quant_us(model, kind, qfmt, M, mode)
        if b is None or k is None or q is None:
            return None
        if pfx and tier != "r5_v1":
            p = self.pfx_us(model, kind, M)
            if p is None:
                return None
            k += p
        return b / 1e3, (k + q) / 1e3

    def saving_ms(self, model, kind, tier, qfmt, layers, mode="central", pfx=False):
        s = 0.0
        for l in layers:
            r = self.per_layer_ms(model, kind, tier, qfmt, l, mode, pfx)
            if r is None:
                return None
            s += r[0] - r[1]
        return s

    def frame_ms(self, model, kind, tier, qfmt, mode="central", layers=None):
        L = LAYERS[model]
        layers = range(L) if layers is None else layers
        tot = 0.0
        for l in layers:
            r = self.per_layer_ms(model, kind, tier, qfmt, l, mode)
            if r is None:
                return None
            tot += r[1]
        return tot
