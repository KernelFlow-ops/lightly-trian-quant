"""R21a: per-step FP4 quantization cost from the measured micro-benchmark (profile/bench_fp4q.json) for the R20a strategies
  (1) head dgrad+wgrad -> nv4        (3) backbone dgrad+wgrad -> nv4 (prefix tokens excluded from the FP4 wgrad, RHT on the token axis, SR on dY)
Counting follows round20 tests/summarize_r20a.py: ViT-L 21 plain (M=7203) + 3 joint (M=7903) blocks, ViT-B 9 + 3 (14406 / 15806); head: 4 predict calls per step x (ConvT1, ConvT2).
Costs are the COLD (DRAM-resident) micro-benchmark numbers; the real step has partial L2 hits (producer -> quantizer), so these are upper bounds for small tensors.
Delta = FP4 quantizers - the r17 MXFP8 quantizers they replace in FINAL (same shapes, same benchmark session):
  head:     FINAL has no quantizer (bf16 head)  -> Delta = full FP4 cost
  backbone: dY: nv4 dual(RHT col, SR, prefix) - mx8 dual ;  X: nv4 col(RHT, prefix) + mx8 row(kept for the MXFP8 forward) - mx8 dual   (qkv X: forward is bf16 in FINAL: + nv4 col only)
Weights (nv4 delayed amax, both copies, once per step per Linear) are listed separately (FINAL already quantizes the weights to MXFP8 -> counted as equal, like round20).
Writes profile/per_step_quant_cost.json and merges into results.json.
"""
import json
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
B = json.loads((ROUND / "profile" / "bench_fp4q.json").read_text())
INV = {"vitl16": dict(nplain=21, njoint=3, Mp=7203, Mj=7903, hM=(7168, 28672), d=1024), "vitb16": dict(nplain=9, njoint=3, Mp=14406, Mj=15806, hM=(14336, 57344), d=768)}
NPRED = 4


def get(model, name, mode):
    return B["act"][model][name]["cold_us"][mode]["us"]


def tag(R, C, X):
    return f"{'X' if X == 'X' else ('h' if X == 'h' else 'dY')}[{R},{C}]"


out = {"basis": "cold micro-benchmark medians (us); see tests/per_step_quant.py docstring", "models": {}}
for model, inv in INV.items():
    d = inv["d"]
    m = {"head": {}, "backbone": {}}
    # ---- (1) head: per predict, ConvT1 / ConvT2: dZ [M,4d] dual (SR + RHT col), X [M,d] col (RHT, RN)
    tot = tot_rn = 0.0
    det = []
    for M in inv["hM"]:
        dz = get(model, f"dZ[{M},{4 * d}]", "nv4_dual_rhtcol_sr")
        xc = get(model, f"convT_in[{M},{d}]", "nv4_col")
        det.append({"M": M, "dZ_dual_rhtcol_sr_us": dz, "X_col_us": xc})
        tot += NPRED * (dz + xc)
        tot_rn += NPRED * (get(model, f"dZ[{M},{4 * d}]", "nv4_dual") + xc)
    m["head"] = {"per_step_ms": round(tot / 1e3, 3), "per_step_ms_if_RN_noSR": round(tot_rn / 1e3, 3), "items": det, "delta_vs_FINAL_ms": round(tot / 1e3, 3)}
    # ---- (3) backbone
    acc = {"fp4_total": 0.0, "mx8_replaced": 0.0, "dY_fp4": 0.0, "X_fp4": 0.0, "dY_mx8": 0.0, "X_mx8_dual": 0.0, "X_mx8_row": 0.0}
    for cls, M, nblk in (("plain", inv["Mp"], inv["nplain"]), ("joint", inv["Mj"], inv["njoint"])):
        for lin, (xr, xc_, dyc) in {"qkv": (d, "X", 3 * d), "proj": (d, "X", d), "fc1": (d, "X", 4 * d), "fc2": (4 * d, "h", d)}.items():
            xn = f"{'X' if xc_ == 'X' else 'h'}[{M},{xr}]"
            dn = f"dY[{M},{dyc}]"
            dy_fp4 = get(model, dn, "nv4_full_prefix_rhtcol_sr")
            x_col = get(model, xn, "nv4_col_prefix_rht")
            dy_m8 = get(model, dn, "r17_mx8_dual")
            x_m8d = get(model, xn, "r17_mx8_dual")
            x_m8r = get(model, xn, "r17_mx8_row")
            acc["dY_fp4"] += nblk * dy_fp4
            acc["X_fp4"] += nblk * x_col
            acc["dY_mx8"] += nblk * dy_m8
            if lin == "qkv":  # bf16 forward in FINAL: no mx8 quantizer for X; the wgrad copy is new
                acc["X_mx8_dual"] += 0.0
            else:
                acc["X_mx8_dual"] += nblk * x_m8d
                acc["X_mx8_row"] += nblk * x_m8r
    fp4_cost = acc["dY_fp4"] + acc["X_fp4"]
    delta = (acc["dY_fp4"] - acc["dY_mx8"]) + (acc["X_fp4"] + acc["X_mx8_row"] - acc["X_mx8_dual"])
    w = [get_w for get_w in []]
    wcost = 0.0
    for (N, K), cnt in (((3 * d, d), 1), ((d, d), 1), ((4 * d, d), 1), ((d, 4 * d), 1)):
        key = f"w[{N},{K}]"
        wcost += B["weight"][model][key]["cold"]["nv4_w_delayed"]["us"]
    wblocks = inv["nplain"] + inv["njoint"]
    m["backbone"] = {"fp4_quant_per_step_ms": round(fp4_cost / 1e3, 3), "mx8_dY_dual_replaced_ms": round(acc["dY_mx8"] / 1e3, 3),
                     "delta_vs_FINAL_ms": round(delta / 1e3, 3),
                     "weights_nv4_delayed_per_step_ms (24 blocks x 4 Linear, both copies)": round(wblocks * wcost / 1e3, 3),
                     "breakdown_ms": {k: round(v / 1e3, 3) for k, v in acc.items()}}
    out["models"][model] = m
(ROUND / "profile" / "per_step_quant_cost.json").write_text(json.dumps(out, indent=1))
for k, v in out["models"].items():
    print(k, "head(1) ms/step:", v["head"]["per_step_ms"], "(RN:", v["head"]["per_step_ms_if_RN_noSR"], ")  backbone(3): FP4 quant", v["backbone"]["fp4_quant_per_step_ms"], "ms, delta vs FINAL mx8", v["backbone"]["delta_vs_FINAL_ms"], "ms, weights", v["backbone"]["weights_nv4_delayed_per_step_ms (24 blocks x 4 Linear, both copies)"])
