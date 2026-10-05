"""R20a: merge the measured matrices, weight them by the per-step GEMM inventory, apply the Amendment-6 11.6 speed conditions.
Reads profile/fp4_gemm_matrix_*.json, fp4_gemm_qutlass.json, fp4_quant_cost.json; writes profile/fp4_gemm_matrix.json and profile/per_step_estimate.json.
Only measured numbers go in; every derived quantity is labelled (derived / estimate) and its inputs are listed.
"""
import json
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
P = ROUND / "profile"
STEP_MS = {"vitl16": 101.8, "vitb16": 86.6}  # FINAL mid-phase step (PLAN_v3 section 0, same machine)
KERNEL_MS = {"vitl16": {"head_gemm": 19.0, "bb_mx8_gemm": 24.4, "bb_bf16_qkv": 4.8, "mx8_quant": 5.1}, "vitb16": {"head_gemm": 21.6, "bb_mx8_gemm": 14.5, "bb_bf16_qkv": 2.7, "mx8_quant": 4.1}}
INV = {"vitl16": dict(nplain=21, njoint=3, Mp=7203, Mj=7903, hM=(7168, 28672), d=1024), "vitb16": dict(nplain=9, njoint=3, Mp=14406, Mj=15806, hM=(14336, 57344), d=768)}
LINS = ("qkv", "proj", "fc1", "fc2")


def load(model):
    cases = {}
    for what in ("bb%d" % INV[model]["Mp"], "bb%d" % INV[model]["Mj"], "head"):
        f = P / f"fp4_gemm_matrix_{model}_{what}.json"
        if f.exists():
            cases[what] = json.load(open(f))["cases"]
    return cases


def pick(us, *names):
    v = [us[n] for n in names if n in us]
    return min(v) if v else None


def case_summary(us, name, kind, head):
    """-> dict(cur, bf16, mx8, nv4_epi/usable, nv4_plain, mx4_usable, mx4_plain, best_plain_backend)"""
    r = {}
    if kind == "fwd":
        r["cur"] = pick(us, "cur_FINAL(bf16_epi)", "cur_FINAL(mx8_epi)", "cur_FINAL(bf16)")
        r["bf16"] = pick(us, "bf16_epi", "cur_FINAL(bf16_epi)", "cur_FINAL(bf16)")
        r["mx8"] = pick(us, "mx8_epi", "mx8", "cur_FINAL(mx8_epi)")
        epi_nv4 = pick(us, "nv4_quack_epi")
        plain_nv4 = pick(us, "nv4_quack_plain")
        r["nv4_plain_quack"] = plain_nv4
        r["nv4"] = plain_nv4 if head else epi_nv4
        r["mx4_plain_quack"] = pick(us, "mx4_quack_plain")
        mx4_epi = pick(us, "mx4_quack_epi")
        if head:
            r["mx4"] = r["mx4_plain_quack"]
        elif mx4_epi is not None:
            r["mx4"] = mx4_epi
        else:  # derived: mx4 plain + the nv4 epilogue overhead
            r["mx4"] = r["mx4_plain_quack"] + (epi_nv4 - plain_nv4)
            r["mx4_derived"] = True
    elif kind == "dgrad":
        r["cur"] = pick(us, "cur_FINAL(mx8)", "cur_FINAL(mx8_dgelu)", "cur_FINAL(bf16)")
        r["bf16"] = pick(us, "bf16", "bf16_dgelu", "cur_FINAL(bf16)")
        r["mx8"] = pick(us, "mx8", "cur_FINAL(mx8)", "cur_FINAL(mx8_dgelu)")
        epi_nv4 = pick(us, "nv4_quack_epi(dgelu)")
        plain_nv4 = pick(us, "nv4_quack_plain")
        r["nv4_plain_quack"] = plain_nv4
        r["nv4"] = epi_nv4 if epi_nv4 is not None else plain_nv4
        r["mx4_plain_quack"] = pick(us, "mx4_quack_plain")
        mx4_epi = pick(us, "mx4_quack_epi(dgelu)")
        r["mx4"] = mx4_epi if mx4_epi is not None else r["mx4_plain_quack"]
    else:  # wgrad, fp32 out
        mx8 = pick(us, "cur_FINAL(mx8_native)")
        r["mx8"] = mx8
        r["cur"] = mx8 if not head else pick(us, "bf16_torch_f32", "bf16_quack_f32", "bf16_quack_f32_splitk")  # head wgrad is bf16 in FINAL
        r["bf16"] = pick(us, "bf16_torch_f32", "bf16_quack_f32", "bf16_quack_f32_splitk")
        r["nv4"] = pick(us, "nv4_quack_f32", "nv4_quack_f32_splitk", "nv4_cublaslt_f32")
        r["nv4_plain_quack"] = pick(us, "nv4_quack_f32", "nv4_quack_f32_splitk")
        r["mx4"] = pick(us, "mx4_quack_f32", "mx4_quack_f32_splitk")
        r["mx4_plain_quack"] = r["mx4"]
        r["prefix5_side"] = pick(us, "prefix5_side_gather+mm_f32+add")
        r["prefix105_side"] = pick(us, "prefix105_side_gather+mm_f32+add")
    # plain-GEMM backend comparison (bf16 out; wgrad uses the bf16-out variants of cuBLASLt / Spark for a like-for-like plain comparison)
    if kind == "wgrad":
        quack = pick(us, "nv4_quack_f32", "nv4_quack_f32_splitk")
        cub = pick(us, "nv4_cublaslt_bf16")
        spark = pick(us, *[k for k in us if k.startswith("nv4_spark")])
    else:
        quack, cub, spark = pick(us, "nv4_quack_plain"), pick(us, "nv4_cublaslt_plain"), pick(us, "nv4_spark", "nv4_spark_padM")
    r["plain_nv4_backends"] = {"quack": quack, "cublaslt": cub, "spark": spark}
    return r


def main():
    out = {"models": {}, "qutlass": None}
    quant = json.load(open(P / "fp4_quant_cost.json")) if (P / "fp4_quant_cost.json").exists() else None
    for model in INV:
        inv = INV[model]
        cases = load(model)
        per = {}  # (cls, lin, kind) -> summary
        for what, cs in cases.items():
            for key, rec in cs.items():
                if "us" not in rec:
                    continue
                lin, kind = key.split("/")
                head = what == "head"
                if head:
                    stage = int(lin[-1])  # convT1 / convT2
                    per[("head", stage, kind)] = {**case_summary(rec["us"], lin, kind, True), "M": rec["M"], "N": rec["N"], "K": rec["K"], "flops": rec["flops"]}
                else:
                    cls = "plain" if rec["M"] == inv["Mp"] else "joint"
                    per[(cls, lin, kind)] = {**case_summary(rec["us"], lin, kind, False), "M": rec["M"], "N": rec["N"], "K": rec["K"], "flops": rec["flops"]}
        out["models"][model] = {"cases": {"/".join(map(str, k)): v for k, v in per.items()}}
        # ------------------------------------------------------------ per-step weighted sums (us)
        def bb_sum(kinds, lins, field):
            tot = 0.0
            for cls, n in (("plain", inv["nplain"]), ("joint", inv["njoint"])):
                for lin in lins:
                    for kind in kinds:
                        c = per.get((cls, lin, kind))
                        if c is None or c.get(field) is None:
                            return None
                        tot += n * c[field]
            return tot

        def head_sum(kinds, field):
            tot = 0.0
            for stage in (1, 2):
                for kind in kinds:
                    c = per.get(("head", stage, kind))
                    if c is None or c.get(field) is None:
                        return None
                    tot += 4 * c[field]
            return tot

        S = {}
        for field in ("cur", "bf16", "mx8", "nv4", "mx4"):
            for name, kinds, lins in (("bb_d", ("dgrad",), LINS), ("bb_w", ("wgrad",), LINS), ("bb_f_qkv", ("fwd",), ("qkv",)), ("bb_f_proj", ("fwd",), ("proj",)), ("bb_f_fc", ("fwd",), ("fc1", "fc2"))):
                S[f"{name}.{field}"] = bb_sum(kinds, lins, field)
            for name, kinds in (("head_f", ("fwd",)), ("head_d", ("dgrad",)), ("head_w", ("wgrad",))):
                S[f"{name}.{field}"] = head_sum(kinds, field)
        # prefix-token side GEMM (gather + bf16 GEMM fp32 out + add into the FP4 wgrad result), summed over all 4 linears of every block
        pf = {}
        for tag, plain_f, joint_f in (("prefix5_all_blocks", "prefix5_side", "prefix5_side"), ("prefix5_plain+prefix105_joint", "prefix5_side", "prefix105_side")):
            tot = 0.0
            for cls, n, f in (("plain", inv["nplain"], plain_f), ("joint", inv["njoint"], joint_f)):
                for lin in LINS:
                    c = per.get((cls, lin, "wgrad"))
                    if c is None or c.get(f) is None:
                        tot = None
                        break
                    tot += n * c[f]
                if tot is None:
                    break
            pf[tag + "_ms"] = None if tot is None else tot / 1e3
        out["models"][model]["prefix_side_gemm_ms"] = pf
        out["models"][model]["per_step_gemm_us"] = S
        # ------------------------------------------------------------ quantizer estimate
        Q = {}
        KAPPA = 1.25  # safety factor over the byte-proportional cost of the measured r17 quantizer
        if quant is not None and model in quant["models"]:
            qm = quant["models"][model]
            d = inv["d"]

            def qm_shape(rows, cols):
                for k, v in qm.items():
                    if v["rows"] == rows and v["cols"] == cols:
                        return k
                return None

            def meas(rows, cols, cand):
                k = qm_shape(rows, cols)
                if k is not None and qm[k]["us"].get(cand) is not None:
                    return qm[k]["us"][cand], False
                for k2, v in qm.items():  # same width, scaled by element count (flagged)
                    if v["cols"] == cols and v["us"].get(cand) is not None and not k2.startswith("head"):
                        return v["us"][cand] * rows / v["rows"], True
                return None, True

            # bytes moved per element (bf16 in = 2): mx8 dual = 2 + 2 + 2/32 ; nv4 dual = 2 + 1 + 2/16 (+2 for a separate amax pass) ; nv4 col = 2 + .5 + 1/16 (+2) ; mx4 dual = 2 + 1 + 2/32 ; mx4 col = 2 + .5 + 1/32
            B8 = 2 + 2 + 2 / 32
            BYTES = {("nv4", "dual", False): 2 + 1 + 2 / 16, ("nv4", "dual", True): 2 + 1 + 2 / 16 + 2, ("nv4", "col", False): 2 + .5 + 1 / 16, ("nv4", "col", True): 2 + .5 + 1 / 16 + 2,
                     ("mx4", "dual", False): 2 + 1 + 2 / 32, ("mx4", "dual", True): 2 + 1 + 2 / 32, ("mx4", "col", False): 2 + .5 + 1 / 32, ("mx4", "col", True): 2 + .5 + 1 / 32}
            SCEN = {"existing_kernels_unfused(nv4)": ("nv4", None), "fused_nv4_amax_from_producer": ("nv4", False), "fused_nv4_separate_amax_pass": ("nv4", True), "fused_mx4": ("mx4", False)}
            tensors = {}
            for cls, M in (("plain", inv["Mp"]), ("joint", inv["Mj"])):
                tensors[cls] = {"X": {"qkv": (M, d), "proj": (M, d), "fc1": (M, d), "fc2": (M, 4 * d)}, "dY": {"qkv": (M, 3 * d), "proj": (M, d), "fc1": (M, 4 * d), "fc2": (M, d)}}
            Q["tensors"] = {c: {k: {l: list(sh) for l, sh in v.items()} for k, v in tt.items()} for c, tt in tensors.items()}
            Q["kappa"] = KAPPA
            Q["scenarios"] = {}
            for scen, (fmt, amax_pass) in SCEN.items():
                acc = {"item3_dY_ms": 0.0, "item3_X_ms": 0.0, "item4_ms": 0.0, "item5a_ms": 0.0, "item1_ms": 0.0, "item2a_ms": 0.0}
                scaled = False
                ok = True

                def cost(rows, cols, what):  # us of an FP4 copy-set (what = dual | col)
                    global_scaled = []
                    if fmt == "nv4" and amax_pass is None:  # existing kernels: Spark row (device amax, 2 launches) + Spark col (transpose copy + quant)
                        row, e1 = meas(rows, cols, "spark_nv4_row(device amax, 2 launches)")
                        col, e2 = meas(rows, cols, "spark_nv4_col(transpose copy + quant, ts given)")
                        if row is None or col is None:
                            return None, e1 or e2
                        return (col if what == "col" else row + col), e1 or e2
                    m8d, e = meas(rows, cols, "mx8_quant_dual(row+col)")
                    if m8d is None:
                        return None, e
                    return m8d * BYTES[(fmt, what, bool(amax_pass))] / B8 * KAPPA, e

                for cls, nblk in (("plain", inv["nplain"]), ("joint", inv["njoint"])):
                    for lin in LINS:
                        for role in ("X", "dY"):
                            rows, cols = tensors[cls][role][lin]
                            m8d, e1 = meas(rows, cols, "mx8_quant_dual(row+col)")
                            m8r, e2 = meas(rows, cols, "mx8_quant_row")
                            (dual, e3), (col, e4) = cost(rows, cols, "dual"), cost(rows, cols, "col")
                            scaled = scaled or e1 or e2 or e3 or e4
                            if None in (m8d, m8r, dual, col):
                                ok = False
                                continue
                            if role == "dY":
                                acc["item3_dY_ms"] += nblk * (dual - m8d) / 1e3
                            else:
                                acc["item3_X_ms"] += nblk * ((m8r + col - m8d) if lin != "qkv" else col) / 1e3
                                if lin in ("fc1", "fc2"):
                                    acc["item4_ms"] += nblk * (dual - (m8r + col)) / 1e3
                                if lin == "qkv":
                                    acc["item5a_ms"] += nblk * m8r / 1e3
                for stage, M in enumerate(inv["hM"], 1):
                    (dz, e1), (xc, e2) = cost(M, 4 * d, "dual"), cost(M, d, "col")
                    r8, e3 = meas(M, d, "mx8_quant_row")
                    scaled = scaled or e1 or e2 or e3
                    if None in (dz, xc, r8):
                        ok = False
                        continue
                    acc["item1_ms"] += 4 * (dz + xc) / 1e3
                    acc["item2a_ms"] += 4 * r8 / 1e3
                acc["any_scaled_by_elements"] = scaled
                Q["scenarios"][scen] = acc if ok else None
        out["models"][model]["quant_us"] = Q

        # ------------------------------------------------------------ items and conditions
        def d(a, b):
            return None if a is None or b is None else a - b

        def ratio(a, b):
            return None if a is None or b is None or b == 0 else a / b

        items = {}
        items["1_head_dw_nv4"] = {"gemm_now_us": (S["head_d.cur"] or 0) + (S["head_w.cur"] or 0), "gemm_fp4_us": (S["head_d.nv4"] or 0) + (S["head_w.nv4"] or 0)}
        items["1b_head_dw_mx8_fallback"] = {"gemm_now_us": items["1_head_dw_nv4"]["gemm_now_us"], "gemm_fp4_us": (S["head_d.mx8"] or 0) + (S["head_w.mx8"] or 0)}
        items["2a_head_f_mx8"] = {"gemm_now_us": S["head_f.cur"], "gemm_fp4_us": S["head_f.mx8"]}
        items["2b_head_f_nv4"] = {"gemm_now_us": S["head_f.cur"], "gemm_fp4_us": S["head_f.nv4"]}
        items["3_bb_dw_nv4"] = {"gemm_now_us": (S["bb_d.cur"] or 0) + (S["bb_w.cur"] or 0), "gemm_fp4_us": (S["bb_d.nv4"] or 0) + (S["bb_w.nv4"] or 0)}
        items["3_bb_dw_mx4"] = {"gemm_now_us": items["3_bb_dw_nv4"]["gemm_now_us"], "gemm_fp4_us": (S["bb_d.mx4"] or 0) + (S["bb_w.mx4"] or 0)}
        items["1_head_dw_mx4"] = {"gemm_now_us": items["1_head_dw_nv4"]["gemm_now_us"], "gemm_fp4_us": (S["head_d.mx4"] or 0) + (S["head_w.mx4"] or 0)}
        items["4_fc_f_nv4"] = {"gemm_now_us": S["bb_f_fc.cur"], "gemm_fp4_us": S["bb_f_fc.nv4"]}
        items["4_fc_f_mx4"] = {"gemm_now_us": S["bb_f_fc.cur"], "gemm_fp4_us": S["bb_f_fc.mx4"]}
        items["5a_qkv_f_mx8"] = {"gemm_now_us": S["bb_f_qkv.cur"], "gemm_fp4_us": S["bb_f_qkv.mx8"]}
        items["5b_qkv_f_nv4"] = {"gemm_now_us": S["bb_f_qkv.cur"], "gemm_fp4_us": S["bb_f_qkv.nv4"]}
        items["x_proj_f_nv4(not proposed: bandwidth bound)"] = {"gemm_now_us": S["bb_f_proj.cur"], "gemm_fp4_us": S["bb_f_proj.nv4"]}
        for k, v in items.items():
            v["dgemm_ms"] = None if v["gemm_now_us"] is None or v["gemm_fp4_us"] is None else (v["gemm_now_us"] - v["gemm_fp4_us"]) / 1e3
            v["speedup_x"] = ratio(v["gemm_now_us"], v["gemm_fp4_us"])
        out["models"][model]["items"] = items
        qs = Q.get("scenarios", {})
        summ = {}
        for scen, acc in qs.items():
            if acc is None:
                summ[scen] = None
                continue
            fm = "mx4" if "mx4" in scen else "nv4"
            it = {"item1_head_dw": (items[f"1_head_dw_{fm}"]["dgemm_ms"], acc["item1_ms"]), "item3_bb_dw": (items[f"3_bb_dw_{fm}"]["dgemm_ms"], acc["item3_dY_ms"] + acc["item3_X_ms"]),
                  "item4_fc_f": (items[f"4_fc_f_{fm}"]["dgemm_ms"], acc["item4_ms"]), "item5a_qkv_f_mx8": (items["5a_qkv_f_mx8"]["dgemm_ms"], acc["item5a_ms"]),
                  "item2a_head_f_mx8": (items["2a_head_f_mx8"]["dgemm_ms"], acc["item2a_ms"])}
            per_item = {k: {"dGEMM_ms": g, "dQuant_ms": q, "saving_ms": g - q} for k, (g, q) in it.items()}
            s1 = per_item["item1_head_dw"]["saving_ms"] + per_item["item3_bb_dw"]["saving_ms"]
            s2 = s1 + sum(per_item[k]["saving_ms"] for k in ("item4_fc_f", "item5a_qkv_f_mx8", "item2a_head_f_mx8"))
            summ[scen] = {"per_item": per_item, "S1_items_1+3_ms": s1, "S1_pct": 100 * s1 / STEP_MS[model], "S2_items_1+3+4+5a+2a_ms": s2, "S2_pct": 100 * s2 / STEP_MS[model],
                          "any_quant_scaled_by_elements": acc["any_scaled_by_elements"]}
        out["models"][model]["step_estimate"] = summ
        # speed conditions
        cond = {}
        cond["backbone_dw_nv4_vs_mx8 (>=1.5x)"] = items["3_bb_dw_nv4"]["speedup_x"]
        cond["backbone_dw_mx4_vs_mx8 (>=1.5x)"] = items["3_bb_dw_mx4"]["speedup_x"]
        cond["head_dw_nv4_vs_bf16 (>=3x)"] = items["1_head_dw_nv4"]["speedup_x"]
        cond["head_f_nv4_vs_bf16 (>=3x, stretch)"] = items["2b_head_f_nv4"]["speedup_x"]
        cond["head_f_mx8_vs_bf16 (>=2x)"] = items["2a_head_f_mx8"]["speedup_x"]
        cond["qkv_f_mx8_vs_bf16 (>=2x)"] = items["5a_qkv_f_mx8"]["speedup_x"]
        cond["fc_f_nv4_vs_mx8 (>=1.5x)"] = items["4_fc_f_nv4"]["speedup_x"]
        out["models"][model]["speed_conditions"] = cond
        # kernel-time context
        out["models"][model]["measured_gemm_vs_r19_kernel_table"] = {"head_bf16_GEMM_ms_from_matrix(d+w+f)": ((S["head_f.cur"] or 0) + (S["head_d.cur"] or 0) + (S["head_w.cur"] or 0)) / 1e3, "r19_head_gemm_ms": KERNEL_MS[model]["head_gemm"],
                                                                         "bb_mx8_GEMM_ms_from_matrix": sum((S[k] or 0) for k in ("bb_d.cur", "bb_w.cur", "bb_f_proj.cur", "bb_f_fc.cur")) / 1e3, "r19_bb_mx8_gemm_ms": KERNEL_MS[model]["bb_mx8_gemm"],
                                                                         "bb_bf16_qkv_f_ms_from_matrix": (S["bb_f_qkv.cur"] or 0) / 1e3, "r19_bb_bf16_qkv_ms": KERNEL_MS[model]["bb_bf16_qkv"]}
    if (P / "fp4_gemm_qutlass.json").exists():
        out["qutlass"] = json.load(open(P / "fp4_gemm_qutlass.json"))
    (P / "per_step_estimate.json").write_text(json.dumps(out, indent=1, default=str))
    # merged matrix file
    merged = {"models": {}}
    for model in INV:
        merged["models"][model] = {"raw": load(model)}
    merged["summary"] = {m: {"cases": out["models"][m]["cases"], "per_step_gemm_us": out["models"][m]["per_step_gemm_us"], "items": out["models"][m]["items"], "speed_conditions": out["models"][m]["speed_conditions"]} for m in INV}
    merged["qutlass"] = out["qutlass"]
    (P / "fp4_gemm_matrix.json").write_text(json.dumps(merged, indent=1, default=str))
    # console
    for model in INV:
        m = out["models"][model]
        print("=====", model)
        print("per-step GEMM ms:", {k: (None if v is None else round(v / 1e3, 2)) for k, v in m["per_step_gemm_us"].items() if k.endswith((".cur", ".nv4", ".mx8", ".bf16"))})
        for k, v in m["items"].items():
            print(f"  {k:46s} now {v['gemm_now_us'] and v['gemm_now_us'] / 1e3:.2f} ms -> {v['gemm_fp4_us'] and v['gemm_fp4_us'] / 1e3:.2f} ms  d={v['dgemm_ms']:.2f} ms  x{v['speedup_x']:.2f}" if v["gemm_now_us"] is not None and v["gemm_fp4_us"] is not None else f"  {k} n/a")
        print("conditions:", {k: (None if v is None else round(v, 2)) for k, v in m["speed_conditions"].items()})
        print("step estimate:", json.dumps(m["step_estimate"], default=lambda x: round(x, 2))[:1500])


if __name__ == "__main__":
    main()
