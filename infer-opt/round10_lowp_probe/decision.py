"""Decision (PLAN_v2 R10 step 5) from the measured matrices + the finalize plans.   CPU only.

  python decision.py stage1     # apply the pre-registered per-(model, kind, tier) go rule (saving >= 0.15 ms / frame) to profile/plan_*.json -> profile/final_<model>_<preset><tag>.json
  python decision.py stage2     # merge GPU verification results (profile/verify_*.json) -> profile/decision.json

Pre-registered go rules (PLAN_v2 3 / R10 step 5): tier item = per-frame saving >= 0.15 ms AND passes the preset's gate (safe: M + H, fast: M);
P1 additionally: instruction rate >= 1.7x bf16 (peaks), prototype incl. epilogue >= 1.3x r5_v1 on our shapes, overflow scan finds a safe pre-scale for >= 90 % of the GEMMs.
Saving of a (model, kind, tier) group = sum over its layers of [r5_v1 time - (tier kernel incl. any separately needed epilogue + activation quantisation + prefix-row side GEMM)]
from speed_model.Speed (measured matrices); 'central' assumes the quantisation fused into the producer where possible, 'pess' uses the stand-alone quantiser kernels.
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "impl"))
from speed_model import Speed, LAYERS, KINDS  # noqa: E402

TIER = {"fp16": "P1", "int8": "P2", "mx8": "P2", "nv4": "P3", "mx4": "P3", "bf16": "P0"}
GO_MS = 0.15


def J(p):
    p = HERE / p
    return json.loads(p.read_text()) if p.exists() else {}


def units_of(entries, nb):
    from emu import HEAD_KINDS, layers_of
    u = OrderedDict()
    for e in entries:
        for k in e["kinds"]:
            if k in HEAD_KINDS:
                u[("head", k)] = e["spec"]
            else:
                for i in layers_of(e.get("layers", "all"), nb):
                    u[(i, k)] = e["spec"]
    return u


def head_saving(model, hk, fmt):
    """ms / frame saved by running head GEMM `hk` (ct1 / ct2) in `fmt` vs the cuBLAS bf16 GEMM measured in the same harness; (central, pess)."""
    ex = J("results_gemm_extra1.json")
    q = J("results_quant.json")
    M = {"ct1": 2048, "ct2": 8192}[hk]
    r = ex.get(f"{model}|{hk}|{M}", {})
    base = r.get("cublas_bf16", {}).get("us")
    if base is None:
        return None, None
    scale = M / 2058.0
    site = q.get(f"{model}|ln|2058", {})
    if fmt == "mx8":
        g = r.get("cublaslt_mx8", {}).get("us")
        qq = (site.get("quant_dual_r17") or {}).get("us")
        if g is None or qq is None:
            return None, None
        return (base - g - 1.0 * scale) / 1e3, (base - g - qq * scale) / 1e3
    if fmt == "fp16":
        g = r.get("mg16_tuned", {}).get("us")
        if g is None:
            return None, None
        return (base - g) / 1e3, (base - g - 2.0 * scale) / 1e3        # pess: stand-alone bf16->fp16 convert pass
    return None, None


def group_plan(model, plan, sp):
    nb = LAYERS[model]
    u = units_of(plan["entries"], nb)
    groups = defaultdict(lambda: defaultdict(list))        # kind -> spec-key -> [(layer, spec)]
    head = {}
    for (l, k), spec in u.items():
        if l == "head":
            head[k] = spec
        else:
            key = json.dumps({x: spec.get(x) for x in ("fmt", "sq", "had", "pfx", "gptq")}, sort_keys=True)
            groups[k][key].append((l, spec))
    return groups, head


def stage1():
    out = {}
    sp = Speed()
    for f in sorted((HERE / "profile").glob("plan_*.json")):
        plan = json.loads(f.read_text())
        model, preset, tag = plan["model"], plan["preset"], plan.get("tag", "")
        groups, head = group_plan(model, plan, sp)
        kept, rows = [], []
        for kind in KINDS:
            for key, lst in sorted(groups.get(kind, {}).items()):
                spec = json.loads(key)
                layers = sorted(l for l, _ in lst)
                fmt = spec["fmt"]
                tier = {"fp16": "fp16", "int8": "int8", "mx8": "mx8", "nv4": "nv4", "mx4": "mx4"}[fmt]
                sc = sp.saving_ms(model, kind, tier, fmt, layers, "central", bool(spec["pfx"]))
                spp = sp.saving_ms(model, kind, tier, fmt, layers, "pess", bool(spec["pfx"]))
                go = sc is not None and sc >= GO_MS
                rows.append({"kind": kind, "tier": TIER[fmt], "spec": spec, "n_layers": len(layers), "layers": layers, "saving_ms_central": sc, "saving_ms_pess": spp, "go": bool(go)})
                if go:
                    kept.append({"kinds": [kind], "layers": layers, "spec": {x: v for x, v in spec.items() if v not in (None, 0, False, "")} | {"fmt": fmt}})
        hrows = []
        for hk, spec in head.items():
            sc, spp = head_saving(model, hk, spec["fmt"])
            go = sc is not None and sc >= GO_MS
            hrows.append({"kind": hk, "tier": TIER[spec["fmt"]], "fmt": spec["fmt"], "saving_ms_central": sc, "saving_ms_pess": spp, "go": bool(go)})
            if go:
                kept.append({"kinds": [hk], "layers": "all", "spec": {"fmt": spec["fmt"]}})
        total_c = sum(r["saving_ms_central"] for r in rows if r["go"]) + sum(r["saving_ms_central"] for r in hrows if r["go"])
        total_p = sum(r["saving_ms_pess"] for r in rows if r["go"] and r["saving_ms_pess"] is not None) + sum(r["saving_ms_pess"] for r in hrows if r["go"] and r["saving_ms_pess"] is not None)
        name = f"{model}_{preset}{tag}"
        out[name] = {"model": model, "preset": preset, "tag": tag, "backbone_groups": rows, "head": hrows, "entries_after_go_rule": kept,
                     "saving_ms_central_after_go_rule": total_c, "saving_ms_pess_after_go_rule": total_p, "plan_before_go_rule": {"saving_ms_central": plan["saving_ms_central"], "M": plan["M"], "H": plan["H"]}}
        (HERE / "profile" / f"final_{name}.json").write_text(json.dumps(out[name], indent=1))
        print(f"{name}: groups {[(r['kind'], r['tier'], r['spec']['fmt'], r['n_layers'], round(r['saving_ms_central'], 3) if r['saving_ms_central'] is not None else None, r['go']) for r in rows]}  head {[(h['kind'], h['fmt'], h['saving_ms_central'], h['go']) for h in hrows]}  total {total_c:.3f}/{total_p:.3f} ms")
    return out


def p1_rules():
    """numbers behind the P1 go rule."""
    peaks = J("results.json").get("peaks", {})
    rate = peaks.get("ratios_vs_bf16_f32acc", {}).get("fp16_f16acc")
    sp = Speed()
    res = {"instruction_rate_vs_bf16": rate, "rule_rate": 1.7}
    per = {}
    for model in LAYERS:
        for kind in KINDS:
            r5 = sp.frame_ms(model, kind, "r5_v1", "bf16")
            f16 = sp.frame_ms(model, kind, "fp16", "bf16")
            f16c = sp.frame_ms(model, kind, "fp16_cublas", "bf16")
            per[f"{model}|{kind}"] = {"r5_v1_ms": r5, "mg16_ms": f16, "ratio_mg16_vs_r5": None if not (r5 and f16) else r5 / f16, "saving_ms": None if not (r5 and f16) else r5 - f16,
                                      "fp16_cublas_plus_epi_ms": f16c, "pass_ratio_1.3": bool(r5 and f16 and r5 / f16 >= 1.3), "pass_saving_0.15": bool(r5 and f16 and r5 - f16 >= GO_MS)}
    res["per_kind"] = per
    ov = {}
    for model in LAYERS:
        d = J(f"profile/overflow_scan_{model}.json")
        s = d.get("summary", {}).get("chunk32", {})
        ov[model] = {"n_gemms": s.get("n_gemms"), "gemms_with_safe_prescale_fraction": 1.0 if s.get("max_p_all") == 0 and s.get("absbound_overflow_all_rows") == 0 else None, "max_prescale_exponent": s.get("max_p_all")}
    res["overflow"] = ov
    return res


def stage2():
    final = {}
    for f in sorted((HERE / "profile").glob("final_*.json")):
        d = json.loads(f.read_text())
        name = f.stem[len("final_"):]
        v = J(f"profile/verify_{name}.json")
        d["verification"] = v or "not measured"
        vr = J(f"profile/verify_{name}_realkernel.json")
        d["verification_real_int8_kernel"] = vr or "not measured"
        final[name] = d
    dec = {"meta": {"date": "2026-10-04", "go_rule": f"per (model, kind, tier) group: per-frame saving >= {GO_MS} ms (r5_v1 time - tier kernel incl. separate epilogue pass - activation quantisation - prefix side GEMM) AND passes the preset gate (safe M+H, fast M); "
                                              "'margin' variants additionally require pooled IoU >= 0.965 and max conf rel diff <= 3 % on the gated sets",
                    "calibration": "16 frames of 1.mp4 not in the 8 eval frames (emu_eval.CAL_IDX); gate H = the other 110 frames",
                    "caveat": "single video; the combination search is a greedy over measured configurations (selection bias); re-validate on other videos before shipping"},
           "p1_rules": p1_rules(), "plans": final}
    (HERE / "profile" / "decision.json").write_text(json.dumps(dec, indent=1))
    print("decision.json written;", list(final))
    return dec


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2}[sys.argv[1]]()
