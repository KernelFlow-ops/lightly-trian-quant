"""Final decision tables (PLAN_v2 R10 step 5).  CPU only.

  python decision2.py stage1     # candidate groups -> go rules -> profile/final_<model>_<variant>.json   (variants: strict = every clause of the pre-registered rule incl. the P1 1.3x prototype clause;
                                 #                                                                   recommended = P1 ratio clause waived, see NOTES)
  python decision2.py stage2     # + verification files (profile/verify_*.json) + robustness files (profile/robust_*.json) -> profile/decision.json

Go rules applied per (model, GEMM kind, tier) group (all layers of the kind):
  R1  per-frame saving >= 0.15 ms  (r5_v1 time - tier kernel incl. separately needed epilogue - activation quantisation [- prefix side GEMM]); 'central' saving is used, 'pess' is reported
  R2  gate: safe = M + H, fast = M, evaluated with the REAL kernel path where one exists (mg_i8) and required in >= 4 of 5 independent noise realisations (robust_plans.py) --
      this robustness requirement is an addition to the pre-registered text (ViT-L conf is chaotic: see NOTES), fp16: both measured accumulation realisations (tuned16 / pkt1 tiles) must pass
  R3  (P1 only) peak rate >= 1.7x bf16 (measured 1.99x), prototype incl. epilogue >= 1.3x r5_v1 on the (model, kind) shapes, overflow scan finds a safe pre-scale for >= 90 % of the GEMMs (measured 100 %: no pre-scale needed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "impl"))
from speed_model import Speed, LAYERS, KINDS  # noqa: E402
from decision import head_saving, p1_rules, J, GO_MS  # noqa: E402

TIER = {"fp16": "P1", "int8": "P2", "mx8": "P2", "nv4": "P3", "mx4": "P3"}
FRAME_MS_NSYS = {"vitl16": 9.744, "vitb16": 3.787}


def robust_flags(model, fmt, kind):
    """(passes_M, passes_MH, n, source) for the single-kind candidate."""
    if fmt == "int8":
        d = J(f"profile/robust_{model}.json").get(f"int8_{kind}_had16|real")
        if not d:
            return None
        return d["pass_M"], d["pass_M_and_H"], d["n_seeds"], f"robust_{model}.json int8_{kind}_had16|real (seeds 0-4, randomised Hadamard signs)"
    if fmt == "fp16":
        runs = []
        for f, key in ((f"profile/robust_{model}_fp16_tuned.json", f"fp16_{kind}|emu"), (f"profile/robust_{model}_fp16_pkt1.json", f"fp16_{kind}|emu")):
            d = J(f).get(key)
            if d:
                runs.append(d)
        if not runs:
            d = J(f"profile/robust_{model}.json").get("fp16_all4|emu")
            if d:
                runs.append(d)
        if not runs:
            return None
        return sum(r["pass_M"] for r in runs), sum(r["pass_M_and_H"] for r in runs), sum(r["n_seeds"] for r in runs), "fp16 kernel realisations (tuned16 tiles / pkt1 tiles)"
    return None


def candidate_groups(sp):
    out = {}
    for model in LAYERS:
        rows = []
        for kind in KINDS:
            for fmt, spec in (("fp16", {"fmt": "fp16"}), ("int8", {"fmt": "int8", "had": 16})):
                layers = list(range(LAYERS[model]))
                sc = sp.saving_ms(model, kind, fmt, fmt, layers, "central")
                spp = sp.saving_ms(model, kind, fmt, fmt, layers, "pess")
                r5 = sp.frame_ms(model, kind, "r5_v1", "bf16")
                k = sp.frame_ms(model, kind, fmt, "bf16")             # kernel only (no quantisation)
                rob = robust_flags(model, fmt, kind)
                rows.append({"kind": kind, "tier": TIER[fmt], "spec": spec, "n_layers": len(layers), "saving_ms_central": sc, "saving_ms_pess": spp,
                             "kernel_ratio_vs_r5_v1": None if not (r5 and k) else r5 / k, "robust_M_pass": None if rob is None else rob[0], "robust_MH_pass": None if rob is None else rob[1],
                             "robust_n": None if rob is None else rob[2], "robust_source": None if rob is None else rob[3]})
        hrows = []
        for hk in ("ct1", "ct2"):
            sc, spp = head_saving(model, hk, "mx8")
            hrows.append({"kind": hk, "tier": "P2", "spec": {"fmt": "mx8"}, "saving_ms_central": sc, "saving_ms_pess": spp})
        out[model] = {"backbone": rows, "head": hrows}
    return out


def decide(model, groups, preset, variant):
    """pick, per kind, the go group with the largest saving."""
    need = "robust_MH_pass" if preset == "safe" else "robust_M_pass"
    chosen, rows = {}, []
    for g in groups["backbone"]:
        rob_ok = g["robust_n"] is not None and g[need] >= (4 if g["robust_n"] == 5 else g["robust_n"])
        sav_ok = g["saving_ms_central"] is not None and g["saving_ms_central"] >= GO_MS
        p1_ok = True
        if g["tier"] == "P1" and variant == "strict":
            p1_ok = g["kernel_ratio_vs_r5_v1"] is not None and g["kernel_ratio_vs_r5_v1"] >= 1.3
        go = rob_ok and sav_ok and p1_ok
        why = []
        if not sav_ok:
            why.append("saving < 0.15 ms")
        if not rob_ok:
            why.append(f"gate not robust ({g[need]}/{g['robust_n']} realisations)" if g["robust_n"] else "gate not measured")
        if not p1_ok:
            why.append(f"P1 prototype ratio {g['kernel_ratio_vs_r5_v1']:.2f} < 1.3")
        rows.append({**g, "go": go, "no_go_reason": "; ".join(why)})
        if go and (g["kind"] not in chosen or g["saving_ms_central"] > chosen[g["kind"]]["saving_ms_central"]):
            chosen[g["kind"]] = g
    hrows = []
    chosen_head = []
    for h in groups["head"]:
        # head mx8: gate evidence = G5 sweep (ViT-L: all pass; single realisation) -> accepted only if saving >= 0.15
        sav_ok = h["saving_ms_central"] is not None and h["saving_ms_central"] >= GO_MS
        hrows.append({**h, "go": bool(sav_ok), "no_go_reason": "" if sav_ok else "saving < 0.15 ms"})
        if sav_ok:
            chosen_head.append(h)
    entries = []
    for k, g in chosen.items():
        entries.append({"kinds": [k], "layers": "all", "spec": g["spec"]})
    for h in chosen_head:
        entries.append({"kinds": [h["kind"]], "layers": "all", "spec": h["spec"]})
    sv_c = sum(g["saving_ms_central"] for g in chosen.values()) + sum(h["saving_ms_central"] for h in chosen_head)
    sv_p = sum(g["saving_ms_pess"] for g in chosen.values()) + sum((h["saving_ms_pess"] or 0.0) for h in chosen_head)
    return {"model": model, "preset": preset, "variant": variant, "backbone_groups": rows, "head_groups": hrows, "per_kind_tier": {k: TIER[g["spec"]["fmt"]] if k in chosen else "P0" for k in KINDS for g in [chosen.get(k, {"spec": {"fmt": "bf16"}})]},
            "per_kind_spec": {k: chosen[k]["spec"] if k in chosen else {"fmt": "bf16"} for k in KINDS}, "head_spec": {h["kind"]: h["spec"] for h in chosen_head},
            "entries": entries, "saving_ms_per_frame_central": sv_c, "saving_ms_per_frame_pess": sv_p,
            "frame_gpu_ms_r5_v1_nsys": FRAME_MS_NSYS[model], "frame_gpu_ms_estimated_central": FRAME_MS_NSYS[model] - sv_c, "frame_gpu_ms_estimated_pess": FRAME_MS_NSYS[model] - sv_p}


def stage1():
    sp = Speed()
    cg = candidate_groups(sp)
    (HERE / "profile" / "candidate_groups.json").write_text(json.dumps(cg, indent=1))
    made = {}
    for model in LAYERS:
        for variant in ("strict", "recommended"):
            for preset in ("safe", "fast"):
                d = decide(model, cg[model], preset, variant)
                made[(model, preset, variant)] = d
            # safe and fast give identical entries here; write one file per (model, variant, preset) anyway
        for variant in ("strict", "recommended"):
            for preset in ("safe", "fast"):
                d = made[(model, preset, variant)]
                (HERE / "profile" / f"final_{model}_{preset}_{variant}.json").write_text(json.dumps({**d, "entries_after_go_rule": d["entries"]}, indent=1))
                print(f"{model} {preset:4s} {variant:11s}: per-kind {d['per_kind_tier']} head {list(d['head_spec'])}  saving {d['saving_ms_per_frame_central']:.3f}/{d['saving_ms_per_frame_pess']:.3f} ms  -> {d['frame_gpu_ms_estimated_central']:.2f} ms/frame (nsys {d['frame_gpu_ms_r5_v1_nsys']})")


def stage2():
    plans = {}
    for f in sorted((HERE / "profile").glob("final_*_*_strict.json")) + sorted((HERE / "profile").glob("final_*_*_recommended.json")):
        name = f.stem[len("final_"):]
        d = json.loads(f.read_text())
        d.pop("entries_after_go_rule", None)
        d["verification"] = J(f"profile/verify_{name}.json") or "not measured"
        d["verification_real_int8_kernel"] = J(f"profile/verify_{name}_realkernel.json") or "not measured"
        plans[name] = d
    robust = {m: {**J(f"profile/robust_{m}.json"), **{f"[fp16 tuned] {k}": v for k, v in J(f"profile/robust_{m}_fp16_tuned.json").items()}, **{f"[fp16 pkt1] {k}": v for k, v in J(f"profile/robust_{m}_fp16_pkt1.json").items()}} for m in LAYERS}
    for m in robust:
        for k, v in robust[m].items():
            v.pop("entries", None)
    greedy = {}
    for f in sorted((HERE / "profile").glob("plan_*.json")):
        d = json.loads(f.read_text())
        greedy[f.stem[len("plan_"):]] = {"saving_ms_central": d["saving_ms_central"], "M": d["M"], "H": d["H"], "n_entries": len(d["entries"])}
    dec = {"meta": {"date": "2026-10-04", "round": "round10_lowp_probe",
                    "rules": ("per (model, kind, tier): saving >= 0.15 ms/frame AND gate (safe M+H, fast M) robust over >= 4/5 noise realisations with the real kernel where one exists; "
                              "P1: peak rate >= 1.7x, prototype incl. epilogue >= 1.3x r5_v1 (waived in the 'recommended' variant), overflow-scan pre-scale for >= 90 % of GEMMs"),
                    "variants": {"strict": "every clause of the pre-registered rule", "recommended": "P1 prototype-ratio clause waived (kernel headroom shown by cuBLAS fp16-acc), everything else unchanged"},
                    "calibration": "16 frames of 1.mp4 not in the 8 eval frames (emu_eval.CAL_IDX); gate H = the other 110 frames",
                    "caveat": "single video; ViT-L confidence is chaotic (see NOTES); re-validate on other videos; tier savings are estimates from measured kernels, not whole-model measurements"},
           "p1_rules": p1_rules(), "plans": plans, "robustness": robust, "greedy_search_plans_before_robustness": greedy,
           "baseline_frame_gpu_ms_nsys": FRAME_MS_NSYS}
    (HERE / "profile" / "decision.json").write_text(json.dumps(dec, indent=1))
    print("decision.json written;", list(plans))


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2}[sys.argv[1]]()
