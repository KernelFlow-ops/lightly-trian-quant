"""Collect round23's profile/*.json into results.json (null for everything not measured) and profile/tables_r23.md (markdown tables for NOTES.md).  Re-runnable at any time.

  python tests/make_results.py
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
PROF = ROUND / "profile"
MODELS = ("vitl16", "vitb16")


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def files(pattern):
    return sorted(glob.glob(str(PROF / pattern)), key=lambda p: Path(p).stat().st_mtime)


def bench_section() -> dict:
    out: dict = {}
    for model in MODELS:
        runs: dict = {}
        for p in files(f"bench_{model}_*_all*.json"):
            d = load(p)
            if not d:
                continue
            s = d["summary"]
            stack = s["stack"]
            runs.setdefault(stack, []).append({
                "file": Path(p).name, "img_s": s["weighted_img_s"], "mid_ms": s["step_ms_mean"]["mid"], "early_ms": s["step_ms_mean"]["early"], "late_ms": s["step_ms_mean"]["late"],
                "peak_reserved_gb": s.get("peak_reserved_gb"), "reserved_after_setup_gb": (s.get("mem") or {}).get("after_setup", {}).get("reserved_gb"),
                "q2": s.get("q2"), "mid_repeat_ms": s["repeat_ms_mean"]["mid"]})
        rows = {}
        for stack, rr in runs.items():
            rows[stack] = {"runs": rr, "n": len(rr), "img_s_mean": statistics.fmean(r["img_s"] for r in rr), "mid_ms_mean": statistics.fmean(r["mid_ms"] for r in rr),
                           "early_ms_mean": statistics.fmean(r["early_ms"] for r in rr), "late_ms_mean": statistics.fmean(r["late_ms"] for r in rr),
                           "peak_reserved_gb_max": max((r["peak_reserved_gb"] or 0) for r in rr),
                           "img_s_drift_pct": 100 * (max(r["img_s"] for r in rr) / min(r["img_s"] for r in rr) - 1) if len(rr) > 1 else None}
        fin = rows.get("FINAL", {}).get("img_s_mean")
        r0 = rows.get("R0", {}).get("img_s_mean")
        for stack, row in rows.items():
            row["x_vs_final"] = row["img_s_mean"] / fin if fin else None
            row["x_vs_r0"] = row["img_s_mean"] / r0 if r0 else None
            row["mid_ms_delta_vs_final"] = row["mid_ms_mean"] - rows["FINAL"]["mid_ms_mean"] if fin else None
        out[model] = rows or None
    return out


def gate_b_section() -> dict:
    out: dict = {}
    for model in MODELS:
        draws = {}
        for p in files(f"gateB_p3_{model}_draw*.json"):
            d = load(p)
            if d:
                draws[Path(p).stem.split("_")[-1]] = d
        if not draws:
            out[model] = None
            continue
        merged: dict = {}
        per_draw = {}
        for name, d in draws.items():
            per_draw[name] = d.get("summary")
            for seed, s in d["seeds"].items():
                merged[f"{name}:{seed}"] = s
        rep = {"seeds": merged, "cands": next(iter(draws.values()))["cands"]}
        names = list(rep["cands"])
        import importlib.util
        spec = importlib.util.spec_from_file_location("gb", ROUND / "tests" / "gate_b_p3_summary.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[model] = {"per_draw": per_draw, "merged": mod.summarize(rep, names), "n_seeds_by_draw": {k: len(v["seeds"]) for k, v in draws.items()}}
    return out


def one(pattern):
    fs = files(pattern)
    return load(fs[-1]) if fs else None


def main() -> None:
    res: dict = {"round": "round23_fp4_final", "stacks": {"P3": "FINAL + round21 head (head.f=mx8,head.dw=nv4+sr) + round22 backbone (all.d=nv4+sr+all.w=nv4+sr [+nofirst for ViT-L])"}}
    res["smoke"] = {m: one(f"smoke_{m}_P3.json") for m in MODELS}
    res["bench"] = bench_section()
    res["gate_a"] = {k: one(pat) for k, pat in (("head_r21b", "gateA_round21_gateA_head_fp4.json"), ("block_vitb16", "gateA_round22_test_block22_vitb16.json"), ("block_vitl16", "gateA_round22_test_block22_vitl16.json"))}
    res["gate_b"] = gate_b_section()

    def pooled(m):
        """P3 / FINAL over the three main draws plus the two extra draws of the 5a queue (n = 40 batches, same-process pairing in every draw)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("gb", ROUND / "tests" / "gate_b_p3_summary.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        seeds = {}
        for f in files(f"gateB_p3_{m}_draw*.json") + files(f"gateB_q8_{m}_draw*.json"):
            for sd, s in load(f)["seeds"].items():
                seeds[f"{Path(f).stem}:{sd}"] = s
        if not seeds:
            return None
        sm = mod.summarize({"seeds": seeds, "cands": {"FINAL": [], "P3": []}}, ["FINAL", "P3"])
        return {n: {k: v for k, v in o.items() if k not in ("seeds",)} for n, o in sm.items()}

    res["gate_b_pooled_40_batches"] = {m: pooled(m) for m in MODELS}
    res["gate_s"] = {m: {p.name: (lambda d: {s: {"ema_max_rel_checkpoints": r["gate"]["curve300_max_rel"], "pass": r["gate"]["pass"], "finite": r["gate"]["finite"], "grad_norm_ratios": r["gate"]["grad_norm_ratios"],
                                                    "vs_final_max_rel": (r.get("gate_vs_final") or {}).get("curve300_max_rel"), "fp4": r["cand"].get("fp4")} for s, r in d["runs"].items()})(load(p))
                         for p in map(Path, files(f"gateS_{m}_P3_graph_vs_R0*.json"))} or None for m in MODELS}
    res["graph_equiv"] = {m: one(f"graph_equiv_{m}_P3.json") for m in MODELS}
    res["export"] = {m: {"run_final": one(f"run_final_{m}_export.json"), "check_export": one(f"export_{m}.json")} for m in MODELS}
    res["e2e"] = {m: one(f"e2e_{m}_P3.json") for m in MODELS}
    # optional 5a (qkv forward MXFP8 on top of P3): recorded only, not published
    def gs_q8(m):
        out = {}
        for p in map(Path, files(f"gateS_{m}_P3_q8_graph_vs_R0*.json")):
            d = load(p)
            for s, r in d["runs"].items():
                out[s] = {"ema_max_rel_checkpoints": r["gate"]["curve300_max_rel"], "pass": r["gate"]["pass"], "finite": r["gate"]["finite"], "grad_norm_ratios": r["gate"]["grad_norm_ratios"],
                          "vs_final_max_rel": (r.get("gate_vs_final") or {}).get("curve300_max_rel")}
        return out or None

    def gb_q8(m):
        draws = {Path(f).stem.split("_")[-1]: load(f) for f in files(f"gateB_q8_{m}_draw*.json")}
        if not draws:
            return None
        import importlib.util
        spec = importlib.util.spec_from_file_location("gb", ROUND / "tests" / "gate_b_p3_summary.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        merged = {"seeds": {f"{k}:{sd}": s for k, d in draws.items() for sd, s in d["seeds"].items()}, "cands": next(iter(draws.values()))["cands"]}
        return {"merged": mod.summarize(merged, [n for n in merged["cands"] if n != "FINAL_again"]), "n_seeds_by_draw": {k: len(v["seeds"]) for k, v in draws.items()}}

    res["optional_5a_qkv_mx8"] = {"policy": "P3 + qkv.f=mx8 in the backbone policy (r17 MXFP8 qkv forward; no new kernel)", "bench": {m: (res["bench"].get(m) or {}).get("P3_q8") for m in MODELS},
                                  "gate_s": {m: gs_q8(m) for m in MODELS}, "gate_b": {m: gb_q8(m) for m in MODELS}}
    res["nsys"] = one("nsys_report23.json")
    head = {}
    for m in MODELS:
        b = res["bench"].get(m) or {}
        gs = [r for f in res["gate_s"].get(m, {}).values() for r in f.values()] if res["gate_s"].get(m) else []
        ge = (res["graph_equiv"].get(m) or {}).get("scenarios", {})
        e2 = (res["e2e"].get(m) or {}).get("runs", [{}])[0] if res["e2e"].get(m) else {}
        pooled_m = (res["gate_b_pooled_40_batches"].get(m) or {}).get("P3")
        head[m] = {
            "img_s": {k: v["img_s_mean"] for k, v in b.items()}, "mid_ms": {k: v["mid_ms_mean"] for k, v in b.items()},
            "x_vs_final": {k: v["x_vs_final"] for k, v in b.items()}, "x_vs_r0": {k: v["x_vs_r0"] for k, v in b.items()},
            "peak_reserved_gb": {k: v["peak_reserved_gb_max"] for k, v in b.items()},
            "gate_s_P3_max_ema_dev_pct": (100 * max(r["ema_max_rel_checkpoints"] for r in gs)) if gs else None, "gate_s_P3_all_pass": all(r["pass"] and r["finite"] for r in gs) if gs else None,
            "graph_equiv_P3_fp4_state_advance_ok": all(r["fp4_state_advance"]["ok"] for r in ge.values()) if ge else None,
            "export_P3_check_pass": ((res["export"].get(m) or {}).get("check_export") or {}).get("pass"),
            "e2e_steady_vs_gpu_only": e2.get("steady_vs_gpu_only"),
            "gate_b_P3_pooled_n": pooled_m and pooled_m["n_batches"],
            "gate_b_P3_pooled_vs_r0_median_min": pooled_m and [pooled_m["vs_r0"]["gcos_median"], pooled_m["vs_r0"]["gcos_min"]],
            "gate_b_P3_pooled_vs_p1_median_min": pooled_m and [pooled_m["vs_p1"]["gcos_median"], pooled_m["vs_p1"]["gcos_min"]],
            "gate_b_P3_pooled_relaxed_pass": pooled_m and pooled_m["relaxed_pass_vs_r0_and_p1"], "gate_b_P3_pooled_strict_pass": pooled_m and pooled_m["strict_pass_vs_r0_and_p1"],
        }
    res["headline"] = head
    res["roofline"] = one("roofline23.json")
    (ROUND / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print("wrote results.json; keys:", {k: (v is not None if not isinstance(v, dict) else {kk: vv is not None for kk, vv in v.items()}) for k, v in res.items() if k not in ("round", "stacks")})


if __name__ == "__main__":
    main()
