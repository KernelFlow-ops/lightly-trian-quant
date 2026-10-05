"""Collect round24 results from profile/*.json into profile/tables_r24.md and results.json (nothing is invented: missing pieces stay null).

    python tests/make_results24.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
PROF = ROUND / "profile"
OUT: dict = {"round": "round24_fp4_quant_opt", "hot_microbench": {}, "bench_step": {}, "nsys": {}, "gate_A": None, "gate_A_bitexact_vs_r21a": None, "gate_B": {}, "gate_S": {}, "graph_equiv": {},
             "export_resume": "not run: fp4q2 changes no parameter / optimizer state (no torch.save / export difference to P3B)", "release": None, "notes": {}}
MD: list[str] = []


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ hot micro-benchmark
def hot():
    base = load(PROF / "bench_hot_base0.json")
    new = None
    for tag in ("final", "v5", "v4"):
        new = load(PROF / f"bench_hot_{tag}.json")
        if new:
            OUT["hot_microbench"]["new_source"] = f"bench_hot_{tag}.json"
            break
    OUT["hot_microbench"]["baseline_source"] = "bench_hot_v5.json (r21a column; same process, interleaved)" if new else None
    if not new:
        return
    MD.append("### Hot-input micro-benchmark (us, producer kernel + quantizer in one CUDA graph, quantizer time only; r21a and fp4q2 interleaved in one process)\n")
    MD.append("| model | site | R21a fp4q | fp4q2 | x |\n|---|---|---:|---:|---:|")
    rows = {}
    for model, sites in new["sites"].items():
        for key, d in sites.items():
            if "r21a" in d and "base" in d:
                a, b = d["r21a"], d["base"]
                rows[f"{model} {key}"] = {"r21a_us": a, "fp4q2_us": b, "speedup": round(a / b, 3)}
                if key.startswith(("dy[7203", "dual[7203", "row[7203", "x[7203", "dysr_row[7203", "dysr_col[7203", "dy[14406", "dual[14406", "x[14406", "dy[15806")) or key.startswith("w["):
                    MD.append(f"| {model} | {key} | {a} | {b} | {a / b:.2f} |")
    OUT["hot_microbench"]["sites"] = rows
    OUT["hot_microbench"]["per_step_us"] = new.get("per_step_us")
    MD.append(f"\nper step (dy + x + weights, counts of the real P3B / P3B_B step, hot): {json.dumps(new.get('per_step_us'))}\n")


# ------------------------------------------------------------------ step bench
LOGS = Path("/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/logs")


def chain_runs(tag: str) -> dict:
    """{model: {stack: [img/s of the runs in order]}} of one chain (chainA.log ... chainD.log: '== bench_<model>_<stack>_graph_<time>' followed by 'wrote .../<file>.json')."""
    p = LOGS / f"chain{tag}.log"
    out: dict = {}
    if not p.exists():
        return out
    cur = None
    for line in p.read_text().splitlines():
        if line.startswith("== bench_"):
            cur = line.split()[1]
        elif line.startswith("wrote ") and cur and "/profile/bench_" in line:
            d = load(line.split()[1])
            if d and "summary" in d:
                s = d["summary"]
                out.setdefault(s["model"], {}).setdefault(s["stack"], []).append({"img_s": s["weighted_img_s"], "mid_ms": s["step_ms_mean"]["mid"], "file": os.path.basename(line.split()[1])})
            cur = None
    return out


def bench():
    chains = {t: chain_runs(t) for t in ("A", "A2", "C", "D")}
    OUT["bench_step"] = chains
    MD.append("### Step speed (StepGraphRunner, 3-phase weighted 0.2/0.6/0.2; chain = one queue = one session, runs listed in execution order)\n")
    MD.append("| chain | model | stack | runs (img/s) | mean img/s | mid ms (mean) |\n|---|---|---|---|---:|---:|")
    ratios = {}
    for t, runs in chains.items():
        for m, d in runs.items():
            summ = {}
            for st, v in d.items():
                ips = [r["img_s"] for r in v]
                mids = [r["mid_ms"] for r in v]
                summ[st] = {"img_s": statistics.fmean(ips), "mid_ms": statistics.fmean(mids), "n": len(v)}
                MD.append(f"| {t} | {m} | {st} | {', '.join(f'{x:.2f}' for x in ips)} | {statistics.fmean(ips):.2f} | {statistics.fmean(mids):.2f} |")
            base = summ.get("P3B") or summ.get("P3B_B")
            fin = summ.get("FINAL")
            r = {}
            for st, v in summ.items():
                if fin and st != "FINAL":
                    r[f"{st}_over_FINAL"] = round(v["img_s"] / fin["img_s"], 4)
                if base and st not in ("FINAL", "P3B", "P3B_B"):
                    r[f"{st}_over_P3B"] = round(v["img_s"] / base["img_s"], 4)
            if len([k for k in summ]) > 1:
                ratios.setdefault(t, {})[m] = {"summary": summ, "ratios": r}
    OUT["bench_step_sessions"] = ratios
    MD.append("\nratios per session: " + json.dumps({t: {m: v["ratios"] for m, v in d.items()} for t, d in ratios.items()}) + "\n")
    d_ = ratios.get("D")
    rel = {}
    if d_:
        for m, v in d_.items():
            key = [k for k in v["ratios"] if k.startswith("P3B") and k.endswith("q2_over_P3B")]
            key = key[0] if key else None
            if key:
                pct = round((v["ratios"][key] - 1) * 100, 2)
                rel[m] = {"speedup_over_P3B_pct": pct, "ge_2pct": bool(pct >= 2.0), "session": "D (P3B-q2-q2-P3B)"}
    OUT["release"] = {"rule": "whole step >= 2% faster than P3B (either model) AND all gates pass", "per_model_step": rel or None,
                      "step_condition_met": bool(any(v["ge_2pct"] for v in rel.values())) if rel else None}


# ------------------------------------------------------------------ nsys
def nsys():
    p = PROF / "nsys_report24.json"
    d = load(p)
    if not d:
        return
    MD.append("### nsys per-step category table (ms/step)\n")
    for label, v in d.items():
        cats = v.get("categories") or {}
        fp4 = [(k, c) for k, c in (cats.items() if isinstance(cats, dict) else []) if "FP4 quantize" in k]
        mx8 = [(k, c) for k, c in (cats.items() if isinstance(cats, dict) else []) if "MXFP8 quantize" in k]
        OUT["nsys"][label] = {"kernel_ms_per_step": v.get("kernel_ms_per_step"), "kernels_per_step": v.get("kernels_per_step"), "fp4_quantize": fp4[0][1] if fp4 else None, "mxfp8_quantize": mx8[0][1] if mx8 else None}
        MD.append(f"- {label}: kernel time {v.get('kernel_ms_per_step')} ms/step, FP4 quantize {fp4[0][1] if fp4 else None}, MXFP8 quantize {mx8[0][1] if mx8 else None}")
    MD.append("")


# ------------------------------------------------------------------ gates
def gates():
    a = load(PROF / "gateA_fp4q2.json")
    if a:
        OUT["gate_A"] = {"all_pass": a.get("all_pass"), "parts": {k: {"cases": v.get("cases") if not isinstance(v.get("cases"), list) else len(v["cases"]), "fail": v.get("fail")} for k, v in a.get("parts", {}).items()}}
        MD.append("### Gate A (tests/test_fp4q2.py, copy of round21 test_fp4q.py on fp4q2 + parts fast / philox7)\n")
        MD.append("| part | cases | failures |\n|---|---:|---:|")
        for k, v in OUT["gate_A"]["parts"].items():
            MD.append(f"| {k} | {v['cases']} | {v['fail']} |")
        if "philox7" in a.get("parts", {}):
            ph = a["parts"]["philox7"]
            OUT["gate_A"]["philox7_stats"] = {k: ph.get(k) for k in ("stream_stats", "avalanche_flip_fraction", "q_bytes_changed_10_vs_7_frac", "sr_mean_vs_rn_mean_abs_diff", "stream_checks_pass")}
        MD.append("")
    for model in ("vitl16", "vitb16"):
        b = load(PROF / f"gateB_r24_{model}.json")
        if b:
            summ = b.get("summary", {})
            OUT["gate_B"][model] = summ
            MD.append(f"### Gate B {model} (n = {max([s.get('n_batches', 0) for s in summ.values()] or [0])} batches; same process: R0, P1, FINAL, round22 + fp4q (real), round22 + fp4q2 (real2))\n")
            MD.append("| cand | n | vsR0 med / min | vsP1 med / min | vs R21a-real cos med / min | relaxed | strict |\n|---|---:|---|---|---|---|---|")
            for n, s in summ.items():
                vr = s.get("vs_real_r21a")
                MD.append(f"| {n} | {s['n_batches']} | {s['vs_r0']['gcos_median']:.4f} / {s['vs_r0']['gcos_min']:.4f} | {s['vs_p1']['gcos_median']:.4f} / {s['vs_p1']['gcos_min']:.4f} | "
                          f"{('%.6f / %.6f' % (vr['gcos_median'], vr['gcos_min'])) if vr else '-'} | {s['relaxed_pass_vs_r0_and_p1']} | {s['strict_pass_vs_r0_and_p1']} |")
            MD.append("")
    for f in sorted(glob.glob(str(PROF / "gateS_vit*_P3B*_q2_graph_vs_R0*.json"))):
        d = load(f)
        if not d:
            continue
        key = os.path.basename(f)
        runs = d.get("runs", {})
        OUT["gate_S"][key] = {scen: {"max_rel": r["gate"].get("curve300_max_rel"), "pass": r["gate"].get("pass"), "finite": r["gate"].get("finite"),
                                      "fp4": {k: r["cand"].get("fp4", {}).get(k) for k in ("total_calls", "total_exceed", "exceed_rate")}} for scen, r in runs.items()}
    if OUT["gate_S"]:
        MD.append("### Gate S (StepGraphRunner, graph vs R0 eager, 50 steps x 2 batches)\n")
        MD.append("| file | scenario | max rel EMA dev | pass | finite | fp4 exceed rate |\n|---|---|---:|---|---|---:|")
        for k, v in OUT["gate_S"].items():
            for scen, r in v.items():
                MD.append(f"| {k} | {scen} | {r['max_rel']} | {r['pass']} | {r['finite']} | {r['fp4'].get('exceed_rate')} |")
        MD.append("")
    for f in sorted(glob.glob(str(PROF / "graph_equiv_vit*_q2.json"))) + sorted(glob.glob(str(PROF / "graph_equiv_vit*_P3B*_q2.json"))):
        d = load(f)
        if d:
            OUT["graph_equiv"][os.path.basename(f)] = {k: d[k] for k in list(d)[:12] if not isinstance(d[k], (dict, list)) or k in ("verdict", "summary", "pass")}


def main():
    hot()
    bench()
    nsys()
    gates()
    (PROF / "tables_r24.md").write_text("\n".join(MD) + "\n")
    old = load(ROUND / "results.json") or {}
    old.update(OUT)
    (ROUND / "results.json").write_text(json.dumps(old, indent=1, allow_nan=False))
    print("wrote", PROF / "tables_r24.md", "and results.json")


if __name__ == "__main__":
    main()
