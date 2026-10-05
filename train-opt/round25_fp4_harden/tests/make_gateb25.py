"""Combine round25 gate-B draws (profile/gateB_<tag>_<model>_draw*.json: one process = one draw = its own 8 real batches) into per-candidate tables.

For every candidate: n, global-gradient-cosine median / min / #batches below 0.98 and 0.97 vs R0 and vs P1, relaxed / strict verdict (CONTRACTS 11.5: relaxed = median >= 0.98, min >= 0.97, loss_rel <= 2e-2,
vs R0 AND vs P1; strict = median >= 0.99, min >= 0.98), the margin of the worst batch over the 0.97 line, and the PAIRED per-batch difference to P3 (candidate - P3, vs R0).
usage: python tests/make_gateb25.py <tag> <model> [--json out.json]     e.g. hr vitl16   /   h2 vitb16
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
PROF = ROUND / "profile"
sys.path.insert(0, str(ROUND / "tests"))
from gate_b_p3_summary import summarize  # noqa: E402


def combine(tag: str, model: str) -> dict:
    files = sorted(glob.glob(str(PROF / f"gateB_{tag}_{model}_draw*.json")))
    merged = {"cands": {}, "seeds": {}, "files": files}
    for f in files:
        d = json.loads(Path(f).read_text())
        draw = Path(f).stem.split("_")[-1]
        for k, v in d["seeds"].items():
            merged["seeds"][f"{draw}:{k}"] = v
        merged["cands"].update(d["cands"])
    return merged


def table(tag: str, model: str) -> dict:
    rep = combine(tag, model)
    names = list(rep["cands"])
    summ = summarize(rep, names)
    out = {"files": [Path(f).name for f in rep["files"]], "n_batches": max((s["n_batches"] for s in summ.values()), default=0), "cands": {}}
    base = "P3"
    cos_r0 = {n: {k: s["cands"][n]["vs_r0"]["grad_cos_global"] for k, s in rep["seeds"].items() if n in s["cands"] and "vs_r0" in s["cands"][n]} for n in names}
    cos_p1 = {n: {k: s["cands"][n]["vs_p1"]["grad_cos_global"] for k, s in rep["seeds"].items() if n in s["cands"] and "vs_p1" in s["cands"][n]} for n in names}
    for n in names:
        if n not in summ:
            continue
        s = summ[n]
        r0, p1 = list(cos_r0[n].values()), list(cos_p1[n].values())
        row = {"policy": s["policy"], "n": s["n_batches"],
               "vs_r0": {"median": s["vs_r0"]["gcos_median"], "min": s["vs_r0"]["gcos_min"], "min_batch": s["vs_r0"]["gcos_min_seed"], "n_lt_098": sum(v < 0.98 for v in r0), "n_lt_097": sum(v < 0.97 for v in r0),
                         "loss_rel_max": s["vs_r0"]["loss_rel_max"], "relaxed": s["vs_r0"]["relaxed_ok"], "strict": s["vs_r0"]["strict_ok"]},
               "vs_p1": {"median": s["vs_p1"]["gcos_median"], "min": s["vs_p1"]["gcos_min"], "min_batch": s["vs_p1"]["gcos_min_seed"], "n_lt_098": sum(v < 0.98 for v in p1), "n_lt_097": sum(v < 0.97 for v in p1),
                         "loss_rel_max": s["vs_p1"]["loss_rel_max"], "relaxed": s["vs_p1"]["relaxed_ok"], "strict": s["vs_p1"]["strict_ok"]},
               "relaxed_both": s["relaxed_pass_vs_r0_and_p1"], "strict_both": s["strict_pass_vs_r0_and_p1"],
               "margin_over_097": min(s["vs_r0"]["gcos_min"], s["vs_p1"]["gcos_min"]) - 0.97,
               "grad_norm_rel_max": s["vs_r0"]["grad_norm_rel_max"], "peak_gb_max": s["peak_gb_max"]}
        if base in cos_r0 and n != base:
            common = [k for k in cos_r0[n] if k in cos_r0[base]]
            diff = [cos_r0[n][k] - cos_r0[base][k] for k in common]
            row["paired_vs_P3_r0"] = {"n": len(diff), "median": statistics.median(diff), "min": min(diff), "max": max(diff), "n_better": sum(d > 0 for d in diff)}
        if n.startswith("FINAL"):
            pass
        out["cands"][n] = row
    # worst 3 batches of P3 and which candidate fixes them
    if base in cos_r0:
        worst = sorted(cos_r0[base].items(), key=lambda kv: kv[1])[:4]
        out["worst_P3_batches_vs_r0"] = {k: {n: round(cos_r0[n].get(k, float("nan")), 4) for n in names} for k, _ in worst}
    return out


def md(model_name: str, t: dict, order: list[str] | None = None) -> str:
    lines = [f"| 候选 | 策略 | n | 对 R0 中位 / 最小 | <0.98 / <0.97 | 对 P1 中位 / 最小 | <0.98 / <0.97 | loss_rel 最大 | 放宽门 | 严格门 | 最差批余量（对 0.97） | 对 P3 逐批差（中位 / 最小 / 变好批数） |",
             "|---|---|---:|---|---|---|---|---:|---|---|---:|---|"]
    names = order or list(t["cands"])
    for n in names:
        if n not in t["cands"]:
            continue
        r = t["cands"][n]
        pr = r.get("paired_vs_P3_r0")
        pairs = f"{pr['median']:+.4f} / {pr['min']:+.4f} / {pr['n_better']}/{pr['n']}" if pr else "-"
        pol = (r["policy"][0] or "-") + " | " + (r["policy"][1] or "-") if r["policy"] else "-"
        pol = f"head `{r['policy'][0] or '-'}`" if r["policy"] and n not in ("FINAL", "FINAL_again") else "FINAL"
        lines.append(f"| {n} | {pol} | {r['n']} | {r['vs_r0']['median']:.4f} / {r['vs_r0']['min']:.4f} | {r['vs_r0']['n_lt_098']} / {r['vs_r0']['n_lt_097']} | {r['vs_p1']['median']:.4f} / {r['vs_p1']['min']:.4f} | "
                     f"{r['vs_p1']['n_lt_098']} / {r['vs_p1']['n_lt_097']} | {max(r['vs_r0']['loss_rel_max'], r['vs_p1']['loss_rel_max']):.1e} | {'过' if r['relaxed_both'] else '不过'} | "
                     f"{'过' if r['strict_both'] else '不过'} | {r['margin_over_097']:+.4f} | {pairs} |")
    return "\n".join(lines)


if __name__ == "__main__":
    tag, model = sys.argv[1], sys.argv[2]
    t = table(tag, model)
    print(md(model, t))
    print("worst P3 batches (vs R0):", json.dumps(t.get("worst_P3_batches_vs_r0")))
    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps(t, indent=1, default=str))
        print("wrote", out)
