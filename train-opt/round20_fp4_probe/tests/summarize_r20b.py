"""Tables for the R20b NOTES section from profile/gateB_r20b_*.json and gateS_r20b_*.json  ->  profile/tables_r20b.md  (stdout too).

    python tests/summarize_r20b.py [--results]     (--results also merges a compact ``r20b`` block into results.json, keeping the R20a content)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
PROF = ROUND / "profile"
MODELS = ("vitl16", "vitb16")


def f(x, n=4):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def gateb_table(rep, title, names=None, min_n=1):
    lines = [f"#### {title}", "", "| 候选 | n | 对 R0 中位/最小 | 对 P1 中位/最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 | 逐参数最小 | eager 仿真 fwd+bwd s（非速度） |", "|---|---:|---|---|---:|---|---|---:|---:|---:|"]
    for n, s in rep["summary"].items():
        if (names and n not in names) or s["n_batches"] < min_n:
            continue
        r0, p1, fn = s["vs_r0"], s["vs_p1"], s["vs_final"]
        relaxed = ("通过" if s["relaxed_pass_vs_r0_and_p1"] else "**不通过**") + ("(n<8)" if s["n_batches"] < 8 else "")
        strict = "通过" if s["strict_pass_vs_r0_and_p1"] else "不通过"
        lines.append(f"| {n} | {s['n_batches']} | {f(r0['gcos_median'])} / {f(r0['gcos_min'])} | {f(p1['gcos_median'])} / {f(p1['gcos_min'])} | {r0['loss_rel_max']:.1e} | {relaxed} | {strict} | "
                     f"{f(fn['gcos_median'])} | {f(r0['pmin_min'], 3)} | {s['step_seconds_mean']:.1f} |")
    return "\n".join(lines) + "\n"


def dg(c):
    d = c.get("delayed_diag")
    if not d or not d.get("n_delayed"):
        return "-" if d is None else "n/a"
    return f"{int(d['n_over'])} / {int(d['n_delayed'])}，{d['max_ratio']:.2f}"


def gates_table(rep, title):
    lines = [f"#### {title}", "", "| 场景 | 候选 | EMA 偏差 vs R0（检查点 / 全 50 步） | EMA 偏差 vs FINAL（检查点 / 全 50 步） | 有限 | grad-norm 比值(对 R0) 范围 | delayed amax：超 2× 次数 / 延迟调用数，最大 cur/prev | 通过 |", "|---|---|---|---|---|---|---|---|"]
    for scen, run in rep["runs"].items():
        for n, c in run["cands"].items():
            gr = [v for v in c["vs_r0"]["grad_norm_ratios"].values() if v is not None]
            lines.append(f"| {scen} | {n} | {c['vs_r0']['curve300_max_rel']:.4f} / {c['max_rel_all_vs_r0']:.4f} | {c['vs_final']['curve300_max_rel']:.4f} / {c['max_rel_all_vs_final']:.4f} | "
                         f"{c['curve']['finite']} | {min(gr):.3f}–{max(gr):.3f} | {dg(c)} | {'通过' if c['pass'] else '**不通过**'} |")
        fr = run.get("FINAL")
        if fr is not None and "R0" in run:
            mr = max(abs(a - b) / max(abs(a), abs(b), 1e-8) for a, b in zip(run["R0"]["loss_ema"], fr["loss_ema"]))
            lines.append(f"| {scen} | (FINAL 对 R0 参考) | {mr:.4f} (全 50 步) | - | {fr['finite']} | - | - | - |")
    return "\n".join(lines) + "\n"


def build():
    """-> (tables {key: markdown}, results dict)."""
    tabs, res = {}, {"screen": {}, "finals": {}, "gateS": {}}
    for m in MODELS:
        scr = load(PROF / f"gateB_r20b_{m}_screen.json")
        fin = load(PROF / f"gateB_r20b_{m}_finals.json")
        if scr:
            tabs[f"screen_{m}"] = gateb_table(scr, f"{m}：筛选（全部候选；n 见表，同一进程、同一初始权重 / 批 / RngTape）")
            res["screen"][m] = {n: {k: s[k] for k in ("n_batches", "policy", "vs_r0", "vs_p1", "vs_final", "relaxed_pass_vs_r0_and_p1", "strict_pass_vs_r0_and_p1", "step_seconds_mean", "peak_gb_max")} for n, s in scr["summary"].items()}
        if fin:
            names = [n for n, s in fin["summary"].items() if s["n_batches"] >= 8 and n != "FINAL_again"]
            tabs[f"finals_{m}"] = gateb_table(fin, f"{m}：决赛 B 门（≥8 批）", names)
            res["finals"][m] = {n: fin["summary"][n] for n in names}
        arm = load(PROF / f"gateB_r20b_{m}_arms.json")
        if arm:
            names = [n for n, x in arm["summary"].items() if x["n_batches"] >= 8]
            tabs[f"arms_{m}"] = gateb_table(arm, f"{m}：对照臂（以 B4 为底，≥8 批，同一进程配对）", names)
            res.setdefault("arms", {})[m] = {n: arm["summary"][n] for n in names}
        gs = [load(p) for p in sorted(PROF.glob(f"gateS_r20b_{m}*.json"))]
        gs = [g for g in gs if g]
        if gs:
            merged = {"runs": {}}
            for g in gs:
                for scen, run in g["runs"].items():
                    tgt = merged["runs"].setdefault(scen, {"cands": {}})
                    tgt.update({k: v for k, v in run.items() if k != "cands"})
                    tgt["cands"].update(run["cands"])
            tabs[f"gateS_{m}"] = gates_table(merged, f"{m}：S 门（4 场景 × 50 步 × 2 批，全部 eager，amax=delayed）")
            res["gateS"][m] = {scen: {n: {"vs_r0_ckpt": c["vs_r0"]["curve300_max_rel"], "vs_r0_all50": c["max_rel_all_vs_r0"], "vs_final_ckpt": c["vs_final"]["curve300_max_rel"],
                                          "vs_final_all50": c["max_rel_all_vs_final"], "finite": c["curve"]["finite"], "pass": c["pass"], "policy": c["policy"],
                                          "grad_norm_ratios_vs_r0": c["vs_r0"]["grad_norm_ratios"]} for n, c in run["cands"].items()} for scen, run in merged["runs"].items()}
    return tabs, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="store_true")
    args = ap.parse_args()
    tabs, res = build()
    text = chr(10).join(tabs.values())
    (PROF / "tables_r20b.md").write_text(text, encoding="utf-8")
    print(text)
    if args.results:
        rp = ROUND / "results.json"
        d = json.loads(rp.read_text())
        d["r20b"] = {**d.get("r20b", {}), **res}
        rp.write_text(json.dumps(d, indent=1))
        print("results.json updated (r20b: screen / finals / gateS)")


if __name__ == "__main__":
    main()
