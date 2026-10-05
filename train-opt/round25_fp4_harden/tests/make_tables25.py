"""Markdown tables + results.json of round25 (reads profile/*.json; prints to stdout, writes profile/tables_r25.md and results.json).  Every section is skipped (null) when its data does not exist yet.

  python tests/make_tables25.py
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
import make_gateb25 as mgb  # noqa: E402

out: list[str] = []
RES: dict = {}


def p(s: str = "") -> None:
    out.append(s)
    print(s)


def files(pattern):
    return sorted(glob.glob(str(PROF / pattern)), key=lambda x: Path(x).stat().st_mtime)


def load(f):
    return json.loads(Path(f).read_text())



# ------------------------------------------------------------------ saturation study (task 1.1)
def sat_section():
    import analyze_sat as A
    import numpy as np

    fs = files("sat_study_*.json")
    if not fs:
        RES["saturation_study"] = None
        return
    rows, res = [], {}
    for f in fs:
        d = load(f)
        model = d["model"]
        runs, data = {}, {}
        for key, r in d["runs"].items():
            runs[key], data[key] = A.analyse_run(key, r)
        names = next(iter(data.values()))[2]["site_names"]
        act = np.zeros(len(names), bool)
        act[next(iter(data.values()))[2]["active_sites"]] = True
        p3 = [k for k in runs if runs[k]["stack"] == "P3"]
        p3b = [k for k in runs if runs[k]["stack"] == "P3B"]
        ent = {"model": model, "scenario": d["runs"][next(iter(d["runs"]))]["scen"], "batch_sum": d["batch_sum"], "n_batches": d.get("n_batches", 2),
               "P3B_dy_at8": [runs[k]["what_if"]["dy"]["8"] for k in p3b], "P3_dy_at8": [runs[k]["what_if"]["dy"]["8"] for k in p3], "calls": runs[p3[0]]["what_if"]["dy"]["calls"] if p3 else None,
               "P3_whatif_dy": {H: [runs[k]["what_if"]["dy"][H] for k in p3] for H in ("2", "4", "8", "16", "32", "64")}, "P3_x_at2": [runs[k]["what_if"]["x"]["2"] for k in p3],
               "monitor_matches_trace": all(o["monitor_matches_trace"] for o in runs.values())}
        if p3 and p3b:
            ent["step0_dy_amax_ratio_P3_over_P3B"] = A.step0_ratio(data[p3[0]], data[p3b[0]], names, act)
            e0, e1 = data[p3[0]][3], data[p3b[0]][3]
            s0 = {(s_, t) for s_, e in e0.items() for t in np.where(e.any(1))[0]}
            s1 = {(s_, t) for s_, e in e1.items() for t in np.where(e.any(1))[0]}
            ent["event_overlap_P3_P3B"] = {"n_P3": len(s0), "n_P3B": len(s1), "common": len(s0 & s1), "steps_P3": sorted({int(t) for _, t in s0}), "steps_P3B": sorted({int(t) for _, t in s1})}
        if len(p3) > 1:
            ent["step0_dy_amax_ratio_P3rep_over_P3"] = A.step0_ratio(data[p3[1]], data[p3[0]], names, act)
        wd = runs[p3[0]]["dy_events_by_block"] if p3 else {}
        ent["P3_dy_events_by_block"] = wd
        ent["P3_dy_events_by_linear"] = runs[p3[0]]["dy_events_by_linear"] if p3 else {}
        res[Path(f).stem] = ent
        rows.append((Path(f).stem, ent))
    RES["saturation_study"] = res
    p("### dY 饱和研究（同一进程、同样的 2 个真实 batch、同样的种子；每个进程自己抽一对 batch；P3B = 只做骨干，P3 = head + 骨干；dy@8 = 骨干 dY 在 headroom 8 下的超限次数 / 9200 次比较）\n")
    p("| 文件（一个进程 = 一对 batch） | 模型 | P3B 的 dy@8 | P3 的 dy@8 | P3 的 what-if（headroom 16 / 32 / 64） | 事件所在步 P3 / P3B | (site,step) 共有 / P3 / P3B | step-0 的 dY amax 比 P3 / P3B：中位（p5–p95） | 监控值 = 轨迹重算 |")
    p("|---|---|---|---|---|---|---|---|---|")
    for stem, e in rows:
        ov = e.get("event_overlap_P3_P3B", {})
        r0 = e.get("step0_dy_amax_ratio_P3_over_P3B", {})
        wi = e["P3_whatif_dy"]
        p(f"| {stem} | {e['model']} | {e['P3B_dy_at8']} | {e['P3_dy_at8']} | {wi['16']} / {wi['32']} / {wi['64']} | {ov.get('steps_P3')} / {ov.get('steps_P3B')} | {ov.get('common')} / {ov.get('n_P3')} / {ov.get('n_P3B')} | "
          f"{r0.get('median', float('nan')):.3f} ({r0.get('p5', float('nan')):.2f}–{r0.get('p95', float('nan')):.2f}) | {e['monitor_matches_trace']} |")
    p()

# ------------------------------------------------------------------ S sweep (task 1.2)
def gate_s_sweep(model="vitl16", mn="ViT-L"):
    fs = files(f"gateS_sweep_{model}_*.json")
    if not fs:
        RES.setdefault("gate_s_sweep", {})[model] = None
        return
    agg: dict = {}
    per_scen: dict = {}
    for f in fs:
        d = load(f)
        for scen, row in d["runs"].items():
            for st, r in row["stacks"].items():
                a = agg.setdefault(st, {"dy_exceed": 0, "dy_calls": 0, "dy_max_ratio": 0.0, "x_exceed": 0, "w_exceed": 0, "head_exceed": 0, "head_calls": 0, "ema_ckpt": 0.0, "ema_all": 0.0,
                                        "gn_lo": 9e9, "gn_hi": 0.0, "scen": [], "finite": True, "pass": True, "loss_last": {}})
                b = r["brief"]
                a["dy_exceed"] += b["dy_exceed"] or 0
                a["dy_calls"] += b["dy_calls"] or 0
                a["dy_max_ratio"] = max(a["dy_max_ratio"], b["dy_max_ratio"] or 0.0)
                a["x_exceed"] += b["x_exceed"] or 0
                a["w_exceed"] += b["w_exceed"] or 0
                a["head_exceed"] += b["head_exceed"] or 0
                a["head_calls"] += b["head_calls"] or 0
                a["ema_ckpt"] = max(a["ema_ckpt"], r["gate"]["curve300_max_rel"])
                a["ema_all"] = max(a["ema_all"], r["ema_max_rel_all_steps"])
                gn = list(r["gate"]["grad_norm_ratios"].values())
                a["gn_lo"] = min(a["gn_lo"], min(gn), r["grad_norm_ratio_all_steps"][0])
                a["gn_hi"] = max(a["gn_hi"], max(gn), r["grad_norm_ratio_all_steps"][1])
                a["scen"].append(scen)
                a["finite"] &= bool(r["gate"]["finite"])
                a["pass"] &= bool(r["gate"]["pass"])
                a["loss_last"][scen] = r["cand"]["loss"][-1]
                per_scen.setdefault(scen, {})[st] = {"dy": f"{b['dy_exceed']}/{b['dy_calls']}", "max_ratio": b["dy_max_ratio"], "ema": r["gate"]["curve300_max_rel"], "ema_all": r["ema_max_rel_all_steps"],
                                                     "head": f"{b['head_exceed']}/{b['head_calls']}", "x": b["x_exceed"]}
    RES.setdefault("gate_s_sweep", {})[model] = {"aggregate": agg, "per_scenario": per_scen, "files": [Path(f).name for f in fs]}
    p(f"### 门 S 扫描（{mn}，StepGraphRunner 下对 R0 eager；同一进程同样 2 批；4 场景 × 50 步；dY headroom = 骨干 `dyhr`，head 保持 8）\n")
    p("| 设置 | 场景数 | 骨干 dY 饱和（exceed / calls） | 饱和率 | 最大 cur/prev | X 超限 | head 超限 | EMA 偏差（检查点最大 / 全部步最大） | grad-norm 比值范围（全部步） | 门 | 有限 |")
    p("|---|---:|---|---:|---:|---:|---|---|---|---|---|")
    for st, a in agg.items():
        p(f"| {st} | {len(a['scen'])} | {a['dy_exceed']} / {a['dy_calls']} | {100 * a['dy_exceed'] / max(1, a['dy_calls']):.2f}% | {a['dy_max_ratio']:.1f} | {a['x_exceed']} | {a['head_exceed']} / {a['head_calls']} | "
          f"{100 * a['ema_ckpt']:.2f}% / {100 * a['ema_all']:.2f}% | {a['gn_lo']:.3f}–{a['gn_hi']:.3f} | {'过' if a['pass'] else '不过'} | {a['finite']} |")
    p()
    scens = list(per_scen)
    stacks = list(agg)
    p("逐场景（dY 饱和 exceed / 9200；EMA 偏差检查点最大）：\n")
    p("| 场景 | " + " | ".join(stacks) + " |")
    p("|---|" + "---|" * len(stacks))
    for sc in scens:
        p(f"| {sc} | " + " | ".join((f"{per_scen[sc][s]['dy']} (max {per_scen[sc][s]['max_ratio']:.1f}), EMA {100 * per_scen[sc][s]['ema']:.2f}%" if s in per_scen[sc] else "-") for s in stacks) + " |")
    p()


# ------------------------------------------------------------------ benches
def benches():
    for model, mn in (("vitl16", "ViT-L"), ("vitb16", "ViT-B")):
        fs = files(f"bench_{model}_*_graph_all*.json")
        if not fs:
            RES.setdefault("bench", {})[model] = None
            continue
        by: dict = {}
        order = []
        for f in fs:
            d = load(f)
            s = d["summary"]
            st = s["stack"]
            by.setdefault(st, []).append({"w": s["weighted_img_s"], "mid_ms": s["step_ms_mean"]["mid"], "early_ms": s["step_ms_mean"]["early"], "late_ms": s["step_ms_mean"]["late"],
                                          "peak_gb": s["peak_reserved_gb"], "file": Path(f).name})
            order.append((st, s["weighted_img_s"]))
        mean = {st: statistics.fmean(x["w"] for x in v) for st, v in by.items()}
        res = {"order": order, "stacks": {st: {"runs": [x["w"] for x in v], "mean_img_s": mean[st], "mid_ms_mean": statistics.fmean(x["mid_ms"] for x in v), "peak_reserved_gb": max(x["peak_gb"] for x in v),
                                              "vs_FINAL": (mean[st] / mean["FINAL"]) if "FINAL" in mean else None, "vs_P3": (mean[st] / mean["P3"]) if "P3" in mean else None} for st, v in by.items()}}
        RES.setdefault("bench", {})[model] = res
        p(f"### 整步 bench（{mn}，StepGraphRunner，三相 0.2/0.6/0.2 加权，warmup 20，50×3；每个进程一个栈，A-B-A-B 交替）\n")
        p("| 栈 | 次数 | 每次 weighted img/s | 均值 img/s | 均值 mid 步时 | 对 FINAL | 对 P3 | 峰值 reserved |")
        p("|---|---:|---|---:|---:|---:|---:|---:|")
        for st, v in by.items():
            r = res["stacks"][st]
            p(f"| {st} | {len(v)} | {' / '.join('%.2f' % x['w'] for x in v)} | {r['mean_img_s']:.2f} | {r['mid_ms_mean']:.2f} ms | "
              f"{'-' if r['vs_FINAL'] is None else 'x%.3f' % r['vs_FINAL']} | {'-' if r['vs_P3'] is None else 'x%.3f' % r['vs_P3']} | {r['peak_reserved_gb']:.2f} GB |")
        p()


# ------------------------------------------------------------------ gate B
def gate_b():
    for tag, model, mn, order in (("hr", "vitl16", "ViT-L", ["FINAL", "P3", "P3_hr16", "P3_hr32", "P3_hr64", "P3_exact", "P3_H2", "P3_H2f", "FINAL_again"]),
                                  ("h2", "vitb16", "ViT-B", ["FINAL", "P3", "P3_H2", "P3_H2f", "P3_hr32", "FINAL_again"])):
        if not files(f"gateB_{tag}_{model}_draw*.json"):
            RES.setdefault("gate_b", {})[model] = None
            continue
        t = mgb.table(tag, model)
        RES.setdefault("gate_b", {})[model] = t
        p(f"### 门 B（{mn}；同进程 R0 / P1 / FINAL / 候选，同权重同 batch 同 RngTape；n = {t['n_batches']}，文件 {', '.join(t['files'])}）\n")
        p(mgb.md(mn, t, order))
        p()
        p("P3 最差的 4 批（对 R0 的全局梯度余弦，各候选在同一批上）：" + json.dumps(t.get("worst_P3_batches_vs_r0")))
        p()


# ------------------------------------------------------------------ task 3
def task3():
    res: dict = {}
    f = files("graph_equiv_vitl16_P3_q8.json")
    if f:
        d = load(f[-1])
        res["graph_equiv"] = {sc: {"update_cos_graph_vs_eager": r["graph_vs_eager"]["update_cos"], "update_cos_eager_vs_eager": r["eager_vs_eager"]["update_cos"], "loss_rel_graph_vs_eager": r["loss_rel_max_graph_vs_eager"],
                                   "loss_rel_eager_vs_eager": r["loss_rel_max_eager_vs_eager"], "first_step_loss_bitwise": r["loss_bitwise_first_step"], "fp4_state_advance_ok": r["fp4_state_advance"]["ok"]}
                              for sc, r in d["scenarios"].items()}
        p("### graph_equiv（ViT-L P3_q8，8 步）\n")
        p("| 场景 | 更新余弦 graph-vs-eager | eager-vs-eager（噪声底） | loss 相对差 graph-vs-eager | eager-vs-eager | 第一步 loss 逐位 | 两个 state 各前进 8 步 |")
        p("|---|---:|---:|---:|---:|---|---|")
        for sc, r in res["graph_equiv"].items():
            p(f"| {sc} | {r['update_cos_graph_vs_eager']:.6f} | {r['update_cos_eager_vs_eager']:.6f} | {r['loss_rel_graph_vs_eager']:.1e} | {r['loss_rel_eager_vs_eager']:.1e} | {r['first_step_loss_bitwise']} | {r['fp4_state_advance_ok']} |")
        p()
    f = files("export_vitl16_P3_q8.json")
    if f:
        d = load(f[-1])
        res["export"] = {k: d[k] for k in list(d)[:40]}
        p(f"export / resume（check_export）：{json.dumps({k: d[k] for k in d if k in ('pass', 'all_pass', 'verdict')})}\n")
    f = files("run_final_vitl16_P3_q8_export.json")
    if f:
        d = load(f[-1])
        res["run_final"] = {k: d.get(k) for k in ("steps", "peak_reserved_gb", "fp4_state_steps_equal_train_steps", "final", "opt_step_chain", "fp4_state")}
    f = files("e2e_vitl16_P3_q8.json")
    if f:
        d = load(f[-1])
        r = d["runs"][0]
        res["e2e"] = {"gpu_only_img_s": d["gpu_only_early"]["img_s"], "steady_img_s": r.get("img_s_steady"), "steady_vs_gpu_only": r.get("steady_vs_gpu_only"), "wait_ms": {k: r.get(k) for k in r if "wait" in k}}
        p(f"e2e：{json.dumps(res['e2e'], default=str)}\n")
    RES["task3_q8_vitl"] = res or None


if __name__ == "__main__":
    sat_section()
    gate_s_sweep()
    benches()
    gate_b()
    task3()
    (PROF / "tables_r25.md").write_text("\n".join(out) + "\n")
    # results.json keeps what other scripts (the saturation analysis) stored under 'saturation_study'
    rp = ROUND / "results.json"
    old = json.loads(rp.read_text()) if rp.exists() else {}
    old.update(RES)
    rp.write_text(json.dumps(old, indent=1, default=str, allow_nan=False if False else True))
    print("wrote", rp)
