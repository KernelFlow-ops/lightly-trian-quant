"""Markdown tables of round23's gates for NOTES.md (reads profile/*.json; prints to stdout and writes profile/tables_r23.md).

  python tests/make_tables.py
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

MODELS = (("vitl16", "ViT-L"), ("vitb16", "ViT-B"))
out: list[str] = []


def p(s: str = "") -> None:
    out.append(s)
    print(s)


def files(pattern):
    return sorted(glob.glob(str(PROF / pattern)), key=lambda x: Path(x).stat().st_mtime)


def load(f):
    return json.loads(Path(f).read_text())


def graph_equiv() -> None:
    p("### graph_equiv（P3，8 步，graph 回放对 eager；两个 FP4 state 在预热后快照并在每次运行前原地恢复）\n")
    p("| 模型 | 场景 | 更新余弦 graph-vs-eager | eager-vs-eager（噪声底） | loss 相对差 graph-vs-eager | eager-vs-eager | 第一步 loss 逐位 | 两个 state 各前进 8 步 |")
    p("|---|---|---:|---:|---:|---:|---|---|")
    for m, mn in MODELS:
        fs = files(f"graph_equiv_{m}_P3.json")
        if not fs:
            p(f"| {mn} | not measured | | | | | | |")
            continue
        d = load(fs[-1])
        for scen, r in d["scenarios"].items():
            p(f"| {mn} | {scen} | {r['graph_vs_eager']['update_cos']:.6f} | {r['eager_vs_eager']['update_cos']:.6f} | {r['loss_rel_max_graph_vs_eager']:.1e} | {r['loss_rel_max_eager_vs_eager']:.1e} | "
              f"{r['loss_bitwise_first_step']} | {r['fp4_state_advance']['ok']} |")
    p()


def gate_s() -> None:
    p("### 门 S（StepGraphRunner 下 P3 graph 对 R0 eager；4 场景 × 50 步 × 2 批；EMA 偏差 = 4 个检查点上的最大相对偏差，门 5%）\n")
    p("| 模型 | 场景 | EMA 偏差（检查点）对 R0 | EMA 偏差（所有 50 步）对 R0 | 对 FINAL graph（检查点） | 门 | 有限 | grad-norm 比值范围（对 R0） | head 饱和（exceed / calls，全部 site） | 骨干 dY 饱和（exceed / calls，最大 cur/prev） |")
    p("|---|---|---:|---:|---:|---|---|---|---|---|")
    for m, mn in MODELS:
        for f in files(f"gateS_{m}_P3_graph_vs_R0*.json"):
            d = load(f)
            for scen, r in d["runs"].items():
                g, c = r["gate"], r["cand"]
                ref = r["ref"]
                ema_all = max(abs(a - b) / abs(b) for a, b in zip(c["loss_ema"], ref["loss_ema"]))
                gn = list(g["grad_norm_ratios"].values())
                vf = (r.get("gate_vs_final") or {}).get("curve300_max_rel")
                fp = c.get("fp4", {})
                hd, bb = fp.get("head", {}), fp.get("backbone", {})
                hz = f"{hd.get('total_exceed')} / {hd.get('total_calls')}" if hd else "n/a"
                bz = "n/a"
                if bb:
                    dy = (bb.get("per_kind") or {}).get("dy", {})
                    bz = f"{dy.get('exceed')} / {dy.get('calls')} (max ratio {dy.get('max_ratio', 0):.1f})"
                p(f"| {mn} | {scen} | {100 * g['curve300_max_rel']:.2f}% | {100 * ema_all:.2f}% | {'-' if vf is None else '%.2f%%' % (100 * vf)} | {'过' if g['pass'] else '不过'} | {g['finite']} | "
                  f"{min(gn):.3f}–{max(gn):.3f} | {hz} | {bz} |")
    p()


def gate_b() -> None:
    p("### 门 B（同进程内 R0 / P1 / FINAL / P3H / P3B / P3，同权重同 batch 同 RngTape；两次独立抽批，每次 8 批；放宽门：对 R0 和对 P1 都要中位 ≥0.98、最小 ≥0.97、loss_rel ≤2e-2；严格门：中位 ≥0.99、最小 ≥0.98）\n")
    for m, mn in MODELS:
        draws = {Path(f).stem.split("_")[-1]: load(f) for f in files(f"gateB_p3_{m}_draw*.json")}
        if not draws:
            p(f"**{mn}**：not measured\n")
            continue
        merged = {"seeds": {}, "cands": {}}
        for name, d in draws.items():
            merged["cands"] = d["cands"]
            for sd, s in d["seeds"].items():
                merged["seeds"][f"{name}:{sd}"] = s
        names = list(merged["cands"])
        p(f"**{mn}**（draw 数 {len(draws)}，批数 " + " + ".join(f"{k}: {len(v['seeds'])}" for k, v in draws.items()) + "）\n")
        p("| 候选 | 抽批 | n | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 / 最小 | 最小值所在批 |")
        p("|---|---|---:|---|---|---:|---|---|---|---|")
        sets = {k: {"seeds": {sd: s for sd, s in merged["seeds"].items() if sd.startswith(k + ":")}, "cands": merged["cands"]} for k in draws}
        sets["合并"] = merged
        for name in names:
            if name == "FINAL_again":
                continue
            for k, rep in sets.items():
                sm = summarize(rep, [name]).get(name)
                if not sm:
                    continue
                a, b, f = sm["vs_r0"], sm["vs_p1"], sm["vs_final"]
                p(f"| {name} | {k} | {sm['n_batches']} | {a['gcos_median']:.4f} / {a['gcos_min']:.4f} | {b['gcos_median']:.4f} / {b['gcos_min']:.4f} | {max(a['loss_rel_max'], b['loss_rel_max']):.1e} | "
                  f"{'过' if sm['relaxed_pass_vs_r0_and_p1'] else '**不过**'} | {'过' if sm['strict_pass_vs_r0_and_p1'] else '不过'} | {f['gcos_median']:.4f} / {f['gcos_min']:.4f} | {a['gcos_min_seed']} |")
        p()
        # per seed table of the main candidates
        main = [n for n in ("FINAL", "P3H", "P3B", "P3") if n in names]
        p(f"{mn} 逐批对 R0 的全局梯度余弦（FINAL_again 与 FINAL 的差 = 同栈重复运行噪声）：\n")
        p("| 批 | P1 | " + " | ".join(main) + " | FINAL_again |")
        p("|---|---:|" + "---:|" * (len(main) + 1))
        for sd, s in merged["seeds"].items():
            c = s["cands"]
            if not all(n in c and "vs_r0" in c[n] for n in main):
                continue
            p(f"| {sd} | {s['ref_p1_vs_r0']['grad_cos_global']:.4f} | " + " | ".join(f"{c[n]['vs_r0']['grad_cos_global']:.4f}" for n in main) + f" | {c['FINAL_again']['vs_r0']['grad_cos_global']:.4f} |"
              if 'FINAL_again' in c and 'vs_r0' in c['FINAL_again'] else f"| {sd} | | | | | | |")
        p()
        cons = [summarize(merged, ["P3"]).get("P3", {}).get(k) for k in ("vs_P3H", "vs_P3B")]
        if all(cons):
            p(f"组合一致性（P3 对 P3H / P3B 的梯度，逐批取最小 / 中位）：vs P3H 全局 {cons[0]['global_cos_min']:.4f}，骨干 blocks {cons[0]['blocks_cos_min']:.4f} / {cons[0]['blocks_cos_median']:.4f}，head ConvT {cons[0]['head_convt_cos_min']:.4f} / {cons[0]['head_convt_cos_median']:.4f}；"
              f"vs P3B 全局 {cons[1]['global_cos_min']:.4f}，blocks {cons[1]['blocks_cos_min']:.4f} / {cons[1]['blocks_cos_median']:.4f}，head ConvT {cons[1]['head_convt_cos_min']:.4f} / {cons[1]['head_convt_cos_median']:.4f}\n")
        # seed 13 (draw 2)
        for name in ("P3", "P3B", "FINAL", "P3H"):
            s13 = merged["seeds"].get("draw2:13", {}).get("cands", {}).get(name)
            if s13 and "vs_r0" in s13:
                p(f"- {mn} seed 13（draw2）{name}：对 R0 {s13['vs_r0']['grad_cos_global']:.4f}，对 P1 {s13['vs_p1']['grad_cos_global']:.4f}，对 FINAL {s13['vs_final']['grad_cos_global']:.4f}")
        p()


def gate_b_extra() -> None:
    """The optional-5a queue also ran P3 (control arm) and FINAL on 16 further batches per model (two more independent draws, same-process pairing with R0 / P1): reported as extra
    gate-B evidence for P3 and pooled with the three main draws."""
    p("### 门 B 补充：5a 队列里的 P3 对照臂（另外 2 次独立抽批 × 8 批，同进程配对 R0 / P1 / FINAL；候选 P3 与 P3_q8）与 5 次抽批合并（P3 / FINAL，n = 40）\n")
    p("| 模型 | 候选 | 抽批 | n | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 / 最小 | 最小值所在批 |")
    p("|---|---|---|---:|---|---|---:|---|---|---|---|")

    def pool(ds):
        return {"seeds": {f"{k}:{sd}": s for k, d in ds.items() for sd, s in d["seeds"].items()}, "cands": next(iter(ds.values()))["cands"]}

    for m, mn in MODELS:
        qd = {"q" + Path(f).stem.split("_")[-1]: load(f) for f in files(f"gateB_q8_{m}_draw*.json")}
        md = {Path(f).stem.split("_")[-1]: load(f) for f in files(f"gateB_p3_{m}_draw*.json")}
        if not qd:
            continue
        for label, rep, names in (("q8 队列 2 次抽批", pool(qd), ["FINAL", "P3", "P3_q8"]), ("5 次抽批合并", pool({**md, **qd}), ["FINAL", "P3"])):
            for name in names:
                sm = summarize(rep, [name]).get(name)
                if not sm:
                    continue
                a, b, f = sm["vs_r0"], sm["vs_p1"], sm["vs_final"]
                p(f"| {mn} | {name} | {label} | {sm['n_batches']} | {a['gcos_median']:.4f} / {a['gcos_min']:.4f} | {b['gcos_median']:.4f} / {b['gcos_min']:.4f} | {max(a['loss_rel_max'], b['loss_rel_max']):.1e} | "
                  f"{'过' if sm['relaxed_pass_vs_r0_and_p1'] else '**不过**'} | {'过' if sm['strict_pass_vs_r0_and_p1'] else '不过'} | {f['gcos_median']:.4f} / {f['gcos_min']:.4f} | {a['gcos_min_seed']} |")
    p()
    p("MXFP8 离群批（FINAL 自己对 R0 的全局梯度余弦 < 0.985 的批，全部 5 次抽批；P1 = bf16 对 R0）：\n")
    p("| 模型 | 批 | P1 | FINAL | P3H | P3B | P3 | P3_q8 |")
    p("|---|---|---:|---:|---:|---:|---:|---:|")
    for m, mn in MODELS:
        for f in files(f"gateB_p3_{m}_draw*.json") + files(f"gateB_q8_{m}_draw*.json"):
            tag = Path(f).stem.replace("gateB_", "").replace(f"_{m}", "")
            for sd, s in load(f)["seeds"].items():
                c = s["cands"]
                if "FINAL" in c and "vs_r0" in c["FINAL"] and c["FINAL"]["vs_r0"]["grad_cos_global"] < 0.985:
                    def g(n):
                        return ("%.4f" % c[n]["vs_r0"]["grad_cos_global"]) if n in c and "vs_r0" in c[n] else "-"
                    p(f"| {mn} | {tag}:{sd} | {s['ref_p1_vs_r0']['grad_cos_global']:.4f} | {g('FINAL')} | {g('P3H')} | {g('P3B')} | {g('P3')} | {g('P3_q8')} |")
    p()


def bench_table() -> None:
    p("### 整步速度表（graph，三相加权；每行 = 该栈在本轮所有 bench 进程的均值；倍数对同场次 FINAL 均值和同机 R0 eager）\n")
    p("| 模型 | 栈 | 次数 | 每次 weighted img/s | 均值 img/s | 均值 mid 步时 | early / late 步时 | 对 FINAL | 对 R0 | 峰值 reserved |")
    p("|---|---|---:|---|---:|---:|---|---:|---:|---:|")
    names = {"R0": "R0（eager）", "FINAL": "FINAL", "P3H": "P3H（head）", "P3B": "P3B（骨干）", "P3": "**P3（head + 骨干）**", "P3_RN": "P3_RN（RN 对照）", "P3_q8": "P3_q8（5a：P3 + qkv 前向 MXFP8，只记录）"}
    for m, mn in MODELS:
        rows = {}
        for f in files(f"bench_{m}_*_all*.json"):
            s = load(f)["summary"]
            if s.get("q2"):
                continue
            rows.setdefault(s["stack"], []).append(s)
        fin = statistics.fmean(s["weighted_img_s"] for s in rows["FINAL"]) if "FINAL" in rows else None
        r0 = statistics.fmean(s["weighted_img_s"] for s in rows["R0"]) if "R0" in rows else None
        for st in ("R0", "FINAL", "P3H", "P3B", "P3", "P3_RN", "P3_q8"):
            if st not in rows:
                continue
            rr = rows[st]
            ips = [s["weighted_img_s"] for s in rr]
            mean = statistics.fmean(ips)
            mid = statistics.fmean(s["step_ms_mean"]["mid"] for s in rr)
            ea = statistics.fmean(s["step_ms_mean"]["early"] for s in rr)
            la = statistics.fmean(s["step_ms_mean"]["late"] for s in rr)
            pk = max(s.get("peak_reserved_gb") or 0 for s in rr)
            p(f"| {mn} | {names[st]} | {len(rr)} | {' / '.join('%.2f' % x for x in ips)} | {mean:.2f} | {mid:.2f} ms | {ea:.2f} / {la:.2f} | {'-' if not fin else '×%.3f' % (mean / fin)} | "
              f"{'-' if not r0 else '%.2f' % (mean / r0)} | {pk:.2f} GB |")
    p()


def main() -> None:
    bench_table()
    graph_equiv()
    gate_s()
    gate_b()
    gate_b_extra()
    (PROF / "tables_r23.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
