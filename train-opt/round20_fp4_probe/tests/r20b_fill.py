"""Strings for the placeholders of r20b_notes.tpl.md and the decision block of results.json["r20b"].  Dynamic parts read the measured json files."""
from __future__ import annotations

import json
from pathlib import Path

PROF = Path(__file__).resolve().parents[1] / "profile"
HEAD = "head.f=mx8,head.dw=nv4+sr"
BB = ",all.d=nv4+sr,all.w=nv4+sr+rht"
POLICY = {
    "vitl16": HEAD + BB + ",nofirst,amax=delayed",   # B4
    "vitb16": HEAD + BB + ",amax=delayed",           # B3
}
HEAD_ONLY = HEAD + ",amax=delayed"                    # H3
AGGRESSIVE = HEAD + BB + ",amax=delayed"              # B3 (the most aggressive passer, both models)

NOT_MEASURED_LIST = [
    "整步速度（graph 下与 FINAL 同场次）：R20b 只做精度门，没有真 kernel",
    "graph_equiv / export / resume：R20b 不发布栈、不改参数",
    "qkv.f=mx8 / fc.f=nv4 的 ≥8 批 B 门（只有筛选阶段 n=3 的记录）",
    "ViT-L 的 amax=exact 对照 S 门；ViT-L S 门的 delayed 诊断计数（诊断代码在 ViT-L S 门跑完之后才加入）",
    "S 门里的 mx4 / RN / w1d / norht 对照臂（它们只做了 B 门）",
]


def _load(name):
    p = PROF / name
    return json.loads(p.read_text()) if p.exists() else None


def _diag_rows():
    rows = []
    for m in ("vitl16", "vitb16"):
        for p in sorted(PROF.glob(f"gateS_r20b_{m}*.json")):
            d = json.loads(p.read_text())
            for scen, run in d["runs"].items():
                for n, c in run["cands"].items():
                    dg = c.get("delayed_diag")
                    if dg is not None:
                        rows.append((m, scen, n, c["policy"], dg, c["vs_r0"]["curve300_max_rel"], c["max_rel_all_vs_r0"], c["curve"]["finite"], c["vs_r0"]["grad_norm_ratios"]))
    return rows


def _arms_sentence(res):
    out = []
    for m in ("vitl16", "vitb16"):
        a = res.get("arms", {}).get(m)
        if not a:
            out.append(f"{m}：对照臂 ≥8 批：not measured")
            continue
        def f(n):
            s = a.get(n)
            return "n/a" if s is None else f"{s['vs_r0']['gcos_median']:.4f}/{s['vs_r0']['gcos_min']:.4f}（放宽门{'通过' if s['relaxed_pass_vs_r0_and_p1'] else '不通过'}）"
        out.append(f"{m}（对 R0 中位/最小）：B4 {f('B4')}；mx4 {f('B4+mx4')}；RN {f('B4+rn')}；w1d {f('B4+w1d')}；不用 rht {f('B4+norht')}")
    return "；".join(out)


def _screen(m):
    d = _load(f"gateB_r20b_{m}_screen.json")
    return None if d is None else d["summary"]


def _cell(s, key):
    x = s.get(key)
    if x is None:
        return "n/a"
    r = x["vs_r0"]
    return f"{r['gcos_median']:.4f}/{r['gcos_min']:.4f}" + ("" if x["relaxed_pass_vs_r0_and_p1"] else "(不过)")


def _item3(res):
    nl = chr(10)
    lines = []
    for m in ("vitl16", "vitb16"):
        s = _screen(m)
        if not s:
            continue
        n = s["B4"]["n_batches"]
        for base in ("B3", "B4", "B5"):
            lines.append(f"  - {m} 筛选 n={n}，底 {base}（对 R0 中位/最小）：本身 {_cell(s, base)}；mx4 {_cell(s, base + '+mx4')}；RN {_cell(s, base + '+rn')}；w1d {_cell(s, base + '+w1d')}；不用 rht {_cell(s, base + '+norht')}；fc.f=nv4 {_cell(s, base + '+fcF4')}；qkv.f=mx8 {_cell(s, base + '+qkvF8')}")
    head = "**3. 格式与对照臂**（§4.4 的 ≥8 批 B4 底对照臂 + 下面的 3 批筛选明细）：" + _arms_sentence(res) + "。" + nl + nl.join(lines) + nl
    body = (
        "- **nv4 对 mx4**：ViT-B 上 mx4 低 0.004–0.005（仍过放宽门）；ViT-L 上 mx4 在 B3 底（block 0 用 FP4）**不过放宽门**（0.9796 < 0.98），在 nofirst / prefix 底只低 0.0001–0.0011（n=8 的 B4 底：0.9932 → 0.9921，仍过放宽门）。mx4 的损失集中在 block 0 的 massive activation 上。按 R20a 估计 mx4 整步只快 0.5 / 0.4 ms（amax 由生产者给出时；delayed amax 下 nv4 没有额外的 amax 遍历），低于 2% 发布线。11.6 的字面规则是“mx4 也过门并且整步更快才改用 mx4”：**按字面规则 mx4 有资格（B4 底 n=8 两个模型都过放宽门：ViT-L 0.9921、ViT-B 0.9874），但收益 ≈0.5%、精度更低，建议保持 nv4**，由协调者定。" + nl
        + "- **SR 对 RN**：两者都过；单步 B 门里 RN 反而略高 0.0000–0.0006（SR 的噪声是无偏的，单步余弦看不出它的长期好处；S 门只测了 SR）→ 按 11.6 选 SR。" + nl
        + "- **w2d 对 w1d**：差在噪声内（ViT-B w1d +0.0009–0.0014（n=8 的 B4 底 +0.0009）；ViT-L B3 底 −0.0013，B4 / B5 底 +0.0002）→ 保持 w2d（W 与 Wᵀ 数值一致，量化器一次出两份）。" + nl
        + "- **wgrad 的 RHT**：ViT-L B3 底去掉 rht 反而高 0.0025（0.9899 → 0.9924，RHT 在 block 0 的 outlier token 上有害，R20a 已见），B4 / B5 底 ±0.0001；ViT-B ±0.0003。B 门里 **RHT 没有可见收益**（§4.4 的 ≥8 批 B4 底对照臂：去掉 rht 的差别 ≤0.0002），有差别的地方是它在有害 → R21 的量化器可以先不做 RHT（更简单）；不带 rht 的等价 POLICY 见 results.json 的 `POLICY_no_rht_variant`。" + nl
        + "- **仅记录项**：`fc.f=nv4` 在 6 个底上全部**不过**放宽门（对 R0 中位 0.963–0.980，最小 0.937–0.965，ViT-B 的 loss_rel 到 1.2e-2）；`qkv.f=mx8` 在 5/6 个底上过（ViT-B 三个底 0.987–0.989；ViT-L 的 B3 底 0.9753 不过，B4 / B5 底 0.989 / 0.992），只有 n=3，信号不稳；两项都按 R20a 规则不发布。" + nl
    )
    return head + body


def _f(res, m, n):
    x = res.get("finals", {}).get(m, {}).get(n)
    if x is None:
        return "not measured"
    r = x["vs_r0"]
    return f"{r['gcos_median']:.4f} / {r['gcos_min']:.4f}"


def _g(res, m, n, key="gcos_median", tag="vs_r0"):
    x = res.get("finals", {}).get(m, {}).get(n)
    return None if x is None else x[tag][key]


def _headline(res):
    parts = []
    for m in ("vitl16", "vitb16"):
        fin = res.get("finals", {}).get(m, {})
        names = [n for n in ("FINAL", "H1", "H2", "H3", "B3", "B4", "B5") if n in fin]
        if not names:
            parts.append(f"{m}：决赛 not measured")
            continue
        rel = [n for n in names if fin[n]["relaxed_pass_vs_r0_and_p1"]]
        stc = [n for n in names if fin[n]["strict_pass_vs_r0_and_p1"]]
        n8 = min(fin[n]["n_batches"] for n in names)
        parts.append(f"{m}（n={n8}）放宽门通过：{'/'.join(rel)}；严格门通过：{'/'.join(stc)}（严格门只记录）")
    return ("；".join(parts) + "。S 门：ViT-L（B3 / H3 / B4 × 4 场景）和 ViT-B（B3 / H3 × 4 场景，加上表中已有的 B4 / exact-amax 对照）全部通过，EMA 偏差对 R0 最大 0.74%、对 FINAL 最大 0.85%（门 5%），无 NaN / Inf，delayed amax 下无异常。"
            "仅记录项（筛选 n=3）：`fc.f=nv4` 在 6 个底上全部不过放宽门；`qkv.f=mx8` 在 5/6 个底上过、ViT-L 的 B3 底不过，信号不稳。")


def _spread(res):
    out = []
    for m in ("vitl16", "vitb16"):
        v = [_g(res, m, n) for n in ("B3", "B4", "B5")]
        if None in v:
            continue
        out.append(f"{m}：B3 / B4 / B5 中位数 {v[0]:.4f} / {v[1]:.4f} / {v[2]:.4f}")
    return "；".join(out) if out else "not measured"


def _batch_noise(res):
    vals = []
    mp = _load("gateB_r20b_vitb16_finals_mixedproc.json")
    if mp and "B4" in mp["summary"] and mp["summary"]["B4"]["n_batches"] >= 8:
        vals.append(("旧混合进程", mp["summary"]["B4"]["vs_r0"]["gcos_median"]))
    x = res.get("finals", {}).get("vitb16", {}).get("B4")
    if x:
        vals.append(("决赛单进程", x["vs_r0"]["gcos_median"]))
    a = res.get("arms", {}).get("vitb16", {}).get("B4")
    if a:
        vals.append(("对照臂单进程", a["vs_r0"]["gcos_median"]))
    return "，".join(f"{k} {v:.4f}" for k, v in vals) if vals else "not measured"


def _vitb_rows():
    rows = {}
    for p in sorted(PROF.glob("gateS_r20b_vitb16*.json")):
        d = json.loads(p.read_text())
        for scen, run in d["runs"].items():
            for n, c in run["cands"].items():
                rows[(scen, n)] = c
    return rows


def _delayed_conclusion(res):
    nl = chr(10)
    rows = _vitb_rows()
    pairs = []
    for scen in ("schedule", "compressed", "fixed_mid", "fixed_late"):
        for dn, en in (("B4", "B4x"),):  # same process, same batches; B3 (earlier process) is not paired with B3x
            if (scen, dn) in rows and (scen, en) in rows:
                pairs.append(f"{scen} {dn} delayed {rows[(scen, dn)]['vs_r0']['curve300_max_rel']:.4f} / {en} exact {rows[(scen, en)]['vs_r0']['curve300_max_rel']:.4f}")
    cmp_txt = "；".join(pairs) if pairs else "not measured"
    tot_over = tot_del = 0
    mx = 0.0
    fr = []
    kinds = {}
    full = []
    for (scen, n), c in rows.items():
        dg = c.get("delayed_diag") or {}
        if not dg.get("n_delayed"):
            continue
        tot_over += dg["n_over"]
        tot_del += dg["n_delayed"]
        mx = max(mx, dg["max_ratio"])
        fr.append(dg["n_over"] / dg["n_delayed"])
        if n == "B4":
            rec = [(site, r) for site, r in dg.get("worst_sites", []) if r > 2.0]
            if len(rec) == dg.get("n_sites_ratio_gt_2", 0):  # every >2 site is in the stored list
                full.append(f"{scen}（{len(rec)} 个）")
                for site, r in rec:
                    k = site.split(".")[-1]
                    kinds[k] = kinds.get(k, 0) + 1
    kinds_txt = "，".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) or "not measured"
    only_grad = bool(kinds) and set(kinds) <= {"dr", "dc"}
    frac = f"{100 * tot_over / tot_del:.2f}%" if tot_del else "n/a"
    rng = f"{100 * min(fr):.1f}%–{100 * max(fr):.1f}%" if fr else "n/a"
    return (
        f"- **delayed 对 exact（ViT-B，对 R0 的 EMA 偏差，检查点最大）**：{cmp_txt}。" + nl
        + f"- **headroom 诊断（ViT-B，B4，4 个场景合计，每个场景 50 步，208 个调用点）**：延迟调用 {int(tot_del)} 次，当前 amax 超过 2× 上一次 amax（即该张量里大于 6·448·ts 的元素被饱和裁剪）的有 {int(tot_over)} 次（{frac}；按场景 {rng}），cur/prev 最大 {mx:.2f}。比值 >2 的调用点按类型（dr / dc = dY 的行 / 转置副本，xr / xc = X 的行 / 转置副本，W* = 权重），只统计保存了全部 >2 调用点的场景 {'、'.join(full) or 'none'}：{kinds_txt}。" + ("**这些场景里超过 2× 的全部是梯度 dY 的副本，X 和权重一次也没有超过。**" if only_grad else "") + nl
        + "- 结论：在 S 门的 50 步里 **delayed 与 exact 没有可见差别**（EMA 偏差相同到 ±0.0001，无 NaN / Inf，grad-norm 比值正常），所以 delayed amax 在这个门上是安全的，B 门（单步）本来就等同于 exact。" + nl
        + f"- **但 2× headroom 对梯度 dY 不够**：{frac} 的延迟调用当前 amax 超过 2× 上一次，最大 {mx:.0f}×，这些张量里最大的几个元素会被饱和裁剪；S 门（2 批 × 50 步）没有显示这个裁剪的后果，但它不能代表长训练里 lr 大步更新或梯度突发的情况。**建议 R21 / R22**：dY 的 amax 由生产者（r13 的 LN-bwd / gamma_prep / head 的 bwd 等 kernel）用 atomicMax 给出 exact 值（这些生产者是我们自己的 kernel，可以改），X 与权重用 delayed（权重变化慢；X 的生产者里 flash attention 的输出 `o` 不能改，只能 delayed），并在量化 kernel 里输出 cur/prev 比值作为运行时饱和监控。" + nl
        + "- **局限**：S 门只有 2 个批、50 步；诊断没有包含 ViT-L（诊断代码在 ViT-L S 门之后才加入）；没有对 `amax=exact` 与 `delayed` 做 B 门以外的长轨迹比较。" + nl
    )


def TEXT(sav, res):
    s_l, s_b = sav["vitl16"], sav["vitb16"]
    rows = _diag_rows()
    diag_txt = "没有带诊断的 S 门结果（not measured）。"
    if rows:
        parts = []
        for m, scen, n, pol, dg, e0, ea, fin, gn in rows:
            if "amax=exact" in pol:
                parts.append(f"{m} {scen} {n}（exact）：EMA 偏差 vs R0 {e0:.4f}")
            else:
                parts.append(f"{m} {scen} {n}（delayed）：EMA 偏差 vs R0 {e0:.4f}，延迟调用 {int(dg.get('n_delayed', 0))} 次，其中当前 amax 超过 2× 上一次 amax 的 {int(dg.get('n_over', 0))} 次，cur/prev 最大 {dg.get('max_ratio', 0):.3f}，调用点 {dg.get('n_sites', 0)} 个、其中最大比值 >2 的 {dg.get('n_sites_ratio_gt_2', 0)} 个，最差 {dg.get('worst_sites', [])[:3]}")
        diag_txt = "；".join(parts)
    headline = _headline(res)
    pol = (f"ViT-L：`{POLICY['vitl16']}`（B4）；ViT-B：`{POLICY['vitb16']}`（B3）；两个模型的仅 head 退路档：`{HEAD_ONLY}`（H3；更保守的 `head.dw=mx8` 对 R0 的余弦与 FINAL 几乎相同）。"
           f"按 R20a `per_step_estimate`（融合量化器、amax 由生产者给出，**不是实测**）：ViT-L B4 ≈ {s_l['B4']['ms']} ms（{s_l['B4']['pct_of_final_step']}%），H3 ≈ {s_l['H3']['ms']} ms（{s_l['H3']['pct_of_final_step']}%）；"
           f"ViT-B B3 ≈ {s_b['B3']['ms']} ms（{s_b['B3']['pct_of_final_step']}%），H3 ≈ {s_b['H3']['ms']} ms（{s_b['H3']['pct_of_final_step']}%）。")
    rdec = "**R21b（head）：执行**（H3 = ①+②a 过 11.6 的三条：graph 下 5.1× / 2.7× 的 GEMM 速度、B+S 门、≥3 ms）。**R22（骨干）：执行**（③ 过三条；ViT-L 用 nofirst，ViT-B 不需要 prefix / nofirst）。④（fc 前向 nv4）、⑤a（qkv 前向 mx8）：不执行（R20a 的速度 / 节省规则不满足，B 门上 fc.f=nv4 也不过）。"
    nm = "；".join(NOT_MEASURED_LIST)

    sg = (
        "**读数**：(1) 所有候选所有场景都过：EMA 偏差最大值（对 R0 / 对 FINAL）：ViT-L 0.32% / 0.28%，ViT-B 0.74% / 0.85%（门 5%），无 NaN / Inf，grad-norm 比值 0.97–1.11。"
        "(2) **偏差几乎全部来自 head 前向 mx8（H3 与 B3 / B4 的偏差相同）**：ViT-B H3 0.65% 对 B3 0.66%，ViT-L H3 0.29% 对 B3 0.32% / B4 0.29%；骨干 nv4 dgrad+wgrad 在 50 步里没有给 EMA 增加可见的偏差。FINAL 自己对 R0 的偏差只有 0.02%–0.14%。"
        "(3) **B3 的 grad-norm 比值在 ViT-L 上会离开 1**：schedule 步 40 为 1.068、compressed 步 50 为 1.064、fixed_mid 1.069、fixed_late 1.105；B4（nofirst）同场景 ≤1.009（compressed 步 50 一次 1.051），H3 ≤1.011——block 0 的 massive activation 在 B3 里确实把梯度范数推高了 5–10%，nofirst 去掉了大部分。ViT-B 没有这个现象（B3 最大 1.048）。")

    concl = f"""**门结果**见 §4.3（决赛）和 §4.5（S 门）；筛选全表见 §4.2。

**1. 推荐 POLICY（分模型）**

| 模型 | 骨干 + head（最激进的推荐） | 仅 head 退路 | 估计节省（R20a，非实测） |
|---|---|---|---|
| ViT-L | `{POLICY['vitl16']}` | `{HEAD_ONLY}` | B4 {s_l['B4']['ms']} ms（{s_l['B4']['pct_of_final_step']}%）；H3 {s_l['H3']['ms']} ms（{s_l['H3']['pct_of_final_step']}%）；（B3 {s_l['B3']['ms']}，B5 {s_l['B5']['ms']}） |
| ViT-B | `{POLICY['vitb16']}` | `{HEAD_ONLY}` | B3 {s_b['B3']['ms']} ms（{s_b['B3']['pct_of_final_step']}%）；H3 {s_b['H3']['ms']} ms（{s_b['H3']['pct_of_final_step']}%）；（B4 {s_b['B4']['ms']}，B5 {s_b['B5']['ms']}） |

**ViT-B 的余量**：最差的一批（seed 5，FINAL 自己对 R0 也只有 0.9824）上 B3 / B4 / B5 对 P1 的余弦为 {_g(res, 'vitb16', 'B3', 'gcos_min', 'vs_p1'):.4f}（放宽门最小 ≥0.97，余量约 0.005）；中位数 {_g(res, 'vitb16', 'B3'):.4f}（放宽门 0.98）。ViT-B 的骨干 FP4 在最差批上离放宽门线不远，R22 的真 kernel 门要盯这一项。**批次噪声**：数据加载器跨进程不确定，同一个模型三次独立抽到的 8 批上 ViT-B B4 的对 R0 中位数分别为 {_batch_noise(res)}（差到 0.0035），所以 ViT-B 的严格门（中位数 ≥0.99）通不通过取决于抽到的批，不能用它区分 B3 / B4 / B5；放宽门在三次抽批里都通过。

选择规则（11.6）：SR 和 RN 都过 → 选 SR；mx4 的判断见第 3 条。其中 ViT-L 的推荐是 B4 而不是字面上“最激进的通过者”B3：**B3 在两个模型上都过放宽门**，但 ViT-L 上 B3 的中位数明显低于 B4 / B5（{_spread(res)}；ViT-L 上 B3 过不了严格门），S 门里 grad-norm 比值到 1.06–1.11；B4 多花约 {s_l['_nofirst_loss_ms']} ms（≈0.3%）就严格门通过、grad-norm 回到 1.00。如果协调者要按字面取最激进者，就是 `{AGGRESSIVE}`（两个模型都通过放宽门和 S 门）。

**2. prefix 与 nofirst 的取舍**

| | 精度（B 门，≥8 批，对 R0 中位 / 最小） | 成本（R20a 估计，ms/步 L / B） | 备注 |
|---|---|---|---|
| B3（都不用） | ViT-L {_f(res, 'vitl16', 'B3')}；ViT-B {_f(res, 'vitb16', 'B3')} | 0 | ViT-L 严格门不过；S 门 grad-norm 比值最高 1.105 |
| B4（nofirst） | ViT-L {_f(res, 'vitl16', 'B4')}；ViT-B {_f(res, 'vitb16', 'B4')} | {s_l['_nofirst_loss_ms']} / {s_b['_nofirst_loss_ms']}（③ 的 1/24、1/12 的 GEMM 节省没了；block 0 回到 FINAL 的 mx8） | ViT-L：与 B5 同精度，S 门 grad-norm 正常 |
| B5（prefix） | ViT-L {_f(res, 'vitl16', 'B5')}；ViT-B {_f(res, 'vitb16', 'B5')} | {s_l['_prefix_side_gemm_ms']} / {s_b['_prefix_side_gemm_ms']}（侧 GEMM 实测，加法并进 epilogue 之前） | 精度和 B4 相同，成本是 nofirst 的 5.6 倍（L）/ 2.7 倍（B） |

- 在全局梯度余弦上 **prefix 和 nofirst 基本无法区分**（{_spread(res)}），nofirst 的成本只有 prefix 的约 1/6（L）/ 约 1/3（B），所以取舍是：**ViT-L 用 nofirst（B4）；ViT-B 两者都不需要（B3）**。
- prefix 的优点只在块级单步测试里可见：block 0 / joint 块的 wgrad 误差明显下降（见 §4.1 d、f），但这一点没有转化为全局梯度的可见差别；prefix 还会把 block 0 以外的所有块的 5 / 105 个 token 都走 bf16 侧 GEMM，需要额外 kernel。如果之后的 R22 验证里 block 0 仍是问题，nofirst 是可以立即切换的退路，prefix 不必实现。
- 局限：B 门只看全局梯度余弦；逐参数最小余弦在 FINAL 上也低（噪声），不能用来区分。

{_item3(res)}
**4. delayed amax**（候选 B4 / B3 的 delayed 与 B3x / B4x 的 exact 的 S 门结果见 §4.5；逐行诊断在 `results.json["r20b"]["delayed_amax_diag"]`）
{_delayed_conclusion(res)}

**5. 失败项**
- 无 autocast 的 fp32 默认块 SDPA 报 `No available kernel`（`RuntimeError`），所以块级合理性测试（§4.1 f）的参考改成 R0 默认块在 bf16 autocast 下的结果；第一版测试曾用 `all.fdw=bf16` 的块作参考，已被这版覆盖。
- 一次启动时策略串里的 `fc.f=nv4`、`qkv.f=mx8` 作为候选名被当成“名字=策略”解析，报 `unknown policy token 'nv4'`，已把候选名改成 `fcF4 / qkvF8` 并重跑（旧 json 已删除重建，没有留下污染的数据）。
- ViT-L 筛选的第二次启动在 seed 1 OOM：`Handle.ctx` 引用 ctx，局部变量 `handle` 在下一个 seed 之前还活着，钉住了 ViT-L 约 6 GB 的模型和优化器状态；已在 seed 循环结尾 `handle.ctx = None; del handle`，重跑通过。
- 三次 `assert_gpu_exclusive`：另一个代理的 GPU 进程在初始化 CUDA 期间启动，我的进程被中止；已加 `retry.sh`（等待并重试）。
- 数据加载器在不同进程里对同一个 seed 给出的 batch 不完全一致（R0 loss 不同），所以旧的混合进程 ViT-B finals 里同一个 seed 的不同候选不是同一批；已用单进程重跑 ViT-B finals，旧文件保留为 `gateB_r20b_vitb16_finals_mixedproc.json`。

**6. 需要协调者决定**
1. ViT-L 取 B4（nofirst，推荐）还是字面上的最激进者 B3；
2. mx4 是否因为“过门且更快”而改用（我建议不）；
3. 是否要为 R22 保留 `nofirst` 作为运行时选项（成本 0.3 ms，不需要额外 kernel），以及是否实现 prefix（我建议不）；
4. wgrad 的 RHT 是否取消（对照臂显示没有可见收益，R21 的量化器可以先不做 RHT 以降低复杂度）；
5. R21b 的 head 策略取 H3（含 `head.f=mx8`）还是 H1（仅 dgrad+wgrad nv4）：H3 多 3.3 / 3.05 ms，B 门严格通过，S 门通过；
6. ④ / ⑤a 的“破例”：本轮的 B 门数据显示 fc.f=nv4 不过放宽门（不建议破例）；qkv.f=mx8 在 ViT-L 的 B3 底上不过、其余底上过，信号不稳（n≤3），不建议。

**7. 下一瓶颈**：骨干 nv4 的 amax 要由生产者给出（delayed amax 已经在仿真里证明不需要额外遍历）、双输出 FP4 量化器的带宽（R21a）；block 0 的 massive activation（ViT-L）要靠 nofirst 或 prefix 绕开；整步里 attention / LN / 逐元素（45 ms）仍不在范围内。

**契约变更请求**：(1) 11.4 / 6a 12.3 的策略串里，选项修饰“紧跟的条目”：`all.w=nv4+sr+rht` 的 `sr`、`rht` 作用于该条目，`prefix`、`nofirst`、`amax=` 是全局选项——已按此实现，请 6b 明确写入；(2) `amax=delayed` 的诊断量（cur/prev 最大比值、超 2× 次数）建议在 R21 的量化 kernel 里也输出，用于运行时发现饱和；(3) 11.5 B 门：数据加载器跨进程不确定，建议明确“批次数按（进程、seed）计，比较只在同一进程内配对”（Amendment 4 已有类似说明）。
"""
    return {
        "STATUS": "完成。计划内的 B 门（筛选 3 批 + 决赛 8 批）、S 门（ViT-L：B3 / H3 / B4；ViT-B：B3 / H3，另有 B4 / exact-amax 对照）、对照臂（8 批，B4 为底）、块测试都已跑完；没测的项见 §0 “没测”。",
        "HEADLINE": headline,
        "POLICY_SUMMARY": pol,
        "R_DECISION": rdec,
        "NOT_MEASURED": nm,
        "SGATE_NOTES": sg,
        "CONCLUSIONS": concl,
    }


def DECISION(sav, res):
    return {
        m: {
            "POLICY": POLICY[m],
            "POLICY_most_aggressive_passer": AGGRESSIVE,
            "POLICY_head_only_fallback": HEAD_ONLY,
            "POLICY_head_only_conservative_mx8": "head.dw=mx8,amax=delayed",
            "format": "nv4 (mx4 passes the relaxed gate but is 0.5% faster only and less accurate)",
            "rounding": "sr (gradient dY of d and w)",
            "weights": "w2d", "wgrad_rht": "kept in POLICY as specified; B-gate arms (n=8, B4 base) show no visible gain from RHT (|diff| <= 0.0002), see POLICY_no_rht_variant", "POLICY_no_rht_variant": POLICY[m].replace("all.w=nv4+sr+rht", "all.w=nv4+sr"), "prefix": False,
            "nofirst": m == "vitl16",
            "amax": "delayed",
            "estimated_saving_ms_not_measured": {k: v for k, v in sav[m].items() if not k.startswith("_")},
            "R21b_head": True, "R22_backbone": True,
            "not_published": ["fc.f=nv4", "qkv.f=mx8"],
        } for m in ("vitl16", "vitb16")
    }


def DIAG_ROWS():
    return [{"model": m, "scenario": scen, "cand": n, "policy": pol, "diag": dg, "ema_vs_r0": e0, "ema_vs_r0_all50": ea, "finite": fin} for (m, scen, n, pol, dg, e0, ea, fin, gn) in _diag_rows()]
