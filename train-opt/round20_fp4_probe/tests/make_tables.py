"""Generate the markdown tables of NOTES.md / results.json from profile/per_step_estimate.json (all numbers measured or labelled derived)."""
import json
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "profile"
d = json.load(open(P / "per_step_estimate.json"))
lines = []
res = {}
for model, m in d["models"].items():
    lines.append(f"\n#### {model}: per (class, linear, kind) GEMM time (us, CUDA-graph replay, median of 3) and TFLOPS\n")
    lines.append("| case | M x N x K | FINAL now | best NVFP4 (usable) | best MXFP4 | NVFP4 vs now | MXFP4 vs now | now = |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    order = {"fwd": 0, "dgrad": 1, "wgrad": 2}
    rows = sorted(m["cases"].items(), key=lambda kv: ({"plain": 0, "head": 1, "joint": 2}[kv[0].split("/")[0]], kv[0].split("/")[1], order[kv[0].split("/")[2]]))
    for key, c in rows:
        cls, lin, kind = key.split("/")
        if cls == "joint":
            continue  # joint cases are in the json; the table shows plain M and the head
        now_name = {"fwd": "bf16 epi" if lin == "qkv" else "mx8 epi", "dgrad": "mx8", "wgrad": "mx8 native"}[kind] if cls == "plain" else "bf16 (head)"
        fl = c["flops"]

        def tf(us):
            return fl / us / 1e6

        lines.append(f"| {cls} {lin} {kind} | {c['M']}x{c['N']}x{c['K']} | {c['cur']:.0f} ({tf(c['cur']):.0f} TF) | {c['nv4']:.0f} ({tf(c['nv4']):.0f} TF) | {c['mx4']:.0f}{'*' if c.get('mx4_derived') else ''} ({tf(c['mx4']):.0f} TF) | {c['cur'] / c['nv4']:.2f}x | {c['cur'] / c['mx4']:.2f}x | {now_name} |")
    lines.append("\n(mx4 的带 epilogue 版本也全部是直接实测（带 * 的才是推导值）；head 在 FINAL 里是 bf16，head 的 fwd/dgrad 没有 epilogue。)")
    # backend comparison over all cases of this model
    tot = defaultdict(float)
    n = 0
    for key, c in m["cases"].items():
        b = c["plain_nv4_backends"]
        if None in b.values():
            continue
        n += 1
        for k, v in b.items():
            tot[k] += v
    lines.append(f"\nPlain NVFP4 backend sums over the {n} cases of {model} (bf16 out; wgrad = bf16-out variants): " + ", ".join(f"{k} {v / 1e3:.2f} ms" for k, v in tot.items()))
    res[model] = {"plain_nv4_backend_sum_ms": {k: v / 1e3 for k, v in tot.items()}, "n_cases": n}
(P / "tables_r20a.md").write_text("\n".join(lines))
print("\n".join(lines))
json.dump(res, open(P / "backend_sums.json", "w"), indent=1)


# ---------------------------------------------------------------- 4.3 per-step GEMM table and 4.6 step estimate (auto-generated from the same json)
STEP = {"vitl16": 101.8, "vitb16": 86.6}
def f2(x):
    return "n/a" if x is None else f"{x / 1e3:.2f}"

t43 = ["| ms/步 | 骨干 d（now→nv4） | 骨干 w | fc 前向 | qkv 前向（bf16→mx8→nv4） | head f（bf16→mx8→nv4） | head d | head w |", "|---|---|---|---|---|---|---|---|"]
for model, m in d["models"].items():
    S = m["per_step_gemm_us"]
    t43.append(f"| {model} | {f2(S['bb_d.cur'])}→{f2(S['bb_d.nv4'])} | {f2(S['bb_w.cur'])}→{f2(S['bb_w.nv4'])} | {f2(S['bb_f_fc.cur'])}→{f2(S['bb_f_fc.nv4'])} | {f2(S['bb_f_qkv.cur'])}→{f2(S['bb_f_qkv.mx8'])}→{f2(S['bb_f_qkv.nv4'])} | {f2(S['head_f.cur'])}→{f2(S['head_f.mx8'])}→{f2(S['head_f.nv4'])} | {f2(S['head_d.cur'])}→{f2(S['head_d.nv4'])} | {f2(S['head_w.cur'])}→{f2(S['head_w.nv4'])} |")
t43.append("\nprefix-token 侧 GEMM（gather + bf16 GEMM fp32 输出 + 加进 FP4 的 wgrad 结果，4 个 Linear × 全部块，ms/步）：")
for model, m in d["models"].items():
    pf = m.get("prefix_side_gemm_ms") or {}
    t43.append(f"- {model}: 所有块只排除 5 个 prefix token：{pf.get('prefix5_all_blocks_ms')}；普通块 5 个 + joint 块 105 个：{pf.get('prefix5_plain+prefix105_joint_ms')}")
(P / "tables_r20a_43.md").write_text("\n".join(t43))

t46 = ["| 项（ms/步，L / B） | ΔGEMM | Δ量化（融合，nv4，amax 由生产者给出） | 节省 | 占步时 | ≥3 ms 或 ≥3% |", "|---|---|---|---|---|---|"]
names = {"item1_head_dw": "① head dgrad+wgrad → nv4", "item3_bb_dw": "③ 骨干 dgrad+wgrad → nv4", "item4_fc_f": "④ fc1/fc2 前向 → nv4", "item5a_qkv_f_mx8": "⑤a qkv 前向 → MXFP8", "item2a_head_f_mx8": "②a head 前向 → MXFP8"}
scen = "fused_nv4_amax_from_producer"
for k, nm in names.items():
    a, b = d["models"]["vitl16"]["step_estimate"][scen]["per_item"][k], d["models"]["vitb16"]["step_estimate"][scen]["per_item"][k]
    ok = lambda x, m_: "满足" if (x["saving_ms"] >= 3 or 100 * x["saving_ms"] / STEP[m_] >= 3) else "**不满足**"
    t46.append(f"| {nm} | {a['dGEMM_ms']:.2f} / {b['dGEMM_ms']:.2f} | {a['dQuant_ms']:.2f} / {b['dQuant_ms']:.2f} | {a['saving_ms']:.2f} / {b['saving_ms']:.2f} | {100 * a['saving_ms'] / STEP['vitl16']:.1f}% / {100 * b['saving_ms'] / STEP['vitb16']:.1f}% | {ok(a, 'vitl16')} / {ok(b, 'vitb16')} |")
t46.append("\n合计（节省为负 = 整步变慢）：\n\n| 场景 | ViT-L S1（①+③） | ViT-L S2（再加 ④⑤a②a） | ViT-B S1 | ViT-B S2 |\n|---|---|---|---|---|")
labels = {"existing_kernels_unfused(nv4)": "现成 kernel 不融合（实测）", "fused_nv4_amax_from_producer": "融合，nv4，amax 由生产者给出", "fused_nv4_separate_amax_pass": "融合，nv4，单独 amax 遍历", "fused_mx4": "融合，mx4"}
for sc, lb in labels.items():
    row = []
    for model in ("vitl16", "vitb16"):
        e = d["models"][model]["step_estimate"][sc]
        row += [f"{e['S1_items_1+3_ms']:.2f} ms（{e['S1_pct']:.1f}%）", f"{e['S2_items_1+3+4+5a+2a_ms']:.2f}（{e['S2_pct']:.1f}%）"]
    t46.append(f"| {lb} | " + " | ".join(row) + " |")
(P / "tables_r20a_46.md").write_text("\n".join(t46))
print("\n".join(t43)); print("\n".join(t46))

extra = ["\n其他（只有 GEMM 项，没有量化估计）：\n", "| 项 | ΔGEMM ms（L / B） | 速度倍数（L / B） |", "|---|---|---|"]
for k, nm in (("1b_head_dw_mx8_fallback", "①b head dgrad+wgrad → mx8（退路，对 bf16）"), ("2b_head_f_nv4", "②b head 前向 → nv4（stretch）"), ("5b_qkv_f_nv4", "⑤b qkv 前向 → nv4（stretch）"), ("x_proj_f_nv4(not proposed: bandwidth bound)", "proj 前向 → nv4（不建议：带宽受限）")):
    a, b = d["models"]["vitl16"]["items"][k], d["models"]["vitb16"]["items"][k]
    extra.append(f"| {nm} | {a['dgemm_ms']:.2f} / {b['dgemm_ms']:.2f} | {a['speedup_x']:.2f}× / {b['speedup_x']:.2f}× |")
(P / "tables_r20a_46.md").write_text("\n".join(t46) + "\n" + "\n".join(extra))
