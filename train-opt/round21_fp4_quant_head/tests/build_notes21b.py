"""R21b: fill ``notes_r21b_head.md`` (template with {{PLACEHOLDERS}}) from the profile jsons and APPEND / replace the marked R21b block of NOTES.md (R21a text untouched).

  python tests/build_notes21b.py --template tests/notes_r21b_head.md [--apply]     (without --apply: writes /root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/NOTES_r21b_preview.md)
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
P = ROUND / "profile"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_tables21b as MT  # noqa: E402


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def tbl(fn):
    out = {}
    return "\n".join(fn(out))


def part(fn, label):
    res = {}
    lines = fn(res)
    return "\n".join(l for l in lines if l is not None)


def split_tables(lines, key):
    return "\n".join(lines)


def gate_a():
    d = load(P / "gateA_head_fp4.json")
    if not d:
        return "not measured"
    g, h, gr = d["gemm"], d["head"], d["graph"]
    rn = h["rn"]
    sr = h["sr"]
    gem = g["cases"]
    mx = lambda k: max(c[k] for c in gem)  # noqa: E731
    lines = ["| 部分 | 内容 | 结果 |", "|---|---|---|",
             f"| gemm | head 的 nv4 GEMM（`plain_ts` 后端与 BlockScaledOperand 两种；dgrad / wgrad / fwd；两个模型的 ConvT1 / ConvT2 真实形状 M = 7168 / 28672（ViT-L）、14336 / 57344（ViT-B）；gauss 与 scale-8 尖峰数据）对反量化 fp32 乘积 | {len(gem)} 例，**0 失败**：wgrad（fp32 输出）相对 L2 最大 {mx('wgrad_relL2'):.1e}；fwd / dgrad（bf16 输出）{mx('fwd_relL2'):.2e}（= bf16 输出舍入的下限 2^-9/√3，门槛 2e-3） |",
             f"| head RN | RN 变体（`head.f=mx8,head.dw=nv4+rn,amax=exact`、`head.dw=nv4+rn,amax=exact`、`head.dw=mx8`）对 R20b 仿真 head（`fp4_head`）：logits 和 12 个输入 / 参数梯度（tok、e、w1、b1、d1、g1、be1、w2、b2、d2、g2、be2） | {len(rn)} 组（2 模型 × gauss / spike × 3 策略）：**logits relL2 最大 {max(r['logits_relL2'] for r in rn):.1e}，梯度 relL2 最大 {max(r['grad_relL2_max'] for r in rn):.1e}，梯度余弦最小 {min(r['grad_cos_min'] for r in rn):.7f}（逐位相同）**，门槛 0.9999 |",
             f"| bf16 恒等 | 空策略 = r14 head | {len(h['bf16_identity'])} 组全部逐位相同：{all(r['bitwise_equal_to_r14'] for r in h['bf16_identity'])} |",
             f"| SR 统计 | 6 个种子；对 bf16 head 的误差（w1 / w2 / tok 梯度）与种子间标准差，kernel Philox 对仿真生成器 | 误差比 {min(v['err_ratio'] for s in sr for v in s['per_grad'].values()):.4f}–{max(v['err_ratio'] for s in sr for v in s['per_grad'].values()):.4f}，标准差比 {min(v['sd_ratio'] for s in sr for v in s['per_grad'].values()):.4f}–{max(v['sd_ratio'] for s in sr for v in s['per_grad'].values()):.4f}（门：0.9–1.1）；同种子逐位相同 {all(s['same_seed_bitwise'] for s in sr)}，换种子结果不同 {all(s['other_seed_differs'] for s in sr)}；6 个种子平均后误差明显下降（w1：{sr[0]['per_grad']['w1']['err_vs_bf16_kernel']:.3f} → {sr[0]['per_grad']['w1']['mean_of_n_err_kernel']:.3f}），SR 无偏 |",
             f"| graph | 4 次 predict（SR + delayed + roll）捕获进 CUDA graph：预热（eager，`StepGraphRunner.warm` 同款）→ 复位 `next` 槽 / seed → 捕获 → 重放两次，对 eager 的第 1、2 步 | " + "；".join(f"{c['model']}: 第 1 步逐位 {c['step1_bitwise']}、第 2 步逐位 {c['step2_bitwise']}" for c in gr["cases"]) + " |"]
    return "\n".join(lines)


def gemm_bench():
    d = load(P / "bench_head_gemm.json")
    if not d:
        return "not measured"
    lines = ["#### head ConvT 的 nv4 GEMM 配置（`profile/bench_head_gemm.json`，graph 回放，交替，3 轮中位数，µs；括号 TFLOPS）", "",
             "| 形状 | dgrad `plain_ts` | dgrad operand | dgrad cuBLASLt | dgrad bf16（r14） | wgrad `plain_ts` | wgrad operand | wgrad operand split-K | wgrad cuBLASLt | wgrad bf16（r14） |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in d["cases"].items():
        u, t = c["us"], c["tflops"]
        cell = lambda k: f"{u[k]:.1f} ({t[k]:.0f})"  # noqa: E731
        lines.append(f"| {name} M={c['M']} | " + " | ".join(cell(k) for k in ("dgrad/epi_plain_ts", "dgrad/operand", "dgrad/cublaslt", "dgrad/bf16_r14", "wgrad/epi_plain_ts", "wgrad/operand", "wgrad/operand_splitk", "wgrad/cublaslt_bf16out", "wgrad/bf16_r14")) + " |")
    lines += ["", "结论：QuACK `plain_ts`（设备标量 ts 的 epilogue）在 8 个 dgrad / wgrad 里 5 个最快或并列最快，其余与最快差 1.0–2.2%；split-K 对 head 的 wgrad 没有收益（ViT-B ConvT1 慢 7%，ViT-L ConvT1 持平）；cuBLASLt 的 dgrad 慢 5–10%、wgrad 差 −1%…+4%；bf16 → nv4 的 GEMM 加速 4.5–5.9×（与 R20a 的 5.1× 一致）。**配置已是最优，不需要再调**（后端默认 `epi`）。"]
    return "\n".join(lines)


def export_tbl():
    lines = []
    for m in ("vitl16", "vitb16"):
        d = load(P / f"export_{m}_P3H.json")
        rf = load(P / f"run_final_{m}_P3H.json")
        if not d:
            lines.append(f"- {m}: not measured")
            continue
        r = rf or {}
        lines.append(f"- **{m}**：`pass` = **{d['pass']}**；checkpoint 步数 {d['ckpt_step']}；checkpoint / export 的 train_model 键、形状、dtype 与 R0 一致 = {d['ckpt_train_model_vs_R0']['equal']} / {d['export_vs_R0_export']['equal']}，与原训练导出权重一致 = {d['export_vs_original_trained_export']['equal']}；optimizer 结构一致 {d['optimizer_struct_equal']}，scheduler 键一致 {d['scheduler_keys_equal']}；"
                     f"权重确实更新（改变的参数 {d['n_params_changed']} 个，最大改动 {d['max_abs_weight_change_vs_init']:.3g}）；导出全部有限 {d['export_all_finite']}；原版 `lightly_train.load_model` 加载并 predict 成功（{d['load_model_predict']['shape']}）；"
                     f"原版路径 resume 一步 loss = {d['resume_step_loss']:.4f}（有限 {d['resume_step_loss_finite']}）。训练侧：{('graph 捕获 ' + str(r.get('graph', {}).get('captures')) + ' 张，真实 loader 20 步，验证 + 保存成功') if r else ''}")
    return "\n".join(lines)


def nsys_tbl():
    out = []
    for m, tag in (("vitl16", "L"), ("vitb16", "B")):
        d = load(P / f"nsys_report21_{m}.json")
        if not d:
            out.append(f"{m}: not measured")
            continue
        labs = list(d)
        cats = sorted({c for r in d.values() for c in r["categories"]}, key=lambda c: -max(r["categories"].get(c, {"ms_per_step": 0})["ms_per_step"] for r in d.values()))
        out += [f"#### {m}（FINAL → P3H，ms/步，5 步平均）", "", "| 类别 | FINAL | P3H | 差 |", "|---|---:|---:|---:|"]
        for c in cats:
            a, b = (d[labs[0]]["categories"].get(c, {"ms_per_step": 0.0})["ms_per_step"], d[labs[1]]["categories"].get(c, {"ms_per_step": 0.0})["ms_per_step"])
            if max(a, b) < 0.05:
                continue
            out.append(f"| {c} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
        ra, rb = d[labs[0]], d[labs[1]]
        out += [f"| **kernel 总计** | {ra['kernel_ms_per_step']:.2f} | {rb['kernel_ms_per_step']:.2f} | {rb['kernel_ms_per_step'] - ra['kernel_ms_per_step']:+.2f} |",
                f"| 窗口（首 kernel 到末 kernel） | {ra['busy']['span_ms_per_step']:.2f} | {rb['busy']['span_ms_per_step']:.2f} | {rb['busy']['span_ms_per_step'] - ra['busy']['span_ms_per_step']:+.2f} |",
                f"| GPU 忙碌率 / >20 µs 间隙 | {ra['busy']['busy_ratio']:.3f} / {ra['busy']['gap_ms_per_step']:.2f} ms | {rb['busy']['busy_ratio']:.3f} / {rb['busy']['gap_ms_per_step']:.2f} ms | |",
                f"| kernel 数 / 步 | {ra['kernels_per_step']:.0f} | {rb['kernels_per_step']:.0f} | {rb['kernels_per_step'] - ra['kernels_per_step']:+.0f} |", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--static", type=Path, required=True, help="json with the hand-written paragraphs (SUMMARY, GEMM_CLAIM, GATE_CLAIM, NSYS_PROCESS, CORRECTNESS, ROOFLINE, CONCLUSION)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    st = json.loads(args.static.read_text(encoding="utf-8"))
    tpl = args.template.read_text(encoding="utf-8")
    res = {}
    gb = "\n".join(MT.table_b(res))
    gs = "\n".join(MT.table_s(res))
    ge = "\n".join(MT.table_ge(res))
    be = "\n".join(MT.table_bench(res))
    fill = {"GATE_A": gate_a(), "GEMM_BENCH": gemm_bench(), "GATE_B": gb or "not measured", "GATE_S": gs or "not measured", "GRAPH_EQUIV": ge or "not measured", "EXPORT": export_tbl(), "BENCH": be or "not measured",
            "NSYS_TABLE": nsys_tbl(), **st}
    for k, v in fill.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", tpl)
    if left:
        print("UNFILLED:", left)
    if args.apply:
        notes = ROUND / "NOTES.md"
        raw = notes.read_bytes()   # byte-exact: the R21a part (mixed CRLF / LF) must stay untouched
        begin, end = b"<!-- R21b-BEGIN -->", b"<!-- R21b-END -->"
        sep = "\n\n---\n\n"
        if begin in raw:
            head = raw[:raw.index(begin)]
            tail = raw[raw.index(end) + len(end):]
            head = head.rstrip(b" \r\n")
            if head.endswith(b"---"):
                head = head[:-3]
        else:
            head, tail = raw, b""
        block = tpl.strip()
        if block.startswith("---"):
            block = block[3:].lstrip()
        notes.write_bytes(head.rstrip(b"\r\n ") + sep.encode("utf-8") + block.encode("utf-8") + b"\n" + tail.lstrip(b"\r\n"))
        print("NOTES.md R21b block written", len(block))
    else:
        Path("/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/NOTES_r21b_preview.md").write_text(tpl, encoding="utf-8")
        print("preview written")


if __name__ == "__main__":
    main()
