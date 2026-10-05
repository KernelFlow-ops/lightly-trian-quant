"""Collect every round*/results.json into results/summary.md (+ summary.csv, latency.png).

results.json layout (per round):
  {"<model>": {"<variant>": {"latency_ms": {"prep"|"model"|"post"|"e2e": {"wall_p50", "gpu_p50"}},
                             "cuda_ops": {"model": {"kernels"}, ...}, "fps_e2e",
                             "accuracy_vs_fp32": {"frame0": {"head_iou", ...}, ...}}},
   "env": {...}}
Each round's primary variant is listed in PRIMARY; everything is compared to
round0 bf16-autocast (= what scripts/infer_dinov3_eomt_head_video.py runs).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ("round0_baseline", "bf16-autocast")
MODELS = ("vitl16", "vitb16")
STAGES = ("prep", "model", "post", "e2e")


def load_rows() -> list[dict]:
    rows = []
    for rdir in sorted(ROOT.glob("round*")):
        f = rdir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for model in MODELS:
            for variant, r in data.get(model, {}).items():
                lat = r.get("latency_ms", {})
                g = r.get("accuracy_vs_fp32", {}).get("gate", {})
                rows.append({
                    "round": rdir.name, "model": model, "variant": variant,
                    **{f"{s}_ms": lat.get(s, {}).get("wall_p50") for s in STAGES},
                    "model_gpu_ms": lat.get("model", {}).get("gpu_p50"),
                    "kernels_model": r.get("cuda_ops", {}).get("model", {}).get("kernels"),
                    "fps_e2e": r.get("fps_e2e"),
                    "pooled_head_iou": g.get("pooled_head_iou"),
                    "max_conf_rel": g.get("max_conf_rel_diff"),
                    "gate_pass": g.get("pass"),
                })
    return rows


def video_section() -> list[str]:
    """Whole-video throughput (rounds with a `video` field, e.g. round8/round9)."""
    lines = ["## 整段视频吞吐（1.mp4，126 帧循环，稳态）", "",
             "原脚本 = decode + predict_head 串行循环。服务延迟 = inflight=1 时从提交到 host 拿到结果。",
             "",
             "| 模型 | round | variant | 原脚本 FPS | 优化 FPS（inflight=2） | 加速 | 服务延迟 p50 (ms) | "
             "126 帧 vs 原脚本 head IoU（pooled / min） | fp32 gate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for rdir in sorted(ROOT.glob("round*")):
        f = rdir / "results.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for model in MODELS:
            for variant, r in data.get(model, {}).items():
                v = r.get("video")
                if not isinstance(v, dict) or "pipeline_inflight2" not in v:
                    continue
                base = v.get("baseline", {}).get("fps")
                fps = v["pipeline_inflight2"].get("fps_steady")
                svc = v.get("pipeline_inflight1", {}).get("service_ms_p50")
                acc = v.get("accuracy_vs_baseline_126f", {})
                g = r.get("accuracy_vs_fp32", {}).get("gate", {})
                gate = {True: "✅", False: "❌"}.get(g.get("pass"), "-")
                sp = fps / base if fps and base else None
                lines.append(
                    f"| {model} | {rdir.name} | {variant} | {fmt(base, 1)} | {fmt(fps, 1)} | "
                    f"{fmt(sp) + 'x' if sp else '-'} | {fmt(svc)} | "
                    f"{fmt(acc.get('head_iou_pooled'), 3)} / {fmt(acc.get('head_iou_min'), 3)} | {gate} |")
    lines.append("")
    return lines


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main() -> None:
    rows = load_rows()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    if not rows:
        print("no results yet")
        return
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ["# 推理优化结果汇总", "",
             f"基线 = `{BASE[0]}` / `{BASE[1]}`（与 scripts/infer_dinov3_eomt_head_video.py 一致）。"
             "延迟为 wall-clock p50（ms，含 CPU 下发开销），输入是真实视频 1.mp4 的 1900×1440 帧，"
             "切成 2 个 512×512 crop。精度门槛（±5%）：与 fp32 eager 相比，8 帧合并的 head IoU ≥ 0.95，"
             "且每帧置信度相对误差 ≤ 5%。", ""]
    for model in MODELS:
        mrows = [r for r in rows if r["model"] == model]
        base = next((r for r in mrows if (r["round"], r["variant"]) == BASE), None)
        lines += [f"## {model}", "",
                  "| round | variant | prep | model | post | e2e | e2e FPS | kernels/model fwd | vs R0 e2e | head IoU (pooled) | conf 相对误差 max | 精度门槛 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in mrows:
            sp = (base["e2e_ms"] / r["e2e_ms"]) if base and r["e2e_ms"] else None
            gate = {True: "✅", False: "❌"}.get(r["gate_pass"], "-")
            lines.append(
                f"| {r['round']} | {r['variant']} | {fmt(r['prep_ms'])} | {fmt(r['model_ms'])} | "
                f"{fmt(r['post_ms'])} | {fmt(r['e2e_ms'])} | {fmt(r['fps_e2e'], 1)} | "
                f"{fmt(r['kernels_model'])} | {fmt(sp) + 'x' if sp else '-'} | "
                f"{fmt(r['pooled_head_iou'], 4)} | {fmt(r['max_conf_rel'], 4)} | {gate} |")
        lines.append("")
    lines += video_section()
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 4.5))
        for ax, model in zip(axes, MODELS):
            mrows = [r for r in rows if r["model"] == model]
            labels = [f"{r['round'].split('_')[0]}\n{r['variant']}" for r in mrows]
            bottom = [0.0] * len(mrows)
            for s in ("prep", "model", "post"):
                vals = [r[f"{s}_ms"] or 0.0 for r in mrows]
                ax.bar(labels, vals, bottom=bottom, label=s)
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_title(f"{model}: per-stage wall p50 (ms)")
            ax.tick_params(axis="x", labelsize=7)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "latency.png", dpi=130)
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
