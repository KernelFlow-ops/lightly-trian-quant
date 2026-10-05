"""Build project-wide training throughput, correctness, and curve summaries."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ("vitl16", "vitb16")


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def collect() -> list[dict[str, Any]]:
    rows = []
    def round_number(path: Path) -> int:
        match = re.match(r"round(\d+)", path.parent.name)
        return int(match.group(1)) if match else 999

    for result_path in sorted(ROOT.glob("round*/results.json"), key=round_number):
        try:
            data = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for model in MODELS:
            for variant, item in data.get(model, {}).items():
                gpu = item.get("gpu_step", {})
                accuracy = item.get("accuracy", {})
                level_b = accuracy.get("level_b", {})
                level_c = accuracy.get("level_c", {})
                level_d = accuracy.get("level_d", {})
                rows.append({
                    "round": item.get("round", result_path.parent.name), "model": model, "variant": variant,
                    "desc": item.get("desc"), "weighted_img_s": gpu.get("weighted_img_s"),
                    "early_ms": gpu.get("early", {}).get("step_ms", {}).get("mean"),
                    "mid_ms": gpu.get("mid", {}).get("step_ms", {}).get("mean"),
                    "late_ms": gpu.get("late", {}).get("step_ms", {}).get("mean"),
                    "e2e_img_s": item.get("e2e", {}).get("img_s"), "epoch_s": item.get("epoch_s", {}).get("total"),
                    "fwd_ms": item.get("breakdown_ms", {}).get("fwd"), "loss_ms": item.get("breakdown_ms", {}).get("loss"),
                    "bwd_ms": item.get("breakdown_ms", {}).get("bwd"), "opt_ms": item.get("breakdown_ms", {}).get("opt"),
                    "kernels_step": item.get("kernels_per_step"), "syncs_step": item.get("syncs_per_step"),
                    "gpu_busy_ratio": item.get("gpu_busy_ratio"), "peak_mem_gb": item.get("peak_mem_gb"), "max_batch": item.get("max_batch"),
                    "grad_cos_min": level_b.get("grad_cos_min"), "level_b_pass": level_b.get("pass"), "curve300_max_rel": level_c.get("curve300_max_rel"),
                    "val_miou": level_d.get("val_miou"), "val_head_iou": level_d.get("val_head_iou"), "gate_d": level_d.get("pass"),
                })
    return rows


def _metric_rows() -> list[str]:
    rows = collect()
    lines: list[str] = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        baseline = next((row for row in subset if row["round"] == "round0_baseline" and row["variant"] in {"B0-harness", "baseline"}), None)
        lines += [f"## {model}", "", "| round | variant | change | weighted img/s | vs R0 | vs prev best | early/mid/late ms | fwd/loss/bwd/opt ms | kernels/step | syncs/step | GPU busy | e2e img/s | epoch s | peak GB | max batch | grad cos | C curve | mIoU/head IoU | D gate |", "|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        previous_best = None
        for row in subset:
            ratio = row["weighted_img_s"] / baseline["weighted_img_s"] if baseline and row["weighted_img_s"] and baseline["weighted_img_s"] else None
            prev_ratio = row["weighted_img_s"] / previous_best if previous_best and row["weighted_img_s"] else None
            lines.append("| " + " | ".join([
                str(row["round"]), str(row["variant"]), str(row["desc"] or "null").replace("|", "/"), _fmt(row["weighted_img_s"]), _fmt(ratio, 3), _fmt(prev_ratio, 3),
                "/".join(_fmt(row[key]) for key in ("early_ms", "mid_ms", "late_ms")),
                "/".join(_fmt(row[key]) for key in ("fwd_ms", "loss_ms", "bwd_ms", "opt_ms")),
                _fmt(row["kernels_step"], 1), _fmt(row["syncs_step"], 1), _fmt(row["gpu_busy_ratio"], 3),
                _fmt(row["e2e_img_s"]), _fmt(row["epoch_s"]), _fmt(row["peak_mem_gb"]), _fmt(row["max_batch"], 0),
                _fmt(row["grad_cos_min"], 5), _fmt(row["curve300_max_rel"], 4),
                f"{_fmt(row['val_miou'], 4)}/{_fmt(row['val_head_iou'], 4)}", _fmt(row["gate_d"]),
            ]) + " |")
            if row["weighted_img_s"] is not None and (row["level_b_pass"] is True or row["round"] == "round0_baseline"):
                previous_best = max(previous_best or 0.0, row["weighted_img_s"])
        lines.append("")
    return lines


def write_summary() -> tuple[Path, Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = collect()
    csv_path, md_path = RESULTS / "summary.csv", RESULTS / "summary.md"
    columns = list(rows[0]) if rows else ["round", "model", "variant"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    heading = ["# train-opt results", "", "All cells without a measurement remain `null`. Primary throughput is the 0.2/0.6/0.2 weighted image/s.", ""]
    md_path.write_text("\n".join(heading + _metric_rows()), encoding="utf-8")
    return md_path, csv_path


def write_plots() -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    rows = collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    fig, axes = plt.subplots(1, len(MODELS), figsize=(max(12, 6 * len(MODELS)), 4.7))
    has_data = False
    for ax, model in zip(axes, MODELS):
        subset = [row for row in rows if row["model"] == model and any(row[key] is not None for key in ("early_ms", "mid_ms", "late_ms"))]
        if not subset:
            ax.set_title(f"{model}: no measurements")
            continue
        has_data = True
        labels = [f"{r['round'].replace('round', 'r')}\n{r['variant']}" for r in subset]
        bottom = [0.0] * len(subset)
        for phase, key, weight in (("early", "early_ms", 0.2), ("mid", "mid_ms", 0.6), ("late", "late_ms", 0.2)):
            values = [weight * (r[key] or 0.0) for r in subset]
            ax.bar(labels, values, bottom=bottom, label=f"{phase} x {weight:g}")
            bottom = [a + b for a, b in zip(bottom, values)]
        for index, row in enumerate(subset):
            if row["weighted_img_s"] is not None:
                ax.text(index, bottom[index], f"{row['weighted_img_s']:.1f} img/s", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("weighted GPU step ms")
        ax.set_title(f"{model} weighted latency and throughput")
        ax.legend()
        ax.tick_params(axis="x", labelsize=7)
    if has_data:
        fig.tight_layout()
        output = RESULTS / "throughput.png"
        fig.savefig(output, dpi=140)
        outputs.append(output)
    plt.close(fig)

    curve_dir = RESULTS / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for curve_path in sorted(ROOT.glob("round*/**/*curve*.json")):
        try:
            data = json.loads(curve_path.read_text())
            round_name = curve_path.relative_to(ROOT).parts[0]
            for label, values in (("loss_ema", data.get("loss_ema", data.get("loss"))), ("val_miou", data.get("val_miou", data.get("miou")))):
                if not isinstance(values, list) or not values:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                x = data.get("steps") if isinstance(data.get("steps"), list) and len(data["steps"]) == len(values) else range(len(values))
                ax.plot(x, values, linewidth=1)
                ax.set(xlabel="step" if label == "loss_ema" else "epoch", ylabel=label, title=round_name)
                fig.tight_layout()
                output = curve_dir / f"{round_name}_{curve_path.stem}_{label}.png"
                fig.savefig(output, dpi=140)
                plt.close(fig)
                outputs.append(output)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return outputs


def main() -> None:
    md, csv_path = write_summary()
    plots = write_plots()
    print(f"wrote {md} and {csv_path}; plots={len(plots)}")


if __name__ == "__main__":
    main()
