"""Compare two matched baseline/final seeds and update the final D gate."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bench import write_results  # noqa: E402
from common.check import gate_d, read_lightly_metrics  # noqa: E402
from common.config import B_AUTO  # noqa: E402

ROUND = Path(__file__).resolve().parent


def _metadata(directory: Path) -> dict:
    opt = directory / "train_opt_run.json"
    baseline = directory / "train_opt_config.json"
    standard = directory / "run_config.json"
    if opt.is_file():
        data = json.loads(opt.read_text())
        return {"model": data["model"], "batch": data["batch"], "epochs": data["epochs"], "seed": data["seed"], "loop": data.get("loop"), "patches": data.get("patches"), "patch_kwargs": data.get("patch_kwargs")}
    if baseline.is_file():
        data = json.loads(baseline.read_text())
        return {"model": data["model"], "batch": data["batch_size"], "epochs": data["epochs"], "seed": data["seed"]}
    if standard.is_file():
        data = json.loads(standard.read_text())
        return {"model": data["model"].split("/")[-1].split("-")[0], "batch": data["batch_size"], "epochs": data["epochs"], "seed": data.get("seed")}
    raise FileNotFoundError(f"No training metadata in {directory}")


def _group(directories: list[Path], model: str, epochs: int) -> list[dict]:
    if len(directories) != 2:
        raise ValueError("Exactly two seed run directories are required")
    runs = []
    for directory in directories:
        metadata = _metadata(directory)
        if metadata["model"] != model or metadata["batch"] != B_AUTO[model] or metadata["epochs"] != epochs:
            raise ValueError(f"Budget mismatch in {directory}: {metadata}")
        metrics = read_lightly_metrics(directory)
        runs.append({"directory": str(directory), "metadata": metadata, "metrics": metrics})
    if len({run["metadata"]["seed"] for run in runs}) != 2:
        raise ValueError("Two distinct seeds are required")
    return sorted(runs, key=lambda item: item["metadata"]["seed"])


def compare(model: str, baseline_dirs: list[Path], final_dirs: list[Path], *, epochs: int = 20) -> dict:
    if model not in B_AUTO or B_AUTO[model] is None:
        raise ValueError(f"Unknown or unresolved batch for {model}")
    baseline = _group(baseline_dirs, model, epochs)
    final = _group(final_dirs, model, epochs)
    baseline_seeds = [item["metadata"]["seed"] for item in baseline]
    final_seeds = [item["metadata"]["seed"] for item in final]
    if baseline_seeds != final_seeds:
        raise ValueError(f"Seed sets differ: baseline {baseline_seeds}, final {final_seeds}")
    if any(not item["metadata"].get("patches") for item in final):
        raise ValueError("Each final run must declare an optimized patch stack")
    selected = {(tuple(item["metadata"].get("patches") or ()), json.dumps(item["metadata"].get("patch_kwargs") or {}, sort_keys=True)) for item in final}
    if len(selected) != 1:
        raise ValueError("Final seeds used different patch stacks or variant options")
    def aggregate(items: list[dict]) -> dict[str, float]:
        return {key: statistics.fmean(item["metrics"][key] for item in items) for key in ("miou", "head_iou")}

    baseline_mean, final_mean = aggregate(baseline), aggregate(final)
    gate = gate_d(final_mean, baseline_mean)
    report = {
        "model": model, "batch": B_AUTO[model], "budget_epochs": epochs,
        "seeds": baseline_seeds, "baseline": baseline, "final": final,
        "baseline_mean": baseline_mean, "final_mean": final_mean,
        "baseline_std": {key: statistics.pstdev(item["metrics"][key] for item in baseline) for key in ("miou", "head_iou")},
        "final_std": {key: statistics.pstdev(item["metrics"][key] for item in final) for key in ("miou", "head_iou")},
        "gate": gate,
    }
    return report


def _curve(directory: Path) -> dict[str, list[float]]:
    metrics_path = directory / "metrics.jsonl"
    if metrics_path.is_file():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        losses = []
        loss_steps = []
        for row in rows:
            train = row.get("train_metrics", {})
            loss = next((train[key] for key in ("train_loss", "train_loss/loss", "loss") if key in train), None)
            if loss is not None:
                losses.append(float(loss))
                loss_steps.append(int(row.get("step", len(loss_steps))) + 1)
        miou_rows = [row for row in rows if row.get("val_miou") is not None]
        head_rows = [row for row in rows if row.get("val_head_iou") is not None]
        return {
            "miou": [float(row["val_miou"]) for row in miou_rows],
            "miou_steps": [int(row.get("step", i)) + 1 for i, row in enumerate(miou_rows)],
            "head_iou": [float(row["val_head_iou"]) for row in head_rows],
            "head_iou_steps": [int(row.get("step", i)) + 1 for i, row in enumerate(head_rows)],
            "loss": losses, "loss_steps": loss_steps,
        }
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    result: dict[str, list[float]] = {"miou": [], "head_iou": [], "loss": [], "miou_steps": [], "head_iou_steps": [], "loss_steps": []}
    tags = {"miou": "val_metric/miou", "head_iou": "val_metric_classwise/iou_head", "loss": "train_loss"}
    for event_file in sorted(directory.rglob("events.out.tfevents*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", ()))
        if tags["loss"] not in available:
            tags["loss"] = next((tag for tag in available if tag.startswith("train_loss")), tags["loss"])
        for key, tag in tags.items():
            if tag in available:
                events = accumulator.Scalars(tag)
                result[key].extend(float(event.value) for event in events)
                result[key + "_steps"].extend(int(event.step) for event in events)
    return result


def _plot_curves(report: dict, output: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    any_data = False
    for group_name, linestyle in (("baseline", "--"), ("final", "-")):
        for run in report[group_name]:
            curve = _curve(Path(run["directory"]))
            seed = run["metadata"]["seed"]
            for axis, key, title in zip(axes, ("miou", "head_iou", "loss"), ("val mIoU", "val head IoU", "train loss")):
                values = curve.get(key, [])
                if values:
                    any_data = True
                    x = curve.get(key + "_steps") or list(range(1, len(values) + 1))
                    axis.plot(x, values, linestyle=linestyle, label=f"{group_name} s{seed}")
                axis.set(xlabel="training step", title=title)
    if any_data:
        for axis in axes:
            if axis.lines:
                axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output, dpi=145)
    plt.close(fig)
    return output if any_data else None


def _write_average_curves(report: dict, output: Path) -> list[Path]:
    generated = []
    for group_name in ("baseline", "final"):
        first, second = (_curve(Path(item["directory"])) for item in report[group_name])
        for metric, output_key in (("miou", "val_miou"), ("loss", "loss_ema")):
            a, b = first.get(metric, []), second.get(metric, [])
            x_a, x_b = first.get(metric + "_steps", []), second.get(metric + "_steps", [])
            if not a or len(a) != len(b) or x_a != x_b:
                continue
            values = [statistics.fmean(pair) for pair in zip(a, b)]
            if metric == "loss":
                ema = []
                for item in values:
                    ema.append(item if not ema else 0.9 * ema[-1] + 0.1 * item)
                values = ema
            path = output.with_name(f"curve_{group_name}_{metric}_{output.stem}.json")
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps({"model": report["model"], "group": group_name, "steps": x_a, output_key: values}, indent=2) + "\n")
            generated.append(path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(B_AUTO), required=True)
    parser.add_argument("--baseline-dir", type=Path, action="append", required=True)
    parser.add_argument("--final-dir", type=Path, action="append", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(args.model, args.baseline_dir, args.final_dir, epochs=args.epochs)
    output = args.out or ROUND / "profile" / f"compare_{args.model}_e{args.epochs}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    report["command"] = [sys.executable, *sys.argv]
    curve_path = output.with_name(output.stem + "_curves.png")
    if curve_path.exists():
        raise FileExistsError(curve_path)
    plotted = _plot_curves(report, curve_path)
    report["curves"] = str(plotted) if plotted is not None else None
    report["curve_json"] = [str(path) for path in _write_average_curves(report, output)]
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    gate = report["gate"]
    selected = report["final"][0]["metadata"]
    write_results(ROUND / "results.json", {args.model: {"final": {
        "batch": B_AUTO[args.model], "patches": selected.get("patches") or [],
        "accuracy": {"level_d": {
            "budget_epochs": args.epochs,
            "val_miou": gate["val_miou"], "val_head_iou": gate["val_head_iou"],
            "ratio_vs_baseline": gate["ratio_vs_baseline"], "pass": gate["pass"],
        }},
        "status": "passed" if gate["pass"] else "failed",
        "logs": [str(output)],
    }}})
    print(json.dumps({"pass": gate["pass"], "ratio_vs_baseline": gate["ratio_vs_baseline"], "log": str(output)}, indent=2))


if __name__ == "__main__":
    main()
