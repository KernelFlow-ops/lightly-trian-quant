"""Run several B-level phase checks under one dual-GPU lock."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock, write_results


def combine(existing: dict | None, latest: dict) -> dict:
    if existing is None or existing.get("pass") is None:
        return {key: latest[key] for key in (
            "loss_rel", "grad_cos_min", "grad_norm_rel", "update_cos_min", "pass"
        )}
    return {
        "loss_rel": max(existing["loss_rel"], latest["loss_rel"]),
        "grad_cos_min": min(existing["grad_cos_min"], latest["grad_cos_min"]),
        "grad_norm_rel": max(existing["grad_norm_rel"], latest["grad_norm_rel"]),
        "update_cos_min": min(existing["update_cos_min"], latest["update_cos_min"]),
        "pass": bool(existing["pass"] and latest["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("batched_torch", "fused", "cuda"), required=True)
    parser.add_argument("--cases", nargs="+", required=True,
                        help="model:phase, such as vitb16:early")
    args = parser.parse_args()
    cases = []
    for item in args.cases:
        model, phase = item.split(":", 1)
        if model not in ("vitl16", "vitb16") or phase not in ("early", "mid", "late"):
            raise ValueError(f"Invalid case: {item}")
        cases.append((model, phase))
    round_dir = Path(__file__).resolve().parent
    results_path = round_dir / "results.json"
    prior = json.loads(results_path.read_text()) if results_path.exists() else {}
    aggregate: dict[str, dict] = {}
    logs: dict[str, list[str]] = {}
    for model, _ in cases:
        if model not in aggregate:
            old = prior.get(model, {}).get(args.variant, {})
            aggregate[model] = old.get("accuracy", {}).get("level_b", {})
            logs[model] = list(old.get("logs", []))
    failed = False
    with gpu_lock():
        for model, phase in cases:
            output = round_dir / "profile" / f"check_{model}_{args.variant}_{phase}.json"
            log = Path("/root/autodl-tmp/train-opt-tmp/C") / f"r3_{model}_{args.variant}_{phase}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(round_dir / "check_step.py"),
                "--model", model, "--variant", args.variant,
                "--phase", phase, "--output", str(output),
            ]
            with log.open("w") as stream:
                completed = subprocess.run(
                    cmd, cwd=ROOT.parent, env=os.environ.copy(),
                    stdout=stream, stderr=subprocess.STDOUT,
                )
            if not output.exists():
                raise RuntimeError(f"Check produced no result; see {log}")
            result = json.loads(output.read_text())
            aggregate[model] = combine(aggregate[model], result)
            for path in (str(output), str(log)):
                if path not in logs[model]:
                    logs[model].append(path)
            print(json.dumps({
                "model": model, "phase": phase, "variant": args.variant,
                "loss_rel": result["loss_rel"],
                "grad_cos_min": result["grad_cos_min"],
                "grad_norm_rel": result["grad_norm_rel"],
                "update_cos_min": result["update_cos_min"],
                "pass": result["pass"], "log": str(log),
            }))
            if completed.returncode != 0 or not result["pass"]:
                failed = True
        for model in aggregate:
            write_results(results_path, {
                model: {args.variant: {
                    "accuracy": {"level_b": aggregate[model]},
                    "logs": logs[model],
                }}
            })
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
