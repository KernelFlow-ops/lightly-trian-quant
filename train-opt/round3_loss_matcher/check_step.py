"""Run the common B-level gate for the round3 patch on one real batch."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.bench import empty_result, env_info, gpu_lock, write_results
from common.check import step_equivalence
from common.config import B_AUTO
from common.harness import RunCfg, build, make_loader
from common.rng import RngTape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("vitl16", "vitb16"), required=True)
    parser.add_argument("--variant", choices=("baseline_control", "batched_torch", "fused", "cuda"), required=True)
    parser.add_argument("--phase", choices=("early", "mid", "late"), default="mid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = B_AUTO[args.model]
    if batch is None:
        raise RuntimeError("B_AUTO must be fixed before B-level checking")
    cfg_ref = RunCfg(model=args.model, batch_size=batch, phase=args.phase, num_workers=4)
    cfg_opt = (
        RunCfg(model=args.model, batch_size=batch, phase=args.phase, num_workers=4)
        if args.variant == "baseline_control"
        else RunCfg(
            model=args.model, batch_size=batch, phase=args.phase, num_workers=4,
            patches=("round3_loss_matcher",),
            patch_kwargs={"round3_loss_matcher": {"variant": args.variant}},
        )
    )
    with gpu_lock():
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
            text=True,
        )
        if len(apps.strip().splitlines()) > 1:
            raise RuntimeError(f"Other GPU process active; B-level check canceled:\n{apps}")
        loader_ctx = build(cfg_ref)
        batch_data = next(iter(make_loader(loader_ctx, "train")))
        del loader_ctx
        gc.collect()
        torch.cuda.empty_cache()
        result = step_equivalence(
            cfg_ref, cfg_opt, [batch_data], tape=RngTape(mode="record")
        )
        result.update(model=args.model, variant=args.variant, phase=args.phase,
                      batch=batch, compute_apps=apps.strip())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.variant != "baseline_control":
            row = empty_result(
                round_name="round3_loss_matcher", variant=args.variant,
                model=args.model, batch=batch, precision=cfg_ref.precision,
                patches=cfg_opt.patches,
                desc="Batched GPU matcher and point-sampled loss; original Fabric fp32 forward",
            )
            row["accuracy"]["level_b"] = {
                key: result[key] for key in (
                    "loss_rel", "grad_cos_min", "grad_norm_rel",
                    "update_cos_min", "pass"
                )
            }
            row["env"] = env_info()
            row["logs"] = [str(args.output.resolve())]
            write_results(
                Path(__file__).resolve().parent / "results.json",
                {args.model: {args.variant: row}},
            )
        print(json.dumps({
            key: result[key]
            for key in ("model", "variant", "phase", "batch", "loss_rel",
                        "grad_cos_min", "grad_norm_rel", "update_cos_min", "pass")
        }, indent=2))
        if not result["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
