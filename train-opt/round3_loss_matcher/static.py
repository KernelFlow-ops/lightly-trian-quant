"""Compile round3 Triton kernels and dump resource usage and SASS."""
# ruff: noqa: E402

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock

from round3_loss_matcher.impl.point_loss import (
    _points_backward,
    _points_forward,
    _points_reduce,
)


def main() -> None:
    output = Path(__file__).resolve().parent / "profile" / "static"
    points, masks, pred_h, pred_w, target_h, target_w, block = (
        12544, 3, 128, 128, 512, 512, 256
    )
    chunks = triton.cdiv(points, block)
    with gpu_lock():
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
            text=True,
        )
        if len(apps.strip().splitlines()) > 1:
            raise RuntimeError(f"Other GPU process active; static compile canceled:\n{apps}")
        device = "cuda"
        pred = torch.empty((masks, pred_h, pred_w), device=device, dtype=torch.float32)
        target = torch.empty((masks, target_h, target_w), device=device, dtype=torch.uint8)
        coords = torch.empty((masks, points, 2), device=device)
        sampled_x = torch.empty((masks, points), device=device)
        sampled_y = torch.empty_like(sampled_x)
        partial = torch.empty((masks, chunks, 4), device=device)
        bce, dice, numerator, denominator = [
            torch.empty(masks, device=device) for _ in range(4)
        ]
        grad_pred = torch.empty_like(pred)
        kernels = {
            "point_forward": (
                _points_forward.warmup(
                    pred, target, coords, sampled_x, sampled_y, partial,
                    pred_h, pred_w, target_h, target_w, points, chunks, block,
                    grid=(chunks, masks),
                ), chunks * masks,
            ),
            "point_reduce": (
                _points_reduce.warmup(
                    partial, bce, dice, numerator, denominator,
                    points, chunks, triton.next_power_of_2(chunks),
                    grid=(masks,),
                ), masks,
            ),
            "point_backward": (
                _points_backward.warmup(
                    coords, sampled_x, sampled_y, numerator, denominator,
                    bce, dice, grad_pred, pred_h, pred_w, points, block,
                    grid=(chunks, masks),
                ), chunks * masks,
            ),
        }
        output.mkdir(parents=True, exist_ok=True)
        report = {}
        for name, (kernel, total_ctas) in kernels.items():
            cubin = kernel.asm.get("cubin")
            if cubin is None:
                raise RuntimeError(f"Triton did not provide cubin for {name}")
            path = output / f"{name}.cubin"
            path.write_bytes(cubin)
            subprocess.run(
                ["bash", str(ROOT / "common" / "kernel_static.sh"),
                 "bin", str(path), str(output)],
                check=True,
            )
            metadata = kernel.metadata
            report[name] = {
                "registers_per_thread": getattr(kernel, "n_regs", None),
                "spills": getattr(kernel, "n_spills", None),
                "shared_memory_bytes": getattr(metadata, "shared", None),
                "warps": getattr(metadata, "num_warps", None),
                "ctas_for_three_masks": total_ctas,
                "active_ctas_per_sm": None,
                "waves_per_170_sm": None,
            }
        (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
