"""Locate round3 real-batch divergence using identical logits and RNG points."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.bench import gpu_lock
from common.config import B_AUTO
from common.fast_forward import forward_train_v2
from common.harness import RunCfg, build, make_loader
from common.rng import RngTape

from round3_loss_matcher.impl.fast_criterion import FastMaskClassificationLoss


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        left.float().flatten(), right.float().flatten(), dim=0
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("vitl16", "vitb16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch_size = B_AUTO[args.model]
    if batch_size is None:
        raise RuntimeError("B_AUTO is unset")
    with gpu_lock():
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
            text=True,
        )
        if len(apps.strip().splitlines()) > 1:
            raise RuntimeError(f"Other GPU process active; diagnostic canceled:\n{apps}")
        ctx = build(RunCfg(model=args.model, batch_size=batch_size, num_workers=4))
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        batch = next(iter(make_loader(ctx, "train")))
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        with torch.no_grad():
            masks, classes = forward_train_v2(ctx, batch["image"])
        mask_ref = [value.detach().requires_grad_() for value in masks]
        class_ref = [value.detach().requires_grad_() for value in classes]
        mask_opt = [value.detach().clone().requires_grad_() for value in masks]
        class_opt = [value.detach().clone().requires_grad_() for value in classes]
        targets = batch["binary_masks"]
        criterion = ctx.train_model.criterion
        reference_indices = []
        reference_points = []
        original_match = criterion.matcher.forward
        original_uncertainty = criterion.sample_points_using_uncertainty

        def record_match(*values, **kwargs):
            result = original_match(*values, **kwargs)
            reference_indices.append(result)
            return result

        def record_points(*values, **kwargs):
            result = original_uncertainty(*values, **kwargs)
            reference_points.append(result.detach())
            return result

        criterion.matcher.forward = record_match
        criterion.sample_points_using_uncertainty = record_points
        tape = RngTape(mode="record")
        ctx.rng = tape
        try:
            ref_total, ref_losses = ctx.ops.loss_all_layers(
                ctx, mask_ref, class_ref, targets
            )
        finally:
            criterion.matcher.forward = original_match
            criterion.sample_points_using_uncertainty = original_uncertainty
        ref_grads = torch.autograd.grad(ref_total, mask_ref + class_ref)

        replay = RngTape(mode="replay", values=tape.values)
        fast = FastMaskClassificationLoss(
            criterion, num_blocks=len(ctx.model.backbone.blocks),
            variant="batched_torch",
        ).to(masks[0].device)
        fast.debug = True
        opt_total, opt_losses = fast.forward_all_layers(
            mask_opt, class_opt, targets,
            coord_provider=lambda name, shape, device: replay.rand(
                name, shape, device=device
            ),
        )
        replay.assert_complete()
        opt_grads = torch.autograd.grad(opt_total, mask_opt + class_opt)
        max_targets = max(target["masks"].shape[0] for target in targets)
        opt_points = fast.last_point_coordinates.reshape(
            len(masks), len(targets), max_targets, -1, 2
        )
        assignments = []
        coordinates = []
        for layer, per_image in enumerate(reference_indices):
            ref_point_offset = 0
            for image, (src, dst) in enumerate(per_image):
                count = targets[image]["masks"].shape[0]
                reference_by_target = [-1] * count
                for query, target in zip(src.tolist(), dst.tolist()):
                    reference_by_target[target] = query
                candidate = fast.last_assignment[layer, image, :count].tolist()
                if reference_by_target != candidate:
                    assignments.append({
                        "layer": layer, "image": image,
                        "reference": reference_by_target, "candidate": candidate,
                    })
                for slot in range(count):
                    expected = reference_points[layer][ref_point_offset + slot]
                    actual = opt_points[layer, image, slot]
                    if not torch.equal(expected, actual):
                        coordinates.append({
                            "layer": layer, "image": image, "slot": slot,
                            "max_abs": float((expected - actual).abs().max()),
                        })
                ref_point_offset += count
        loss_rel = {
            key: float((ref_losses[key] - opt_losses[key]).abs() / ref_losses[key].abs().clamp_min(1e-8))
            for key in ref_losses
        }
        result = {
            "model": args.model, "batch": batch_size,
            "loss_rel": loss_rel,
            "assignment_mismatches": assignments,
            "coordinate_mismatches": coordinates,
            "mask_grad_cos": [cosine(a, b) for a, b in zip(ref_grads[:len(masks)], opt_grads[:len(masks)])],
            "class_grad_cos": [cosine(a, b) for a, b in zip(ref_grads[len(masks):], opt_grads[len(masks):])],
            "compute_apps": apps.strip(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
