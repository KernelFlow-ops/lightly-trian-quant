"""Analytical DINOv3-EoMT training FLOPs and measured-kernel roofline helper.

The operation inventory follows infer-opt/common/roofline.py and PLAN's train
accounting (forward, dgrad, and wgrad separately). Kernel-time attribution uses
the nsys kernel summary; shapes absent from a profile remain explicitly unknown.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Op:
    name: str
    flops: float
    min_bytes: float
    count: int = 1
    phase: str = "forward"


CONFIGS = {"vitl16": {"dim": 1024, "depth": 24, "joint": 3}, "vitb16": {"dim": 768, "depth": 12, "joint": 3}}


def training_ops(model: str, batch: int, *, image: int = 512, patch: int = 16, queries: int = 100, bytes_per_el: int = 2) -> list[Op]:
    if model not in CONFIGS:
        raise ValueError(f"unknown model {model}")
    if batch <= 0 or image % patch:
        raise ValueError("batch and image/patch geometry must be valid")
    d = CONFIGS[model]["dim"]
    depth = CONFIGS[model]["depth"]
    joint = CONFIGS[model]["joint"]
    grid = image // patch
    plain_tokens = grid * grid + 5
    joint_tokens = plain_tokens + queries
    result = [Op("patch_embed", 2 * batch * grid**2 * (3 * patch**2) * d, bytes_per_el * batch * (3 * image**2 + 3 * patch**2 * d + grid**2 * d))]
    for tag, tokens, count in (("block", plain_tokens, depth - joint), ("joint_block", joint_tokens, joint)):
        # Attention and projections are separate because nsys attribution maps
        # them to distinct kernels; backward GEMMs are shown as an approximate
        # 2x-forward GEMM workload until exact per-kernel shapes are exported.
        gemm_flops = 2 * batch * tokens * d * d * (3 + 1 + 4 + 4)
        attn_flops = 4 * batch * tokens * tokens * d
        gemm_bytes = bytes_per_el * batch * (tokens * d + (3 + 1 + 4 + 4) * d * d)
        attn_bytes = bytes_per_el * batch * 4 * tokens * d
        for phase, multiplier in (("forward", 1), ("dgrad", 1), ("wgrad", 1)):
            result.append(Op(f"{tag}.gemm.{phase}", gemm_flops * multiplier, gemm_bytes, count, phase))
        result.append(Op(f"{tag}.attention.forward", attn_flops, attn_bytes, count, "forward"))
        result.append(Op(f"{tag}.attention.backward", attn_flops * 2, attn_bytes * 2, count, "backward"))
        result.append(Op(f"{tag}.elementwise", 0, bytes_per_el * batch * 12 * tokens * d, count, "mixed"))
    # Four prediction calls: three joint outputs plus the final output.
    pred_count = joint + 1
    class_head = 2 * batch * queries * d * (2 + 1)
    mask_head = 2 * batch * queries * d * d * 3
    scale_flops = sum(2 * batch * (grid * 2**(i + 1)) ** 2 * (d * d + 9 * d) for i in range(2))
    mask_grid = grid * 4
    einsum = 2 * batch * queries * d * mask_grid**2
    result += [
        Op("head.class_head.forward", class_head, bytes_per_el * batch * queries * (d + 3), pred_count),
        Op("head.mask_mlp.forward", mask_head, bytes_per_el * batch * queries * (4 * d + 3 * d * d), pred_count),
        Op("head.upscale.forward", scale_flops, bytes_per_el * batch * mask_grid**2 * d * 2, pred_count),
        Op("head.einsum.forward", einsum, bytes_per_el * batch * (queries * d + d * mask_grid**2 + queries * mask_grid**2), pred_count),
        Op("head.backward", (class_head + mask_head + scale_flops + einsum) * 2, bytes_per_el * batch * mask_grid**2 * d * 2, pred_count, "backward"),
    ]
    return result


def summarize(model: str, batch: int, *, peak_tflops: float | None = None, peak_gbps: float | None = None) -> dict[str, Any]:
    ops = training_ops(model, batch)
    total_flops = sum(op.flops * op.count for op in ops)
    total_bytes = sum(op.min_bytes * op.count for op in ops)
    return {
        "model": model, "batch": batch, "ops": [asdict(op) for op in ops],
        "total_gflop_per_step": total_flops / 1e9,
        "total_gflop_per_image": total_flops / batch / 1e9,
        "minimum_gb_per_step": total_bytes / 1e9,
        "peak_tflops": peak_tflops, "peak_gbps": peak_gbps,
    }


def match_profile(prefix: str, model: str, batch: int, *, peak_tflops: float | None = None, peak_gbps: float | None = None, iterations: int = 5, shape_map: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    profile = Path(f"{prefix}_cuda_gpu_kern_sum.csv")
    with profile.open(newline="") as stream:
        kernels = list(csv.DictReader(stream))
    # nsys aggregates by kernel symbol, not tensor shape. Avoid spreading the
    # model's total work over arbitrary kernels; unknown mappings stay null.
    profiles = []
    for row in kernels:
        symbol = row.get("Name", "")
        matches = [entry for entry in shape_map or [] if entry["pattern"] in symbol]
        elapsed_ms = float(row.get("Total Time (ns)", 0)) / iterations / 1e6
        item: dict[str, Any] = {"kernel": symbol, "instances": int(float(row.get("Instances", 0))), "measured_total_ms": elapsed_ms, "shape": None, "flops": None, "bytes": None, "tflops": None, "gbps": None, "pct_peak_compute": None, "pct_peak_bw": None}
        if len(matches) == 1 and elapsed_ms > 0:
            shape = matches[0]
            m, n, k = (int(shape[axis]) for axis in ("m", "n", "k"))
            count = float(shape.get("count_per_step", item["instances"] / iterations))
            element_bytes = int(shape.get("bytes_per_el", 2))
            flops = 2 * m * n * k * count
            min_bytes = element_bytes * (m * k + k * n + m * n) * count
            item.update(shape={key: shape.get(key) for key in ("m", "n", "k", "phase")}, flops=flops, bytes=min_bytes,
                        tflops=flops / (elapsed_ms * 1e-3) / 1e12,
                        gbps=min_bytes / (elapsed_ms * 1e-3) / 1e9)
            item["pct_peak_compute"] = 100 * item["tflops"] / peak_tflops if peak_tflops else None
            item["pct_peak_bw"] = 100 * item["gbps"] / peak_gbps if peak_gbps else None
        profiles.append(item)
    data = summarize(model, batch, peak_tflops=peak_tflops, peak_gbps=peak_gbps)
    data["kernel_profile"] = profiles
    data["mapping_note"] = "Kernel symbols alone do not encode problem shape. A unique shape-map entry is required for achieved per-kernel TFLOPS/GB/s; unmatched rows stay null."
    Path(f"{prefix}_roofline.json").write_text(json.dumps(data, indent=2))
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=CONFIGS)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--profile-prefix")
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--peak-gbps", type=float)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--shape-map", type=Path, help="JSON list of {pattern,m,n,k,phase,count_per_step,bytes_per_el} mappings")
    args = parser.parse_args()
    result = summarize(args.model, args.batch, peak_tflops=args.peak_tflops, peak_gbps=args.peak_gbps)
    if args.profile_prefix:
        mapping = json.loads(args.shape_map.read_text()) if args.shape_map else None
        match_profile(args.profile_prefix, args.model, args.batch, peak_tflops=args.peak_tflops, peak_gbps=args.peak_gbps, iterations=args.iterations, shape_map=mapping)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
