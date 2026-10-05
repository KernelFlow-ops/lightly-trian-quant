"""Locked GEMM probes at the actual forward, dgrad, and wgrad train shapes."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean

import torch

from .bench import assert_gpu_exclusive, env_info, gpu_lock, write_results
from .config import B_AUTO, MODEL_SHAPES, TRAIN_OPT_ROOT


def _matrices(m: int, n: int, k: int, dtype: str):
    source_dtype = torch.float32 if dtype == "f32_tf32" else torch.bfloat16
    a = torch.randn((m, k), device="cuda", dtype=source_dtype)
    b = torch.randn((k, n), device="cuda", dtype=source_dtype)
    if dtype != "fp8_e4m3":
        return a, b, lambda: a @ b
    a = a.to(torch.float8_e4m3fn)
    b = b.to(torch.float8_e4m3fn)
    scale_a = torch.ones((), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((), device="cuda", dtype=torch.float32)
    return a, b, lambda: torch._scaled_mm(
        a, b, scale_a=scale_a, scale_b=scale_b,
        out_dtype=torch.bfloat16, use_fast_accum=False,
    )


def _time(fn, *, warmup: int = 10, iters: int = 30, repeats: int = 3) -> tuple[float, list[float]]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / iters)
    return mean(samples), samples


def _probe(m: int, n: int, k: int, dtype: str) -> dict:
    try:
        _, _, fn = _matrices(m, n, k, dtype)
        step_ms, repeats = _time(fn)
        flops = 2 * m * n * k
        return {
            "shape": [m, n, k], "step_ms": step_ms,
            "repeats_ms": repeats, "tflops": flops / (step_ms / 1000) / 1e12,
            "error": None,
        }
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        torch.cuda.empty_cache()
        return {"shape": [m, n, k], "step_ms": None, "repeats_ms": [], "tflops": None, "error": str(exc)}


def run(model: str, dtype: str) -> dict:
    if model not in MODEL_SHAPES or dtype not in {"f32_tf32", "bf16", "fp8_e4m3"}:
        raise ValueError(f"Unsupported model/dtype: {model}, {dtype}")
    batch = B_AUTO[model]
    if batch is None:
        raise RuntimeError(f"B_AUTO[{model}] is unset")
    d = MODEL_SHAPES[model]["dim"]
    torch.set_float32_matmul_precision("high")
    result = {"batch": batch, "dtype": dtype, "shapes": {}, "env": None}
    with gpu_lock():
        assert_gpu_exclusive()
        for tokens in (1029, 1129):
            rows = batch * tokens
            for name, in_features, out_features in (
                ("qkv", d, 3 * d), ("fc1", d, 4 * d), ("fc2", 4 * d, d)
            ):
                for phase, m, n, k in (
                    ("fwd", rows, out_features, in_features),
                    ("dgrad", rows, in_features, out_features),
                    ("wgrad", out_features, in_features, rows),
                ):
                    padded_k = math.ceil(k / 16) * 16 if dtype == "fp8_e4m3" and phase == "wgrad" else k
                    measurement = _probe(m, n, padded_k, dtype)
                    measurement["logical_shape"] = [m, n, k]
                    measurement["padding_ratio"] = padded_k / k
                    if measurement["tflops"] is not None:
                        measurement["logical_tflops"] = measurement["tflops"] * k / padded_k
                    else:
                        measurement["logical_tflops"] = None
                    result["shapes"][f"{name}_{tokens}_{phase}"] = measurement
        result["env"] = env_info()
        assert_gpu_exclusive()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_SHAPES), required=True)
    parser.add_argument("--dtype", choices=("f32_tf32", "bf16", "fp8_e4m3"), required=True)
    args = parser.parse_args()
    result = run(args.model, args.dtype)
    write_results(TRAIN_OPT_ROOT / "results" / "peaks_train.json", {args.model: {args.dtype: result}})
    supported = [entry for entry in result["shapes"].values() if entry["tflops"] is not None]
    print(f"{args.model} {args.dtype}: {len(supported)}/{len(result['shapes'])} shapes, max {max((entry['tflops'] for entry in supported), default=math.nan):.1f} TFLOPS")


if __name__ == "__main__":
    main()
