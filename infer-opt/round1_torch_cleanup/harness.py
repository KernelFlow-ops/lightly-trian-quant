"""Shared measurement harness for rounds 1/2/4 (same caliber as round0_baseline/run.py).

model stage = model_fn(batch) with batch = baseline prep output (fp32 (2,3,512,512));
e2e         = baseline prep + model_fn + baseline post (host copy + numpy argmax).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import model_factory as mf  # noqa: E402
from bench import env_info, peak_mem_mb, time_fn, write_results  # noqa: E402


def count_cuda_ops(fn) -> dict[str, int]:
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
    memcpy = sum(1 for n in names if "Memcpy" in n or "memcpy" in n)
    memset = sum(1 for n in names if "Memset" in n or "memset" in n)
    return {"kernels": len(names) - memcpy - memset, "memcpy": memcpy, "memset": memset}


def accuracy(model_name: str, ref_model, model_fn: Callable) -> dict:
    out = {}
    for i, frame in enumerate(mf.load_frames()):
        # prep/post run exactly like the baseline (bf16 autocast); OUR model runs outside autocast.
        with torch.inference_mode(), mf.autocast_ctx("bf16-mixed"):
            batch, meta = mf.base_prep(ref_model, frame)
        with torch.inference_mode():
            raw = model_fn(batch)
        with torch.inference_mode(), mf.autocast_ctx("bf16-mixed"):
            logits = mf.base_post_logits(ref_model, raw, meta)
        mask, _, conf = mf.host_finalize(ref_model, logits.float().cpu().numpy())
        out[f"frame{i}"] = mf.compare_to_ref(model_name, i, logits, mask, conf)
    out["gate"] = mf.gate(out)
    return out


def bench_variant(model_name: str, variant: str, ref_model, model_fn: Callable,
                  out_json: Path, *, iters: int = 100, warmup: int = 20,
                  extra: dict | None = None, extra_fns: dict[str, Callable] | None = None,
                  prep_ctx: str = "bf16-mixed") -> dict:
    """ref_model: fp32 lightly model (used for the baseline prep/post only)."""
    frame = mf.load_frames()[0]
    with torch.inference_mode(), mf.autocast_ctx(prep_ctx):
        batch, meta = mf.base_prep(ref_model, frame)
    with torch.inference_mode():
        for _ in range(3):
            raw = model_fn(batch)
    torch.cuda.synchronize()

    def model_only():
        return model_fn(batch)

    def e2e():
        with torch.inference_mode():
            with mf.autocast_ctx(prep_ctx):
                b, mt = mf.base_prep(ref_model, frame)
            r = model_fn(b)  # our model: no autocast
            with mf.autocast_ctx(prep_ctx):
                return mf.base_post(ref_model, r, mt)

    fns = {"model": model_only, "e2e": e2e}
    if extra_fns:
        fns.update(extra_fns)
    torch.cuda.reset_peak_memory_stats()
    res = {"batch_shape": list(batch.shape), "latency_ms": {}, "cuda_ops": {}}
    for name, fn in fns.items():
        t = time_fn(fn, warmup=warmup, iters=iters)
        # wall_p50 = median over repeats of the per-repeat p50 (round-0 convention); wall_p50_min = best repeat
        # (robust to other workers' un-locked correctness runs disturbing a repeat).
        res["latency_ms"][name] = {"wall_p50": t["wall_ms_p50"], "gpu_p50": t["gpu_ms_p50"],
                                   "wall_p50_min": min(r["wall_ms"]["p50"] for r in t["runs"]),
                                   "runs": t["runs"]}
        print(f"[{model_name}/{variant}] {name:12s} wall p50 {t['wall_ms_p50']:8.3f} ms "
              f"| gpu p50 {t['gpu_ms_p50']:8.3f} ms | best-repeat wall p50 "
              f"{res['latency_ms'][name]['wall_p50_min']:8.3f} ms", flush=True)
    with torch.inference_mode():
        for name in fns:
            res["cuda_ops"][name] = count_cuda_ops(fns[name])
    res["peak_mem_mb"] = peak_mem_mb()
    res["accuracy_vs_fp32"] = accuracy(model_name, ref_model, model_fn)
    print(f"[{model_name}/{variant}] cuda_ops model {res['cuda_ops']['model']} "
          f"gate {res['accuracy_vs_fp32']['gate']}", flush=True)
    if extra:
        res.update(extra)
    write_results(out_json, {model_name: {variant: res}, "env": env_info()})
    return res
