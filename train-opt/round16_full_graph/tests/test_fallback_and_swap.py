"""Graph runner edge cases on a real model: (1) a batch with more targets than the static capacity falls back to an eager step while the
graph pool is resident (memory!), (2) graphs are re-used when the phase returns (no recapture), (3) LRU eviction with max_resident=1.
Usage: python test_fallback_and_swap.py [model]"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import importlib.util  # noqa: E402

from common import bench  # noqa: E402
from common.harness import apply_patches, build, preload_batches, set_phase  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round16_full_graph.impl.graph_step import StepGraphRunner  # noqa: E402

spec = importlib.util.spec_from_file_location("r16run", str(ROOT / "round16_full_graph" / "run.py"))
r16run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r16run)

model = sys.argv[1] if len(sys.argv) > 1 else "vitb16"
out = {"model": model}
with bench.gpu_lock():
    r16run.wait_exclusive()
    ctx = build(r16run.make_cfg(model, "P2,r15"))
    apply_patches(ctx)
    ctx.rng = RngTape(mode="off")
    batches = preload_batches(ctx, k=2)
    runner = StepGraphRunner(ctx, batches[0], max_targets=2, max_resident=1)
    ctx.meta["phase_override"] = True
    set_phase(ctx, "mid")
    runner.prepare(batches[0])
    losses = []
    for i in range(3):
        losses.append(float(runner.step(batches[i % 2], i, batches[(i + 1) % 2]).loss))
    out["captures_after_3_steps"] = runner.report.captures
    # (1) overflow batch -> eager fallback while the graph pool is resident
    b = dict(batches[0])
    bm = [dict(x) for x in b["binary_masks"]]
    bm[0] = {"masks": torch.cat([bm[0]["masks"], bm[0]["masks"][:1]], 0), "labels": torch.cat([bm[0]["labels"], bm[0]["labels"][:1]])}
    b["binary_masks"] = bm
    torch.cuda.synchronize()
    out["reserved_gb_before_fallback"] = torch.cuda.memory_reserved() / 2**30
    try:
        fb = runner.step(b, 3)
        out["fallback_loss"] = float(fb.loss)
        out["fallback_ok"] = True
    except torch.OutOfMemoryError as exc:
        out["fallback_ok"] = False
        out["fallback_error"] = f"OutOfMemoryError: {str(exc).splitlines()[0][:160]}"
    out["eager_fallbacks"] = runner.report.eager_fallbacks
    out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    # graph still works afterwards (same signature, no recapture)
    c0 = runner.report.captures
    after = float(runner.step(batches[0], 4, batches[1]).loss)
    out["graph_after_fallback_loss"] = after
    out["recaptured_after_fallback"] = runner.report.captures - c0
    # (3) LRU eviction (max_resident=1): switch phase twice
    set_phase(ctx, "late")
    runner.sync_probs_state()
    float(runner.step(batches[0], 5, batches[1]).loss)
    out["captures_after_phase_change"] = runner.report.captures
    out["evictions"] = runner.report.evictions
    set_phase(ctx, "mid")
    runner.sync_probs_state()
    float(runner.step(batches[0], 6, batches[1]).loss)
    out["captures_after_return"] = runner.report.captures      # max_resident=1: the mid graph was evicted -> recapture (2 -> 3)
    out["reserved_gb_end"] = torch.cuda.memory_reserved() / 2**30
    out["losses_finite"] = all(torch.isfinite(torch.tensor(losses)))
    out["report"] = runner.report.as_dict()
print(json.dumps(out, indent=1, default=str))
Path(__file__).resolve().parents[1].joinpath("profile", f"fallback_and_swap_{model}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
