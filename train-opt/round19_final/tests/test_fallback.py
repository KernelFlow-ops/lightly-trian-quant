"""Over-capacity batch on the FINAL stack under the graph runner (round16 notes: the eager fallback may OOM on ViT-L while the graph pool is resident).
A batch with 3 target masks in one image (> max_targets=2) is pushed through ``runner.step``; reports fallback count, re-capture, peak reserved, OOM.
  python tests/test_fallback.py --model vitl16 --out profile/fallback_vitl16.json"""
import argparse
import json
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND))
sys.path.insert(0, str(ROUND.parent))
import torch  # noqa: E402

import run as _r19run  # noqa: E402,F401
from round16_full_graph import run as r16  # noqa: E402
from common import bench  # noqa: E402
from common.harness import apply_patches, build, preload_batches  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vitb16")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
res = {"model": args.model, "oom": False}
with bench.gpu_lock():
    r16.wait_exclusive()
    cfg = r16.make_cfg(args.model, "FINAL", seed=0)
    ctx = build(cfg)
    apply_patches(ctx)
    import run_final as rf
    batches = preload_batches(ctx, k=2)
    runner = rf.make_graph_runner(ctx, batches[0], run_steps=10)
    for s in range(3):
        runner.step(batches[s % 2], s, None)
    torch.cuda.synchronize()
    res["reserved_gb_graph_resident"] = torch.cuda.memory_reserved() / 2**30
    big = dict(batches[1])
    items = [dict(x) for x in big["binary_masks"]]
    m = items[0]["masks"]
    items[0] = {"masks": torch.cat([m, m[:1], m[:1]]), "labels": torch.cat([items[0]["labels"], items[0]["labels"][:1], items[0]["labels"][:1]])}
    big["binary_masks"] = items
    res["targets_in_big_batch"] = [int(x["masks"].shape[0]) for x in items]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        out = runner.step(big, 3, None)
        torch.cuda.synchronize()
        res["fallback_loss"] = float(out.loss)
        res["fallback_loss_finite"] = bool(torch.isfinite(out.loss).item())
    except torch.OutOfMemoryError as exc:
        res["oom"] = True
        res["error"] = str(exc)[:300]
    res["fallback_s"] = time.time() - t0
    res["peak_reserved_gb_during_fallback"] = torch.cuda.max_memory_reserved() / 2**30
    if not res["oom"]:
        before = runner.report.captures
        out = runner.step(batches[0], 4, None)
        torch.cuda.synchronize()
        res["loss_after_fallback"] = float(out.loss)
        res["recaptured_after_fallback"] = runner.report.captures - before
    res["report"] = runner.report.as_dict()
args.out.write_text(json.dumps(res, indent=2, default=str))
print(json.dumps({k: res[k] for k in ("model", "oom", "reserved_gb_graph_resident", "peak_reserved_gb_during_fallback", "fallback_s")}, default=str))
print(json.dumps({k: res.get(k) for k in ("fallback_loss", "fallback_loss_finite", "loss_after_fallback", "recaptured_after_fallback")}), json.dumps({k: res["report"][k] for k in ("captures", "eager_fallbacks", "replays")}))
