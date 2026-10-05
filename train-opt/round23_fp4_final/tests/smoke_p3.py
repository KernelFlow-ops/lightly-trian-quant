"""Smoke test of the combined stack P3 (round21 head + round22 backbone on top of FINAL): the two patches coexist, ``ctx.ops.opt_step`` wraps in the expected order, the
StepGraphRunner warms / captures / replays, and BOTH FP4States advance exactly once per step (graph replay and eager) with finite losses.

  python tests/smoke_p3.py --model vitb16 --steps 6 --out profile/smoke_vitb16_P3.json  [--stack P3] [--q2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["QUACK_CACHE_DIR"] = "/root/autodl-tmp/train-opt-tmp/round23_fp4_final/quack_cache"
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))

from round23_fp4_final.impl import presets23 as p23  # noqa: E402

p23.maybe_q2()

import torch  # noqa: E402

from common import bench  # noqa: E402
from common.harness import apply_patches, build, preload_batches, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round19_final import run as r19run  # noqa: E402,F401
from round16_full_graph import run as r16  # noqa: E402
from round16_full_graph.impl.graph_step import padded_targets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stack", default="P3")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--q2", action="store_true")
    args = ap.parse_args()
    p23.register(ROUND, args.model)
    cfg = r16.make_cfg(args.model, args.stack, seed=0)
    rep: dict = {"command": [sys.executable, *sys.argv], "model": args.model, "stack": args.stack, "q2": p23.Q2_ACTIVE, "patches": list(cfg.patches), "patch_kwargs": cfg.patch_kwargs}
    with bench.gpu_lock():
        r16.wait_exclusive()
        t0 = time.time()
        ctx = build(cfg)
        apply_patches(ctx)
        ctx.rng = RngTape(mode="off")
        batches = preload_batches(ctx, k=2)
        rep["build_s"] = time.time() - t0
        h = p23.fp4_handles(ctx)
        rep["both_handles_present"] = bool(h["head"] is not None and h["backbone"] is not None)
        rep["distinct_states"] = bool(h["head"].state is not h["backbone"].state) if rep["both_handles_present"] else None
        rep["head_policy"] = h["head"].policy.describe() if h["head"] is not None else None
        rep["backbone_policy"] = h["backbone"].describe() if h["backbone"] is not None else None
        rep["opt_step_chain"] = p23.opt_step_chain(ctx)
        ops = ctx.ops
        rep["ops_owner"] = {"predict": getattr(ops.predict, "__module__", None), "block": getattr(ops.block, "__module__", type(ops.block).__module__), "opt_step": type(ops.opt_step).__name__}
        rep["patch_order"] = list(cfg.patches)
        print("chain", json.dumps(rep["opt_step_chain"]), flush=True)
        print("ops", json.dumps(rep["ops_owner"]), flush=True)
        n_head, n_bb = h["head"].state.n_sites, h["backbone"].state.n_sites
        rep["n_sites"] = {"head": n_head, "backbone": n_bb}
        # graph path
        runner = r16.make_runner(ctx, batches, SimpleNamespace(max_targets=2, max_resident=8, metrics="lazy", no_side_stream=False))
        runner.prepare(batches[0])
        torch.cuda.synchronize()
        steps0 = {k: int(v.state.step.item()) for k, v in h.items()}
        rep["state_step_after_warmup_capture"] = steps0
        rep["head_inited_after_warmup"] = [bool(x) for x in h["head"].state.inited].count(True)
        rep["backbone_active_inited_after_warmup"] = all(h["backbone"].state.inited[i] for i in h["backbone"].active_sites())
        runner.sync_probs_state()
        losses, rolls = [], []
        for k in range(args.steps):
            o = runner.step(batches[k % 2], k, batches[(k + 1) % 2])
            torch.cuda.synchronize()
            losses.append(float(o.loss))
            rolls.append({kk: int(v.state.step.item()) - steps0[kk] for kk, v in h.items()})
        rep["graph_losses"] = losses
        rep["graph_roll_counts_after_each_step"] = rolls
        rep["graph_each_state_rolls_once_per_step"] = all(r["head"] == k + 1 and r["backbone"] == k + 1 for k, r in enumerate(rolls))
        rep["graph_losses_finite"] = all(math.isfinite(x) for x in losses)
        rep["graph_report"] = runner.report.as_dict()
        runner.release()
        del runner
        r16.free_gpu()
        # eager path (same ctx; train_step goes through ops.opt_step -> both rolls)
        padded = [{"image": b["image"], "binary_masks": padded_targets(b, 2)} for b in batches]
        s1 = {k: int(v.state.step.item()) for k, v in h.items()}
        el = []
        for k in range(2):
            o = train_step(ctx, padded[k % 2], k)
            torch.cuda.synchronize()
            el.append(float(o.loss))
        s2 = {k: int(v.state.step.item()) for k, v in h.items()}
        rep["eager_losses"] = el
        rep["eager_roll_counts_2_steps"] = {k: s2[k] - s1[k] for k in s1}
        rep["eager_each_state_rolls_once_per_step"] = bool(all(v == 2 for v in rep["eager_roll_counts_2_steps"].values()))
        rep["eager_losses_finite"] = all(math.isfinite(x) for x in el)
        # saturation statistics of both states (host side, outside any graph)
        from round23_fp4_final import run as _r  # only for the summarisers; its main() is not run
        rep["fp4_stats_brief"] = {k: {kk: v.get(kk) for kk in ("policy", "step", "total_calls", "total_exceed", "exceed_rate")} for k, v in _r.collect_fp4(ctx).items()}
        rep["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        _r.release_fp4(ctx)
    rep["opt_step_chain_ok"] = all(v for v in rep["opt_step_chain"].values() if isinstance(v, bool))
    rep["ok"] = bool(rep["both_handles_present"] and rep["distinct_states"] and rep["opt_step_chain_ok"]
                     and rep["graph_each_state_rolls_once_per_step"] and rep["eager_each_state_rolls_once_per_step"] and rep["graph_losses_finite"] and rep["eager_losses_finite"]
                     and rep["backbone_active_inited_after_warmup"])
    args.out.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("ok", "both_handles_present", "distinct_states", "graph_each_state_rolls_once_per_step", "eager_each_state_rolls_once_per_step", "graph_losses_finite",
                                          "eager_losses_finite", "opt_step_chain_ok", "backbone_active_inited_after_warmup", "head_inited_after_warmup", "n_sites", "peak_reserved_gb")}), flush=True)
    print("graph losses", [round(x, 4) for x in rep["graph_losses"]], "eager losses", [round(x, 4) for x in rep["eager_losses"]], flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
