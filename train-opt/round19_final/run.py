"""round19_final entrypoint: measurement / gates for the FINAL stack.  Thin wrapper around round16_full_graph/run.py (same bench / profile /
gate_b / graph_equiv actions; its outputs are redirected to round19_final/) plus the round19 actions below.

Stacks (--stack, --ref): R0, P1 (=TB1), P2, P2R15, P2R14, FINAL (= P2 + r14 + r15, the candidate final stack), bisect presets FINAL_noR14 / FINAL_noR15 /
FINAL_fc / FINAL_noMX, or any "name[:variant],..." spec (see impl/presets.py and round16 run.py).

Actions (every GPU action holds the dual lock):
    bench        0.2/0.6/0.2 weighted early/mid/late timing, warmup 20, 50 x 3; --runner eager|graph       (round16 action)
    profile      steady-state window for nsys, --runner eager|graph                                          (round16 action)
    graph_equiv  graph vs eager on identical RNG                                                             (round16 action)
    gate_b       one real step vs --ref, several seeds (single process per seed; the multi-stack version is tests/gate_b_multi.py)
    gate_s       50-step trajectories on 2 real batches: --stack under --runner (graph) vs --ref eager (default R0), gate_c(interval=10, max_rel=0.05)
    maxbatch     one batch size per process: build, capture, time the mid phase, report peak reserved / device memory (OOM is caught and reported)
    e2e          real DataLoader + graph runner: GPU-only steps vs 200-step end-to-end for several loader settings (workers / prefetch / pin / persistent)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROUND = Path(__file__).resolve().parent

import torch  # noqa: E402

from common import bench  # noqa: E402
from common.check import gate_c  # noqa: E402
from common.config import B_AUTO  # noqa: E402
from common.harness import apply_patches, build, preload_batches  # noqa: E402
from round19_final.impl import presets  # noqa: E402

presets.register(ROUND)
from round16_full_graph import run as r16  # noqa: E402


def device_mem() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {"device_used_gb": (total - free) / 2**30, "allocated_gb": torch.cuda.memory_allocated() / 2**30, "reserved_gb": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}


# ------------------------------------------------------------------ gate S: --stack (graph) vs --ref (eager, default R0)
def action_gate_s(args) -> None:
    cfg_c = r16.make_cfg(args.model, args.stack, seed=0)
    cfg_r = r16.make_cfg(args.model, args.ref, seed=0)
    report = {"command": [sys.executable, *sys.argv], "ref": args.ref, "stack": args.stack, "runner": args.runner, "runs": {}}
    with bench.gpu_lock():
        r16.wait_exclusive()
        loader_ctx = build(cfg_c)
        batches = preload_batches(loader_ctx, k=2)
        del loader_ctx
        r16.free_gpu()
        for scen in args.scenarios:
            t0 = time.time()
            ref = r16.curve_run(cfg_r, batches, args.steps, scen, "eager", args)
            cand = r16.curve_run(cfg_c, batches, args.steps, scen, args.runner, args)
            gate = gate_c(ref, cand, interval=10, max_rel=0.05)
            report["runs"][scen] = {"gate": gate, "ref": ref, "cand": cand}
            print(scen, json.dumps({"max_rel": gate["curve300_max_rel"], "pass": gate["pass"], "finite": gate["finite"], "checkpoints": gate["checkpoints"],
                                    "grad_norm_ratios": gate["grad_norm_ratios"], "loss_first": [ref["loss"][0], cand["loss"][0]], "loss_last": [ref["loss"][-1], cand["loss"][-1]],
                                    "captures": cand.get("graph", {}).get("captures"), "seconds": round(time.time() - t0, 1)}), flush=True)
    out = r16.unique_path(ROUND / "profile" / f"gateS_{args.model}_{r16.safe_name(args.stack)}_{args.runner}_vs_{r16.safe_name(args.ref)}.json")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", flush=True)


# ------------------------------------------------------------------ max batch (one B per process)
def action_maxbatch(args) -> None:
    B = args.batch
    cfg = r16.make_cfg(args.model, args.stack, seed=args.seed, batch=B)
    res: dict = {"model": args.model, "stack": args.stack, "runner": args.runner, "batch": B, "ok": False, "oom": False}
    t0 = time.time()
    try:
        with bench.gpu_lock():
            r16.wait_exclusive()
            ctx = build(cfg)
            apply_patches(ctx)
            batches = preload_batches(ctx, k=2)
            step_fn, runner = r16.make_step(ctx, batches, args)
            res["mem_after_setup"] = device_mem()
            measured = bench.time_steps(ctx, batches, phases=("mid",), warmup=args.warmup, iters=args.iters, repeats=args.repeats, segments=False, step_fn=step_fn)
            m = measured["mid"]
            res.update(ok=True, step_ms=m["step_ms"]["mean"], img_s=m["img_s"], repeat_ms=[r["step_ms"]["mean"] for r in m["repeats"]], mem=device_mem(),
                       gpu_snapshot=bench.gpu_snapshot()["gpu"], graph=runner.report.as_dict() if runner is not None else None)
    except torch.OutOfMemoryError as exc:
        res.update(oom=True, error=str(exc)[:400], mem=device_mem() if torch.cuda.is_available() else None)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            res.update(oom=True, error=str(exc)[:400])
        else:
            res.update(error=f"{type(exc).__name__}: {str(exc)[:400]}")
    res["seconds"] = time.time() - t0
    out = r16.unique_path(ROUND / "profile" / f"maxbatch_{args.model}_{r16.safe_name(args.stack)}_{args.runner}_B{B}.json")
    out.write_text(json.dumps({"result": res, "command": [sys.executable, *sys.argv]}, indent=2, default=str) + "\n")
    print(json.dumps({k: res.get(k) for k in ("model", "batch", "runner", "ok", "oom", "step_ms", "img_s")}), json.dumps(res.get("mem")), flush=True)
    print(f"wrote {out}", flush=True)


# ------------------------------------------------------------------ end to end with the real loader
def action_e2e(args) -> None:
    import run_final as rf
    from lightly_train._data.infinite_cycle_iterator import InfiniteCycleIterator
    from round10_integration.impl.fast_loop import _advance_transform

    cfg = rf.final_cfg(args.model, batch=args.batch or None, epochs=20, workers=max(args.worker_list), seed=args.seed, stack=args.stack)
    settings = [dict(zip(("workers", "prefetch", "pin", "persistent"), s)) for s in (tuple(x.split(":")) for x in args.loader_settings)]
    for s in settings:
        s["workers"], s["prefetch"], s["pin"], s["persistent"] = int(s["workers"]), int(s["prefetch"]), s["pin"] == "1", s["persistent"] == "1"
    report: dict = {"command": [sys.executable, *sys.argv], "model": args.model, "stack": args.stack, "batch": cfg.batch_size, "steps": args.steps, "runs": []}
    with bench.gpu_lock():
        r16.wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        batches = preload_batches(ctx, k=2)
        runner = rf.make_graph_runner(ctx, batches[0], run_steps=args.steps + args.warmup + 200)
        # GPU-only reference in the same process: preloaded batches, early phase (the first steps of the real schedule are all 'one'), then the real schedule
        step_fn = runner.as_step_fn(batches)
        gpu = bench.time_steps(ctx, batches, phases=("early",), warmup=args.warmup, iters=50, repeats=3, segments=False, step_fn=step_fn)
        runner.sync_probs_state()
        report["gpu_only_early"] = {"step_ms": gpu["early"]["step_ms"]["mean"], "img_s": gpu["early"]["img_s"], "repeat_ms": [r["step_ms"]["mean"] for r in gpu["early"]["repeats"]]}
        print("gpu-only early", json.dumps(report["gpu_only_early"]), flush=True)
        for s in settings:
            t0 = time.time()
            loader = rf.make_loader(ctx, "train", workers=s["workers"], prefetch=s["prefetch"], pin=s["pin"], persistent=s["persistent"])
            iterator = InfiniteCycleIterator(iterable=loader)
            torch.cuda.synchronize()
            # (a) the common.bench.time_e2e pattern: wall clock from before the first next() (includes worker spawn + first-batch latency)
            stats = rf.train_loop(ctx, runner, loader, iterator, steps=args.steps, start_step=0, warm_steps=args.warmup, log_every=0, workers=s["workers"], quiet=True)
            row = {"setting": s, **{k: v for k, v in stats.items() if k != "losses"}, "seconds": time.time() - t0}
            row["steady_vs_gpu_only"] = row.get("img_s_steady", 0) / report["gpu_only_early"]["img_s"]
            report["runs"].append(row)
            print("e2e", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            del loader, iterator
            r16.free_gpu()
        report["graph"] = runner.report.as_dict()
        report["mem"] = device_mem()
        runner.release()
    out = r16.unique_path(ROUND / "profile" / f"e2e_{args.model}_{args.tag}.json")
    out.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=tuple(B_AUTO), required=True)
    ap.add_argument("--stack", default="FINAL")
    ap.add_argument("--name")
    ap.add_argument("--action", choices=("bench", "profile", "gate_b", "gate_s", "graph_equiv", "maxbatch", "e2e"), required=True)
    ap.add_argument("--runner", choices=("eager", "graph"), default="graph")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--ref", default="R0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision-change", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--scenarios", nargs="*", default=["schedule", "fixed_mid", "fixed_late"])
    ap.add_argument("--equiv-steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--preload", type=int, default=2)
    ap.add_argument("--batch", type=int, default=0, help="maxbatch / e2e: batch size (0 = B_AUTO)")
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--max-resident", type=int, default=8)
    ap.add_argument("--metrics", choices=("step", "lazy", "off"), default="lazy")
    ap.add_argument("--no-side-stream", action="store_true")
    ap.add_argument("--cpu-batches", action="store_true")
    ap.add_argument("--no-segments", action="store_true")
    ap.add_argument("--no-results", action="store_true", default=True)
    ap.add_argument("--loader-settings", nargs="*", default=["32:2:1:0"], help="e2e: workers:prefetch:pin(0/1):persistent(0/1)")
    ap.add_argument("--tag", default="x")
    args = ap.parse_args()
    args.worker_list = [int(s.split(":")[0]) for s in args.loader_settings]
    if args.action not in ("bench",) and args.phases == "all":
        args.phases = "mid"
    if args.action == "maxbatch":
        args.phases = "mid"
        if args.iters == 50:
            args.iters, args.repeats = 30, 1
    {"bench": r16.action_bench, "profile": r16.action_profile, "gate_b": r16.action_gate_b, "graph_equiv": r16.action_graph_equiv,
     "gate_s": action_gate_s, "maxbatch": action_maxbatch, "e2e": action_e2e}[args.action](args)


if __name__ == "__main__":
    sys.path.insert(0, str(ROUND))
    main()
