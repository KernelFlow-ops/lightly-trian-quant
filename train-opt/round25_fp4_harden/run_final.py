"""[round25 copy of round23_fp4_final/run_final.py: same code, round25 presets (P3_q8, P3_hr*, P3_ex, P3_H2*), caches / outputs of round25]
Train with the phase-3 stack P3 = FINAL (P2R14R15) + round21 NVFP4 head + round22 NVFP4 backbone, captured by round16's whole-step CUDA graph, fed by the real Lightly
DataLoader.  Template: round19_final/run_final.py (its loader / graph-runner / loop / validation + checkpoint helpers are imported unmodified, as is the round15 scheduler
``fix_scheduler_state`` workaround).

    # 200 real steps (takes the GPU locks itself; add --val / --save-ckpt DIR to also run one validation pass / write a checkpoint + export)
    ulimit -n 65536; PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src \\
    /root/miniconda3/bin/python /root/lightly-train/train-opt/round25_fp4_harden/run_final.py --model vitl16 --steps 200 --workers 16 --persistent

    # N steps + validation + checkpoint + export (what round23 checked with tests/check_export.py)
    ... run_final.py --model vitb16 --steps 20 --val --save-ckpt /root/autodl-tmp/train-opt-tmp/round25_fp4_harden/ckpt/vitb16 --workers 16 --persistent

    # the whole 20-epoch schedule (validation + checkpoint at every epoch end; NOT run in round23, only the short ones above)
    ... run_final.py --model vitl16 --steps 0 --epochs 20 --val --save-ckpt /root/autodl-tmp/runs/p3_vitl16 --workers 16 --persistent

Stacks (--stack, default P3 = the model's combined stack; P3H = head only = the strict tier of ViT-B; P3B = backbone only; FINAL = phase-2 stack; P3_RN = RN instead of SR; ...).
  ViT-L: P3 = head.f=mx8,head.dw=nv4+sr  +  backbone all.d=nv4+sr+all.w=nv4+sr+nofirst
  ViT-B: P3 = head.f=mx8,head.dw=nv4+sr  +  backbone all.d=nv4+sr+all.w=nv4+sr

Notes
  * The two ``FP4State`` objects (delayed amax tables + the Philox seed counter of the head and of the backbone) are NOT part of the checkpoint.  A resumed run starts both from
    scratch: the first call of every delayed-amax site re-initialises it with the exact amax of the current tensor (the same thing that happens on the warm-up of a fresh run),
    and the SR seed counter restarts.  This only costs the first step after a resume a different (exact) per-tensor scale; parameters / optimizer / scheduler resume unchanged.
  * checkpoint / export stay in the ORIGINAL format (state_dict keys / shapes / dtypes are untouched by the patches); resume them in the original (R0) code path or in this one.
  * ``ulimit -n 65536`` is needed with >= 16 DataLoader workers; --workers 16 --persistent is the recommended setting (round23 e2e).
  * the schedule (lr warm-up + poly decay, mask annealing) is defined by --epochs (default 20); --steps N only runs the first N steps of it.
  * the batch must have <= --max-targets (default 2) target masks per image (round19 dataset scan: max 2); a batch that does not fit falls back to an eager step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["QUACK_CACHE_DIR"] = "/root/autodl-tmp/train-opt-tmp/round25_fp4_harden/quack_cache"
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROUND = Path(__file__).resolve().parent

from round25_fp4_harden.impl import presets25 as p23  # noqa: E402


import torch  # noqa: E402

from common import bench  # noqa: E402
from common.config import B_AUTO  # noqa: E402
from common.harness import apply_patches, build  # noqa: E402
from round19_final import run_final as rf19  # noqa: E402  (registers r19's presets; undone by p23.register below)
from round16_full_graph import run as r16  # noqa: E402

# helpers reused unmodified from round19 (stack agnostic); fix_scheduler_state is the round15 checkpoint workaround
final_cfg = rf19.final_cfg
fix_scheduler_state = rf19.fix_scheduler_state
make_loader = rf19.make_loader
make_graph_runner = rf19.make_graph_runner
train_loop = rf19.train_loop


def validate_and_save(ctx, out_dir, train_loader, val_loader, step, **kw) -> dict:
    """round19's ``validate_and_save`` with ONE fix for ``--persistent``: the checkpoint state contains the train DataLoader object (as in Lightly's own checkpoints) and, with
    ``persistent_workers=True``, torch keeps the live ``_MultiProcessingDataLoaderIter`` in ``DataLoader._iterator``, which cannot be pickled
    (``NotImplementedError: ('{} cannot be pickled', '_MultiProcessingDataLoaderIter')`` from ``torch.save``; round19's persistent option never ran with a checkpoint).  The iterator is
    detached from the loader (and from Fabric's wrapper) for the duration of the call and put back afterwards; the workers keep running.  The saved loader state is then the one of a
    non-persistent loader, i.e. what the original training writes."""
    holders, saved = [train_loader, getattr(train_loader, "_dataloader", None)], []
    for h in holders:
        if h is not None and getattr(h, "_iterator", None) is not None:
            saved.append((h, h._iterator))
            h._iterator = None
    try:
        return rf19.validate_and_save(ctx, out_dir, train_loader, val_loader, step, **kw)
    finally:
        for h, it in saved:
            h._iterator = it


def shutdown_loader(loader) -> None:
    """Stop persistent DataLoader workers explicitly (an exception while they are alive would otherwise hang the interpreter at exit and keep the GPU locks)."""
    for h in (loader, getattr(loader, "_dataloader", None)):
        it = getattr(h, "_iterator", None) if h is not None else None
        if it is not None and hasattr(it, "_shutdown_workers"):
            try:
                it._shutdown_workers()
            except Exception:  # noqa: BLE001
                pass


def fp4_report(ctx) -> dict:
    """Host-side saturation statistics of both FP4 states (call after the run; one device sync each)."""
    out: dict = {}
    h = p23.fp4_handles(ctx)
    if h["head"] is not None:
        st = h["head"].state.stats()
        out["head"] = {"policy": h["head"].policy.describe(), "step": st["step"], "total_calls": st["total_calls"], "total_exceed": st["total_exceed"], "exceed_rate": st["exceed_rate"]}
    if h["backbone"] is not None:
        st = h["backbone"].stats()
        out["backbone"] = {"policy": h["backbone"].describe(), **{k: st.get(k) for k in ("step", "total_calls", "total_exceed", "exceed_rate", "per_kind")}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=tuple(B_AUTO), required=True)
    ap.add_argument("--stack", default="P3")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = fixed B of the project (ViT-L 7, ViT-B 14)")
    ap.add_argument("--epochs", type=int, default=20, help="defines the lr / mask-annealing schedule")
    ap.add_argument("--steps", type=int, default=200, help="steps to run; 0 = the whole schedule (epochs x steps/epoch)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--persistent", action=argparse.BooleanOptionalAction, default=True, help="persistent DataLoader workers (default on)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--warm-steps", type=int, default=20, help="steps excluded from the steady-state img/s")
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--val", action="store_true", help="one validation pass at the end (and at every epoch end of a full run)")
    ap.add_argument("--save-ckpt", type=Path, default=None, help="out dir: write last.ckpt + exported_models/exported_last.pt at the end (and every epoch end of a full run)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    p23.register(ROUND, args.model)
    cfg = final_cfg(args.model, batch=args.batch_size or None, epochs=args.epochs, workers=args.workers, seed=args.seed, stack=args.stack)
    B = cfg.batch_size
    steps_per_epoch = 3672 // B
    run_steps = args.steps or cfg.total_steps
    report: dict = {"command": [sys.executable, *sys.argv], "model": args.model, "stack": args.stack, "batch": B, "total_steps_schedule": cfg.total_steps,
                    "steps": run_steps, "patches": list(cfg.patches), "patch_kwargs": cfg.patch_kwargs}
    with bench.gpu_lock():
        r16.wait_exclusive()
        t0 = time.time()
        ctx = build(cfg)
        apply_patches(ctx)
        fix_scheduler_state(ctx)
        loader = make_loader(ctx, "train", workers=args.workers, prefetch=args.prefetch, pin=not args.no_pin, persistent=args.persistent)
        val_loader = make_loader(ctx, "val", workers=min(args.workers, 16), prefetch=args.prefetch, pin=not args.no_pin) if args.val else None
        from lightly_train._data.infinite_cycle_iterator import InfiniteCycleIterator
        iterator = InfiniteCycleIterator(iterable=loader)
        from round10_integration.impl.fast_loop import _advance_transform
        _advance_transform(loader, iterator, 0, args.workers)
        first = next(iterator)
        runner = make_graph_runner(ctx, first, run_steps=run_steps, max_targets=args.max_targets)
        report["setup_s"] = time.time() - t0
        print(f"setup {report['setup_s']:.1f}s; graph signatures warmed: {sorted(runner.warmed)}", flush=True)
        report["opt_step_chain"] = p23.opt_step_chain(ctx)
        # the first batch cannot be pushed back into the iterator: run step 0 on it explicitly
        out = runner.step(first, 0, None)
        step, stats_all = 1, []
        out_dir = args.save_ckpt
        while step < run_steps:
            chunk = min(steps_per_epoch - step % steps_per_epoch, run_steps - step)
            stats = train_loop(ctx, runner, loader, iterator, steps=chunk, start_step=step, warm_steps=args.warm_steps if step == 1 else 0,
                               log_every=args.log_every, workers=args.workers)
            stats_all.append(stats)
            step += chunk
            if step % steps_per_epoch == 0 and step < run_steps and (args.val or out_dir):
                report.setdefault("epoch_end", []).append(validate_and_save(ctx, out_dir or Path("/tmp"), loader, val_loader, step, do_val=args.val, do_save=bool(out_dir)))
                print("epoch end", json.dumps(report["epoch_end"][-1]), flush=True)
        if args.val or out_dir:
            report["final"] = validate_and_save(ctx, out_dir or Path("/tmp"), loader, val_loader, run_steps, do_val=args.val, do_save=bool(out_dir))
            print("final", json.dumps(report["final"]), flush=True)
        report["train"] = [{k: v for k, v in s.items() if k != "losses"} for s in stats_all]
        report["graph"] = runner.report.as_dict()
        report["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        torch.cuda.synchronize()
        report["fp4_state"] = fp4_report(ctx)
        report["fp4_state_steps_equal_train_steps"] = bool(all(v.get("step") == run_steps for v in report["fp4_state"].values()))
        print("fp4 state", json.dumps({k: {kk: v.get(kk) for kk in ("step", "total_calls", "total_exceed", "exceed_rate")} for k, v in report["fp4_state"].items()}), flush=True)
        runner.release()
        shutdown_loader(loader)
        if val_loader is not None:
            shutdown_loader(val_loader)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("model", "stack", "batch", "steps", "peak_reserved_gb", "fp4_state_steps_equal_train_steps")}), flush=True)
    for s in report["train"]:
        print("train chunk", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001  an error with live persistent DataLoader workers would hang at interpreter exit while holding the GPU locks: print and leave at once
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
