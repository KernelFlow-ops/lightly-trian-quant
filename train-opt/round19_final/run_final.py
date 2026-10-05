"""Train with the FINAL stack: TB1 BF16 + round13 fused block + round17 MXFP8 (fc+proj+qkv_dw) + round14 fused head + round15 flat optimizer,
captured by round16's whole-step CUDA graph, fed by the real Lightly DataLoader.  (common.config.P2R14R15_STACK / P2R14R15_PATCH_KWARGS)

    # 200 real steps (takes the GPU locks itself; add --val / --save-ckpt DIR to also run one validation pass / write a checkpoint + export)
    PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src ulimit -n 65536; \
    /root/miniconda3/bin/python /root/lightly-train/train-opt/round19_final/run_final.py --model vitl16 --steps 200 --workers 32

    # the whole 20-epoch schedule (validation + checkpoint at every epoch end; this path was NOT run in round19, only the short one above)
    ... run_final.py --model vitl16 --steps 0 --epochs 20 --val --save-ckpt /root/autodl-tmp/runs/final_vitl16

Notes
  * ``ulimit -n 65536`` is needed with >= 32 DataLoader workers (default soft limit 1024 -> "received 0 items of ancdata").
  * the schedule (lr warm-up + poly decay, mask annealing) is defined by --epochs (default 20, as the source script); --steps N only runs the first N steps of it.
  * the batch must have <= --max-targets (default 2) target masks per image: the dataset has background + head only (round19 dataset scan: max 2).
    A batch that does not fit falls back to an eager step (report.eager_fallbacks); on ViT-L that fallback can OOM while the graph pool is resident.
  * train metrics are lazy (updated at log steps only), the host never synchronises between steps except at --log-every.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
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
from common.config import B_AUTO, MODEL_NAMES  # noqa: E402
from common.harness import DINOv3EoMTSemanticSegmentationTrain, RunCfg, apply_patches, build  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round19_final.impl import presets  # noqa: E402

presets.register(ROUND)
from round16_full_graph import run as r16  # noqa: E402


def final_cfg(model: str, *, batch: int | None = None, epochs: int = 20, workers: int = 32, seed: int = 0, stack: str = "FINAL") -> RunCfg:
    patches, kwargs = r16.parse_stack(stack)
    B = batch or B_AUTO[model]
    return RunCfg(model=model, batch_size=B, phase="early", patches=patches, patch_kwargs=kwargs, seed=seed, num_workers=workers,
                  total_steps=max(1, (3672 // B) * epochs))


def fix_scheduler_state(ctx) -> bool:
    """Workaround for a round15 bug (round15 is read-only for round19): ``install_checkpoint_compat`` replaces ``scheduler.state_dict`` / ``load_state_dict``
    by local closures, and torch's ``LRScheduler.state_dict()`` returns the whole instance ``__dict__`` (minus the optimizer), closures included, so
    ``torch.save`` of a checkpoint fails with ``AttributeError: Can't pickle local object 'install_checkpoint_compat.<locals>.sched_state'``.
    Filter the two callables out of the returned dict.  Returns True if the wrapper was installed."""
    sch = ctx.scheduler
    if ctx.meta.get("round15_optimizer_flat") is None or not hasattr(sch, "state_dict"):
        return False
    inner = sch.state_dict

    def state_dict():
        return {k: v for k, v in inner().items() if k not in ("state_dict", "load_state_dict")}

    sch.state_dict = state_dict
    return True


def make_loader(ctx, split: str, *, workers: int, prefetch: int = 2, pin: bool = True, persistent: bool = False):
    """Same dataset / transform / collate / Fabric wrapper as common.data.make_loader, with the loader knobs exposed."""
    from lightly_train._commands import train_task_helpers as helpers

    train = split == "train"
    data_args = ctx.meta["data_args"]
    targs = ctx.meta["train_transform_args"] if train else ctx.meta["val_transform_args"]
    dargs = data_args.get_train_args() if train else data_args.get_val_args()
    dataset = dargs.get_dataset_cls()(dataset_args=dargs, image_info=list(dargs.list_image_info()))
    transform = (helpers.get_train_transform if train else helpers.get_val_transform)(DINOv3EoMTSemanticSegmentationTrain, MODEL_NAMES[ctx.cfg.model], targs)
    dataset.set_transform(transform)
    loader_args: dict = {"pin_memory": bool(pin)}
    if workers > 0:
        loader_args["prefetch_factor"] = int(prefetch)
        if persistent:
            loader_args["persistent_workers"] = True
    build_fn = helpers.get_train_dataloader if train else helpers.get_val_dataloader
    loader = build_fn(ctx.fabric, dataset, batch_size=ctx.meta["batch_size"], num_workers=workers, loader_args=loader_args)
    collate = dataset.get_batch_collate_fn_cls()(split=split, transform_args=targs)
    loader.collate_fn = collate
    helpers.disable_persistent_workers_for_step_aware_transforms(loader, transform, collate)
    return ctx.fabric.setup_dataloaders(loader)


def make_graph_runner(ctx, example_batch: dict, *, run_steps: int | None = None, max_targets: int = 2, metrics: str = "lazy"):
    """StepGraphRunner for the FINAL stack; warms (eagerly) every mask signature the first ``run_steps`` steps will visit."""
    from round16_full_graph.impl.graph_step import StepGraphRunner

    ctx.rng = RngTape(mode="off")
    if run_steps is not None:
        ctx.meta["total_steps_full"] = ctx.meta["total_steps"]
        ctx.meta["total_steps"] = int(run_steps)          # only used by the runner to decide which signatures to warm up
    runner = StepGraphRunner(ctx, example_batch, max_targets=max_targets, max_resident=8, metrics=metrics)
    runner.prepare(example_batch)
    return runner


def train_loop(ctx, runner, loader, iterator, *, steps: int, start_step: int = 0, warm_steps: int = 0, log_every: int = 0, workers: int = 0, quiet: bool = False) -> dict:
    """``steps`` real training steps (graph replays) on batches from ``iterator``.  Never syncs the host except at log steps and the two timing points."""
    from round10_integration.impl.fast_loop import _advance_transform

    B = ctx.meta["batch_size"]
    torch.cuda.synchronize()
    t_begin = time.perf_counter()
    t_warm, t_log, last_log_step = None, t_begin, start_step
    waits: list[float] = []
    out = None
    losses: list[tuple[int, float]] = []
    for i in range(steps):
        step = start_step + i
        _advance_transform(loader, iterator, step, workers)
        t0 = time.perf_counter()
        batch = next(iterator)
        waits.append((time.perf_counter() - t0) * 1e3)
        out = runner.step(batch, step, None)
        if warm_steps and i + 1 == warm_steps:
            torch.cuda.synchronize()
            t_warm = time.perf_counter()
        if log_every and (step + 1) % log_every == 0:
            runner.update_metrics()
            loss, gn = float(out.loss), float(out.grad_norm)         # one host sync per log interval
            now = time.perf_counter()
            losses.append((step + 1, loss))
            if not quiet:
                print(f"step {step + 1:6d} loss {loss:.4f} grad_norm {gn:.3f} {B * (step + 1 - last_log_step) / (now - t_log):7.1f} img/s", flush=True)
            t_log, last_log_step = now, step + 1
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    stats = {
        "steps": steps, "batch": B, "images": steps * B, "wall_s_total": t_end - t_begin, "img_s_total": steps * B / (t_end - t_begin),
        "first_batch_s": waits[0] / 1e3 if waits else None,
        "loss_last": float(out.loss) if out is not None else None, "losses": losses,
    }
    if t_warm is not None and steps > warm_steps:
        n = steps - warm_steps
        stats.update(steady_steps=n, steady_wall_s=t_end - t_warm, img_s_steady=n * B / (t_end - t_warm), step_ms_steady=(t_end - t_warm) / n * 1e3,
                     wait_ms_mean=statistics.fmean(waits[warm_steps:]), wait_ms_p95=sorted(waits[warm_steps:])[int(0.95 * (n - 1))], wait_ms_max=max(waits[warm_steps:]))
    return stats


def validate_and_save(ctx, out_dir: Path, train_loader, val_loader, step: int, *, do_val: bool, do_save: bool, best=None) -> dict:
    """One validation pass and/or one checkpoint (Lightly helpers, same files as the source training: last.ckpt + exported_last.pt), timed."""
    from lightly_train._commands import train_task_helpers as helpers
    from round10_integration.impl.fast_loop import _checkpoint_state, _export_dict, _validate

    res: dict = {"step": step}
    state = _checkpoint_state(ctx, train_loader)
    state["step"] = step
    if do_save:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        helpers.save_checkpoint(fabric=ctx.fabric, out_dir=out_dir, state=state, best_or_last="last")
        t1 = time.perf_counter()
        helpers.export_model(out_dir=out_dir, model_dict=_export_dict(ctx, state), best_or_last="last")
        t2 = time.perf_counter()
        ck, ex = helpers.get_checkpoint_path(out_dir=out_dir, best_or_last="last"), helpers.get_exported_model_path(out_dir, "last")
        res.update(ckpt_save_s=t1 - t0, export_save_s=t2 - t1, ckpt_path=str(ck), export_path=str(ex),
                   ckpt_gb=Path(ck).stat().st_size / 2**30, export_gb=Path(ex).stat().st_size / 2**30)
    if do_val:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        val_result = _validate(ctx, val_loader)
        torch.cuda.synchronize()
        res["val_s"] = time.perf_counter() - t0
        res["val_images"] = len(val_loader.dataset) if hasattr(val_loader, "dataset") else None
        res["val_metrics"] = {k: float(v) for k, v in val_result.metric_values.items() if isinstance(v, (int, float)) or hasattr(v, "item")}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=tuple(B_AUTO), required=True)
    ap.add_argument("--stack", default="FINAL")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = fixed B of the project (ViT-L 7, ViT-B 14)")
    ap.add_argument("--epochs", type=int, default=20, help="defines the lr / mask-annealing schedule")
    ap.add_argument("--steps", type=int, default=200, help="steps to run; 0 = the whole schedule (epochs x steps/epoch)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--persistent", action="store_true", help="persistent DataLoader workers (skips the worker respawn at every epoch)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--warm-steps", type=int, default=20, help="steps excluded from the steady-state img/s")
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--val", action="store_true", help="one validation pass at the end (and at every epoch end of a full run)")
    ap.add_argument("--save-ckpt", type=Path, default=None, help="out dir: write last.ckpt + exported_models/exported_last.pt at the end (and every epoch end of a full run)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    cfg = final_cfg(args.model, batch=args.batch_size or None, epochs=args.epochs, workers=args.workers, seed=args.seed, stack=args.stack)
    B = cfg.batch_size
    steps_per_epoch = 3672 // B
    run_steps = args.steps or cfg.total_steps
    report: dict = {"command": [sys.executable, *sys.argv], "model": args.model, "stack": args.stack, "batch": B, "total_steps_schedule": cfg.total_steps, "steps": run_steps}
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
        # first batch: push it back by wrapping the iterator with itself is not possible, so run step 0 on it explicitly
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
        runner.release()
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("model", "batch", "steps", "peak_reserved_gb")}), flush=True)
    for s in report["train"]:
        print("train chunk", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)


if __name__ == "__main__":
    main()
