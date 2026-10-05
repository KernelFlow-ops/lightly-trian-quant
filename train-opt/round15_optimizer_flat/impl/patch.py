"""round15_optimizer_flat: flat fp32 master/grad buffers + two-kernel clip+AdamW as ``ops.opt_step``.

Variant ``flat`` (default).  The torch AdamW object (``ctx.optimizer``) is kept as the metadata store (param groups,
state_dict format); its per-parameter state is NOT used: m / v / step live in flat device buffers.  ``state_dict()`` and
``load_state_dict()`` of the optimizer are wrapped so checkpoints keep torch AdamW's format.

Driving the optimizer: through ``ctx.ops.opt_step(ctx)`` (clip + step + zero_grad + host scheduler).  Inside a CUDA graph the
host scheduler is skipped (``flat.graph_mode = True``); the device counters are authoritative and ``flat_sync_host()``
copies them back into the scheduler / param_group lr when needed (it is called by ``optimizer.state_dict()``).
"""
from __future__ import annotations

from typing import Any

import torch

NAME = "round15_optimizer_flat"
REQUIRES: tuple[str, ...] = ()
CONFLICTS: tuple[str, ...] = ()


def _underlying(optimizer: Any) -> Any:
    return getattr(optimizer, "optimizer", optimizer)


def schedule_from(scheduler: Any, n_groups: int) -> tuple[dict[str, float] | None, list[int]]:
    """(sched params, per-group kind).  kind 0 non-ViT / 1 ViT for TwoStageWarmupPolySchedule, 2 = external lr."""
    try:
        from lightly_train._task_models.dinov3_eomt_semantic_segmentation.scheduler import TwoStageWarmupPolySchedule
    except Exception:  # pragma: no cover
        TwoStageWarmupPolySchedule = ()  # type: ignore[assignment]
    if TwoStageWarmupPolySchedule and isinstance(scheduler, TwoStageWarmupPolySchedule):
        nw, vw = scheduler.warmup_steps
        sched = dict(nw=float(nw), vw=float(vw), total=float(scheduler.total_steps), power=float(scheduler.poly_power))
        kinds = [1 if g < scheduler.num_backbone_params else 0 for g in range(n_groups)]
        return sched, kinds
    return None, [2] * n_groups


def build_flat(optimizer: Any, scheduler: Any, names: dict[int, str] | None = None, max_norm: float = 0.0):
    from .flat import FlatAdamW

    opt = _underlying(optimizer)
    groups = opt.param_groups
    if any(g.get("amsgrad") or g.get("maximize") for g in groups):
        raise ValueError("round15 flat AdamW supports plain AdamW only (no amsgrad / maximize)")
    sched, kinds = schedule_from(scheduler, len(groups))
    base_lrs = list(getattr(scheduler, "base_lrs", [None] * len(groups))) if scheduler is not None else [None] * len(groups)
    params, pnames, group_of, gdefs = [], [], [], []
    for gi, group in enumerate(groups):
        lr0 = base_lrs[gi]
        if lr0 is None:
            lr0 = group.get("initial_lr", group["lr"])
        lr0 = float(lr0.detach().cpu()) if isinstance(lr0, torch.Tensor) else float(lr0)
        gdefs.append(dict(base_lr=lr0, wd=float(group.get("weight_decay", opt.defaults.get("weight_decay", 0.0))),
                          betas=tuple(float(b) for b in group.get("betas", opt.defaults.get("betas", (0.9, 0.999)))),
                          eps=float(group.get("eps", opt.defaults.get("eps", 1e-8))), kind=kinds[gi]))
        for p in group["params"]:
            if not p.requires_grad:
                continue
            params.append(p)
            pnames.append((names or {}).get(id(p), f"g{gi}.p{len(params)}"))
            group_of.append(gi)
    flat = FlatAdamW(params, gdefs, group_of, names=pnames, sched=sched, max_norm=max_norm)
    flat.optimizer = opt
    flat.scheduler = scheduler
    # carry over any existing torch AdamW state (resume before patching)
    for i, p in enumerate(flat.flat.params):
        item = opt.state.get(p) if hasattr(opt, "state") else None
        if item and "exp_avg" in item:
            flat.load_param_state(i, item["exp_avg"], item["exp_avg_sq"])
            step = item.get("step")
            if step is not None:
                flat.ctl[0] = int(float(step))
    if scheduler is not None and hasattr(scheduler, "last_epoch"):
        flat.ctl[1] = int(scheduler.last_epoch)
    if hasattr(opt, "state"):
        opt.state.clear()
    return flat


def install_checkpoint_compat(flat, optimizer: Any, scheduler: Any) -> None:
    """Wrap ``optimizer.state_dict/load_state_dict/zero_grad`` so the flat buffers look like a torch AdamW."""
    opt = _underlying(optimizer)
    orig_state_dict, orig_load = opt.state_dict, opt.load_state_dict
    index_of = {id(p): i for i, p in enumerate(flat.flat.params)}

    def sync_host() -> None:
        """Copy the device counters / lr back into the host scheduler and param_group lr floats."""
        adam_t, sched_step = flat.device_counters()
        if scheduler is not None and hasattr(scheduler, "last_epoch"):
            scheduler.last_epoch = sched_step
            try:
                lrs = [float(x) for x in scheduler.get_lr()]
                scheduler._last_lr = lrs
                for group, lr in zip(opt.param_groups, lrs):
                    if isinstance(group["lr"], torch.Tensor):
                        group["lr"].fill_(lr)
                    else:
                        group["lr"] = lr
            except Exception:
                pass

    def state_dict():
        sync_host()
        raw = orig_state_dict()
        adam_t, _ = flat.device_counters()
        exported = {**raw, "param_groups": [dict(g) for g in raw["param_groups"]], "state": {}}
        pid = 0
        for gi, group in enumerate(opt.param_groups):
            for p in group["params"]:
                i = index_of.get(id(p))
                if i is not None and not flat.dead[i]:
                    m, v = flat.param_state(i, clone=True)
                    exported["state"][pid] = {"step": torch.tensor(float(adam_t)), "exp_avg": m, "exp_avg_sq": v}
                pid += 1
        return exported

    def load_state_dict(saved):
        adjusted = {**saved, "state": {}}
        orig_load(adjusted)
        pid, steps = 0, []
        for group in opt.param_groups:
            for p in group["params"]:
                item = saved["state"].get(pid)
                i = index_of.get(id(p))
                if item is not None and i is not None and "exp_avg" in item:
                    flat.load_param_state(i, item["exp_avg"].to(flat.flat.device), item["exp_avg_sq"].to(flat.flat.device))
                    steps.append(int(float(item["step"])))
                pid += 1
        if steps:
            flat.ctl[0] = max(steps)
        if scheduler is not None and hasattr(scheduler, "last_epoch"):
            flat.ctl[1] = int(scheduler.last_epoch)

    def zero_grad(set_to_none: bool = True):  # views must survive: always zero in place
        flat.zero_grad()

    opt.state_dict, opt.load_state_dict, opt.zero_grad = state_dict, load_state_dict, zero_grad
    flat.sync_host = sync_host
    if scheduler is not None and hasattr(scheduler, "load_state_dict"):
        orig_sched_load, orig_sched_state = scheduler.load_state_dict, scheduler.state_dict

        def sched_load(sd):
            orig_sched_load(sd)
            flat.ctl[1] = int(scheduler.last_epoch)

        def sched_state():
            sync_host()
            # LRScheduler.state_dict() returns the instance __dict__, which now holds these two wrappers;
            # they are local closures and cannot be pickled by torch.save, so drop them.
            return {k: v for k, v in orig_sched_state().items() if k not in ("load_state_dict", "state_dict")}

        scheduler.load_state_dict, scheduler.state_dict = sched_load, sched_state


def apply(ctx, variant: str = "flat", **kw) -> dict:
    if kw:
        raise ValueError(f"Unknown round15 options: {sorted(kw)}")
    if variant != "flat":
        raise ValueError(f"Unknown round15 variant: {variant}")
    if ctx.meta.get(NAME) is not None:
        return {"ops": ["opt_step"], "desc": f"{NAME}:{variant} (already installed)"}
    names = {id(p): n for n, p in ctx.train_model.named_parameters()}
    max_norm = float(ctx.train_model.model_args.gradient_clip_val)
    flat = build_flat(ctx.optimizer, ctx.scheduler, names, max_norm)
    install_checkpoint_compat(flat, ctx.optimizer, ctx.scheduler)
    opt = _underlying(ctx.optimizer)
    scheduler = ctx.scheduler

    @torch.no_grad()
    def opt_step(c):
        norm = flat.step()
        if not flat.graph_mode:
            opt._opt_called = True
            scheduler.step()
        return norm

    ctx.ops.opt_step = opt_step
    ctx.meta[NAME] = flat
    ctx.meta["flat_params"] = flat.flat          # contract: FlatParams (master / grad / register_weight_writer)
    return {"ops": ["opt_step"], "desc": f"{NAME}:{variant} (flat fp32 buffers, 2-kernel clip+AdamW, device lr schedule)"}
