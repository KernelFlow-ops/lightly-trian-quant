"""Real-model smoke / compat check of the flat optimizer inside the P2 stack: dead (no-grad) params, views, host/device schedule
agreement over a few eager steps, state_dict export + reload round trip.  Usage: python check_real_flat.py <model> [stack]"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "round15_optimizer_flat"))
from common import bench  # noqa: E402
from common.harness import apply_patches, build, preload_batches, set_phase, train_step  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("r15run", str(ROOT / "round15_optimizer_flat" / "run.py"))
r15run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r15run)

model = sys.argv[1] if len(sys.argv) > 1 else "vitb16"
stack = sys.argv[2] if len(sys.argv) > 2 else "P2,r15"
out = {"model": model, "stack": stack}
with bench.gpu_lock():
    ctx = build(r15run.make_cfg(model, stack))
    apply_patches(ctx)
    batches = preload_batches(ctx, k=2)
    flat = ctx.meta["round15_optimizer_flat"]
    ctx.meta["phase_override"] = True
    set_phase(ctx, "mid")
    names = flat.flat.names
    gn = []
    for step in range(4):
        o = train_step(ctx, batches[step % 2], step)
        gn.append(float(o.grad_norm))
    out["dead_params"] = [n for n, d in zip(names, flat.dead) if d]
    out["n_params_flat"] = len(names)
    out["flat_floats"] = flat.flat.total
    out["grad_norms"] = gn
    f = flat.flat
    out["views_ok"] = all(p.data_ptr() >= f.master.data_ptr() and p.data_ptr() < f.master.data_ptr() + 4 * f.total for p in f.params) and \
        all((p.grad is None) or (p.grad.data_ptr() >= f.grad.data_ptr() and p.grad.data_ptr() < f.grad.data_ptr() + 4 * f.total) for p in f.params)
    adam_t, sched = flat.device_counters()
    out["device_counters"] = (adam_t, sched)
    out["host_scheduler_last_epoch"] = ctx.scheduler.last_epoch
    # device lr table vs host scheduler (the host scheduler was stepped by the eager opt_step)
    host_lrs = [float(x) for x in ctx.scheduler.get_lr()]
    # eff holds the lr used in the LAST update = scheduler step (last_epoch - 1)
    ctx.scheduler.last_epoch -= 1
    ref = [float(x) for x in ctx.scheduler.get_lr()]
    ctx.scheduler.last_epoch += 1
    dev = flat.lr_table()
    out["lr_max_rel_err"] = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(dev, ref) if b != 0.0) if any(b != 0 for b in ref) else 0.0
    out["lr_table_head"] = dev[:3]
    out["lr_ref_head"] = ref[:3]
    # checkpoint round trip through the optimizer / scheduler wrappers
    sd = ctx.optimizer.state_dict() if hasattr(ctx.optimizer, "state_dict") else None
    opt = getattr(ctx.optimizer, "optimizer", ctx.optimizer)
    sd = opt.state_dict()
    out["sd_state_entries"] = len(sd["state"])
    out["sd_param_groups"] = len(sd["param_groups"])
    out["sd_group_keys"] = sorted(k for k in sd["param_groups"][0] if k != "params")
    first = next(iter(sd["state"].values()))
    out["sd_state_keys"] = sorted(first.keys())
    before = f.m.clone()
    opt.load_state_dict(sd)
    out["load_roundtrip_exact"] = bool(torch.equal(before, f.m))
    ssd = ctx.scheduler.state_dict()
    out["scheduler_state_last_epoch"] = ssd.get("last_epoch")
    out["torch_adamw_loadable"] = None
    # load our exported state into a plain torch AdamW over cloned params
    try:
        ps = [p for g in opt.param_groups for p in g["params"]]
        clones = [torch.nn.Parameter(p.detach().clone()) for p in ps]
        groups = []
        i = 0
        for g in opt.param_groups:
            groups.append({**{k: v for k, v in g.items() if k != "params"}, "params": clones[i:i + len(g["params"])]})
            i += len(g["params"])
        for g in groups:
            if isinstance(g["lr"], torch.Tensor):
                g["lr"] = float(g["lr"])
            g["fused"] = None
            g["capturable"] = False
        o2 = torch.optim.AdamW(groups)
        o2.load_state_dict(sd)
        out["torch_adamw_loadable"] = len(o2.state)
    except Exception as exc:
        out["torch_adamw_loadable"] = f"ERROR {type(exc).__name__}: {str(exc)[:200]}"
print(json.dumps(out, indent=1, default=str))
(ROOT / "round15_optimizer_flat" / "profile" / f"check_real_flat_{model}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
