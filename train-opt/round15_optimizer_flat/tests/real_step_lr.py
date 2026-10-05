"""Real-gradient optimizer check at a step where lr > 0.

Gate B at training step 0 compares updates that are exactly zero (both warmup lr = 0), so it says nothing about the update itself.  Here the flat
optimizer's device counters are moved to scheduler step 1500 (past both warmups: lr > 0 in every group), ONE real eager step (real batch, P1,r15 stack) is run,
the real gradients are captured at ``opt_step``, and the same gradients are applied by torch ``AdamW(fused=True)`` + ``clip_grad_norm_(0.01)`` from the same
weights and zero Adam state (per-group lr taken from the real scheduler at step 1500).  Compares updated parameters, update direction/size and grad norm.

  python real_step_lr.py <model> [stack]          (takes the GPU lock)
"""
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import importlib.util  # noqa: E402

from common import bench  # noqa: E402
from common.harness import apply_patches, build, preload_batches, set_phase, train_step  # noqa: E402

spec = importlib.util.spec_from_file_location("r15run", str(ROOT / "round15_optimizer_flat" / "run.py"))
r15run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r15run)

model = sys.argv[1] if len(sys.argv) > 1 else "vitb16"
stack = sys.argv[2] if len(sys.argv) > 2 else "P1,r15"
SCHED_STEP = 1500
out = {"model": model, "stack": stack, "sched_step": SCHED_STEP}
with bench.gpu_lock():
    r15run.wait_exclusive()
    ctx = build(r15run.make_cfg(model, stack))
    apply_patches(ctx)
    batches = preload_batches(ctx, k=1)
    flat = ctx.meta["round15_optimizer_flat"]
    opt = getattr(ctx.optimizer, "optimizer", ctx.optimizer)
    ctx.meta["phase_override"] = True
    set_phase(ctx, "mid")
    # lr of every group at the chosen scheduler step (real scheduler)
    ctx.scheduler.last_epoch = SCHED_STEP
    lrs = [float(x) for x in ctx.scheduler.get_lr()]
    out["lr_min_max"] = [min(lrs), max(lrs)]
    flat.ctl[0] = 0
    flat.ctl[1] = SCHED_STEP
    f = flat.flat
    init = {i: p.detach().clone() for i, p in enumerate(f.params)}
    captured = {}
    orig = ctx.ops.opt_step

    def grab(c):
        captured["grads"] = {i: p.grad.detach().clone() for i, p in enumerate(f.params) if p.grad is not None}
        return orig(c)

    ctx.ops.opt_step = grab
    o = train_step(ctx, batches[0], 0)
    ctx.ops.opt_step = orig
    torch.cuda.synchronize()
    out["grad_norm_flat"] = float(o.grad_norm)
    after = {i: p.detach().clone() for i, p in enumerate(f.params)}
    # reference: torch AdamW(fused) + clip_grad_norm_ on cloned parameters
    clones = [torch.nn.Parameter(init[i].clone()) for i in range(len(f.params))]
    index = {id(p): i for i, p in enumerate(f.params)}
    groups = []
    for gi, g in enumerate(opt.param_groups):
        groups.append({"params": [clones[index[id(p)]] for p in g["params"]], "lr": lrs[gi], "weight_decay": g.get("weight_decay", 0.05)})
    ref = torch.optim.AdamW(groups, weight_decay=0.05, fused=True)
    for i, c in enumerate(clones):
        c.grad = captured["grads"][i].clone()
    gn = torch.nn.utils.clip_grad_norm_(clones, 0.01, error_if_nonfinite=False)
    ref.step()
    torch.cuda.synchronize()
    out["grad_norm_ref"] = float(gn)
    dd = aa = bb = ab = 0.0
    maxabs = 0.0
    per_param_cos_min = 1.0
    for i in range(len(clones)):
        da, db = (after[i] - init[i]).double().reshape(-1), (clones[i].detach() - init[i]).double().reshape(-1)
        dd += float(((da - db) ** 2).sum())
        aa += float((da * da).sum())
        bb += float((db * db).sum())
        ab += float((da * db).sum())
        maxabs = max(maxabs, float((after[i] - clones[i].detach()).abs().max()))
        na, nb = float(da.norm()), float(db.norm())
        if na > 0 and nb > 0:
            per_param_cos_min = min(per_param_cos_min, float((da * db).sum()) / (na * nb))
    out["update_norm_flat"] = math.sqrt(aa)
    out["update_norm_torch"] = math.sqrt(bb)
    out["update_rel_l2"] = math.sqrt(dd) / max(math.sqrt(aa), 1e-30)
    out["update_cos_global"] = ab / max(math.sqrt(aa * bb), 1e-30)
    out["update_cos_min_per_param"] = per_param_cos_min
    out["param_maxabs_diff"] = maxabs
    out["grad_norm_rel"] = abs(out["grad_norm_flat"] - out["grad_norm_ref"]) / max(out["grad_norm_ref"], 1e-30)
    out["update_nonzero"] = out["update_norm_flat"] > 0
    out["pass"] = bool(out["update_nonzero"] and out["update_cos_global"] >= 0.9999 and out["update_rel_l2"] <= 1e-3 and out["grad_norm_rel"] <= 1e-5)
print(json.dumps(out, indent=1))
(ROOT / "round15_optimizer_flat" / "profile" / f"real_step_lr_{model}.json").write_text(json.dumps(out, indent=1) + "\n")
