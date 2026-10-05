"""Export / checkpoint compatibility of the FINAL stack (round19), compare stage.  The checkpoint is produced by
  run_final.py --model M --steps 20 --val --save-ckpt DIR       (FINAL stack under the graph runner, real loader; Lightly's own save_checkpoint + export_model)
and checked here, in a fresh process, against the ORIGINAL (R0) code path:

  1. checkpoint ``train_model`` keys / shapes / dtypes == a fresh R0 ``train_model.state_dict()``; checkpoint top-level keys / ``model_class_path`` /
     ``model_init_args`` == R0's; optimizer param_groups and per-parameter state (keys, shapes) == R0's AdamW after one R0 step; scheduler state keys equal;
  2. export dict keys, ``train_model`` keys / shapes == R0's ``get_export_state_dict()`` and == the original trained export (/root/autodl-tmp/vit{l,b}-best.pth);
  3. no NaN/Inf in the exported weights; the weights actually moved vs the initial (R0) weights;
  4. ``lightly_train.load_model(export)`` (original Lightly model class, no patches) loads and predicts one validation image;
  5. RESUME: the checkpoint's model / optimizer / scheduler load (strict) into a fresh R0 context and one original eager training step runs with a finite loss.

  python tests/check_export.py --model vitl16 --ckpt-dir /root/autodl-tmp/train-opt-tmp/round19_final/ckpt/vitl16 --out profile/export_vitl16.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND))
sys.path.insert(0, str(ROUND.parent))

import torch  # noqa: E402

import run as _r19run  # noqa: E402,F401
from round16_full_graph import run as r16  # noqa: E402
from common import bench  # noqa: E402
from common.config import DATASET_ROOT, EXPORT_WEIGHTS  # noqa: E402
from common.harness import apply_patches, build, preload_batches, train_step  # noqa: E402


def shapes(sd: dict) -> dict:
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items() if hasattr(v, "shape")}


def diff(a: dict, b: dict) -> dict:
    return {"missing": sorted(a.keys() - b.keys())[:6], "extra": sorted(b.keys() - a.keys())[:6],
            "changed": sorted(k for k in a.keys() & b.keys() if a[k] != b[k])[:6], "n_a": len(a), "n_b": len(b), "equal": a == b}


def opt_struct(sd: dict) -> dict:
    st = sd["state"]
    first = next(iter(st.values())) if st else {}
    return {"n_state": len(st), "state_keys": sorted(first.keys()), "param_groups": len(sd["param_groups"]),
            "group_keys": sorted(sd["param_groups"][0].keys()), "group_sizes": [len(g["params"]) for g in sd["param_groups"]],
            "state_shapes_equal_to_params": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ck_path = args.ckpt_dir / "checkpoints" / "last.ckpt"
    ex_path = args.ckpt_dir / "exported_models" / "exported_last.pt"
    if not ck_path.exists():
        found = sorted(args.ckpt_dir.rglob("last.ckpt"))
        ck_path = found[0] if found else ck_path
    rep: dict = {"command": [sys.executable, *sys.argv], "ckpt": str(ck_path), "export": str(ex_path)}
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ex = torch.load(ex_path, map_location="cpu", weights_only=False)
    rep["ckpt_top_keys"] = sorted(ck.keys())
    rep["export_top_keys"] = sorted(ex.keys())
    rep["ckpt_step"] = int(ck.get("step", -1))
    orig = torch.load(EXPORT_WEIGHTS[args.model], map_location="cpu", weights_only=False, mmap=True)
    rep["original_export_top_keys"] = sorted(orig.keys())
    rep["export_vs_original_trained_export"] = diff(shapes(orig["train_model"]), shapes(ex["train_model"]))
    rep["export_meta_equal_original"] = bool(orig["model_class_path"] == ex["model_class_path"] and orig["model_init_args"] == ex["model_init_args"])
    finite = all(bool(torch.isfinite(v).all()) for v in ex["train_model"].values() if v.is_floating_point())
    rep["export_all_finite"] = finite

    with bench.gpu_lock():
        r16.wait_exclusive()
        cfg = r16.make_cfg(args.model, "R0", seed=0)
        ctx = build(cfg)
        apply_patches(ctx)
        r0_sd = ctx.train_model.state_dict()
        init_model = {k: v.detach().cpu().clone() for k, v in r0_sd.items()}
        rep["ckpt_train_model_vs_R0"] = diff(shapes(r0_sd), shapes(ck["train_model"]))
        r0_exp = ctx.train_model.get_export_state_dict()
        rep["export_vs_R0_export"] = diff(shapes(r0_exp), shapes(ex["train_model"]))
        task = ctx.train_model.get_task_model()
        rep["ckpt_meta_equal_R0"] = bool(ck["model_class_path"] == task.class_path and ck["model_init_args"] == task.init_args)
        # weights moved vs the initial R0 weights (ckpt after the FINAL steps)
        moved = [float((ck["train_model"][k].float() - init_model[k].float()).abs().max()) for k in init_model if k in ck["train_model"] and init_model[k].is_floating_point()]
        rep["max_abs_weight_change_vs_init"] = max(moved)
        rep["n_params_changed"] = sum(1 for m in moved if m > 0)
        # R0 optimizer structure after one original step
        full = preload_batches(ctx, k=1)[0]
        # 2-image mini batch: R0 at the project's B needs ~29 GB, and resume (extra optimizer-state copies) would not fit in 32 GB; the batch size is irrelevant here
        batches = [{"image": full["image"][:2], "binary_masks": full["binary_masks"][:2]}]
        del full
        gc.collect()
        torch.cuda.empty_cache()
        train_step(ctx, batches[0], 0)
        r0_opt = ctx.optimizer.state_dict()
        r0_sched = ctx.scheduler.state_dict()
        rep["optimizer_R0_after_1_step"] = opt_struct(r0_opt)
        rep["optimizer_ckpt"] = opt_struct(ck["optimizer"])
        rep["optimizer_struct_equal"] = bool(
            rep["optimizer_R0_after_1_step"]["n_state"] == rep["optimizer_ckpt"]["n_state"] and rep["optimizer_R0_after_1_step"]["state_keys"] == rep["optimizer_ckpt"]["state_keys"]
            and rep["optimizer_R0_after_1_step"]["param_groups"] == rep["optimizer_ckpt"]["param_groups"] and rep["optimizer_R0_after_1_step"]["group_sizes"] == rep["optimizer_ckpt"]["group_sizes"]
            and rep["optimizer_R0_after_1_step"]["group_keys"] == rep["optimizer_ckpt"]["group_keys"])
        bad = [k for k in r0_opt["state"] if k in ck["optimizer"]["state"] and any(tuple(r0_opt["state"][k][s].shape) != tuple(ck["optimizer"]["state"][k][s].shape) for s in ("exp_avg", "exp_avg_sq"))]
        rep["optimizer_state_shape_mismatches"] = bad[:6]
        rep["scheduler_keys_equal"] = bool(sorted(r0_sched.keys()) == sorted(ck["scheduler"].keys()))
        rep["scheduler_ckpt_last_epoch"] = ck["scheduler"].get("last_epoch")
        # RESUME the checkpoint in the original code path
        gc.collect()
        torch.cuda.empty_cache()
        info = ctx.train_model.load_state_dict(ck["train_model"], strict=True)
        rep["resume_load_state_dict"] = str(info)
        ctx.optimizer.load_state_dict(ck["optimizer"])
        ctx.scheduler.load_state_dict(ck["scheduler"])
        out = train_step(ctx, batches[0], int(ck.get("step", 0)) + 1)
        torch.cuda.synchronize()
        rep["resume_step_loss"] = float(out.loss)
        rep["resume_step_loss_finite"] = bool(torch.isfinite(out.loss).item())
        rep["resume_step_grad_norm"] = float(out.grad_norm) if out.grad_norm is not None else None
        del ctx, out, batches
        gc.collect()
        torch.cuda.empty_cache()
        # the original Lightly model class, no patches
        import lightly_train

        image = next(iter(sorted((DATASET_ROOT / "valid").glob("*.jpg"))))
        model = lightly_train.load_model(ex_path, device="cuda")
        pred = model.predict(image)
        rep["load_model_predict"] = {"image": str(image), "shape": list(pred.shape), "dtype": str(pred.dtype), "classes": [int(v) for v in torch.unique(pred).cpu().tolist()],
                                     "class": type(model).__name__}
    keys = ["ckpt_train_model_vs_R0", "export_vs_R0_export", "export_vs_original_trained_export"]
    rep["pass"] = bool(all(rep[k]["equal"] for k in keys) and rep["ckpt_meta_equal_R0"] and rep["export_meta_equal_original"] and rep["export_all_finite"]
                       and rep["optimizer_struct_equal"] and not rep["optimizer_state_shape_mismatches"] and rep["scheduler_keys_equal"]
                       and rep["resume_step_loss_finite"] and bool(rep["load_model_predict"]["shape"]) and rep["n_params_changed"] > 0)
    args.out.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("pass", "ckpt_step", "optimizer_struct_equal", "scheduler_keys_equal", "resume_step_loss", "n_params_changed", "max_abs_weight_change_vs_init")}, default=str))
    for k in keys:
        print(k, json.dumps(rep[k]))
    print("predict", json.dumps(rep["load_model_predict"]))


if __name__ == "__main__":
    main()
