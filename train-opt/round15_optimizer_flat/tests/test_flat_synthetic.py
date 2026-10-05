"""Gate for round15 flat AdamW on synthetic data: 100-step trajectory vs the r7 fp32 oracle (optimizer_ref.py) and
torch AdamW(fused) + clip_grad_norm_, device lr schedule vs torch's TwoStageWarmupPolySchedule, checkpoint round trips,
determinism, zeroed grads and frozen (no-grad) parameters.  Run: python test_flat_synthetic.py  (takes the GPU lock)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import bench  # noqa: E402
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.scheduler import TwoStageWarmupPolySchedule  # noqa: E402
from round7_optimizer.impl.optimizer_ref import clipped_adamw_reference_step  # noqa: E402
from round15_optimizer_flat.impl.patch import build_flat, install_checkpoint_compat  # noqa: E402

DEV = torch.device("cuda")
SHAPES = [(1,), (3,), (1000,), (4096,), (5000,), (12345,), (64, 64, 3, 3), (32, 16, 3, 3), (300, 1024), (7, 1031), (2, 96), (777,)]
CL_INDEX = 7                       # channels_last parameter
DEAD_INDEX = 11                    # never receives a gradient
GROUPS = [  # (param indices, lr, weight_decay)
    ([0, 2, 8], 2e-4, 0.05), ([1, 3], 1.5e-4, 0.0), ([4, 5], 1e-4, 0.05),
    ([6, 7], 4e-4, 0.05), ([9], 4e-4, 0.0), ([10, 11], 3e-4, 0.05),
]
NUM_BACKBONE_GROUPS = 3
WARMUP = (6, 9)
TOTAL = 100
POWER = 0.9
MAX_NORM = 0.01
STEPS = 100


def make_params(seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ps = []
    for i, s in enumerate(SHAPES):
        t = torch.randn(*s, generator=g)
        if i == CL_INDEX:
            t = t.contiguous(memory_format=torch.channels_last)
        ps.append(torch.nn.Parameter(t.to(DEV)))
    return ps


def make_opt(params, *, fused=False):
    groups = [{"params": [params[i] for i in idx], "lr": lr, "weight_decay": wd} for idx, lr, wd in GROUPS]
    opt = torch.optim.AdamW(groups, weight_decay=0.05, fused=fused)
    sch = TwoStageWarmupPolySchedule(opt, NUM_BACKBONE_GROUPS, WARMUP, TOTAL, POWER)
    return opt, sch


def grads_for(step: int):
    g = torch.Generator(device="cpu").manual_seed(1000 + step)
    out = []
    for i, s in enumerate(SHAPES):
        t = torch.randn(*s, generator=g) * (0.5 + step % 7)
        if step % 5 == 0 and t.numel() > 8:
            t.view(-1)[:: max(1, t.numel() // 7)] *= 8.0          # spiky scale-8 data
        out.append(t.to(DEV))
    return out


def _maxdiff(a, b):
    d = (a.detach().float().reshape(-1) - b.detach().float().reshape(-1)).abs()
    return float(d.max()) if d.numel() else 0.0


def main() -> dict:
    res: dict = {}
    with bench.gpu_lock():
        # ---------------- three contenders on identical initial weights
        pa, pb, pc = make_params(0), make_params(0), make_params(0)
        oa, sa = make_opt(pa)
        ob, sb = make_opt(pb, fused=True)
        oc, sc = make_opt(pc)
        state_a: dict = {}
        flat = build_flat(oc, sc, {id(p): f"p{i}" for i, p in enumerate(pc)}, MAX_NORM)
        install_checkpoint_compat(flat, oc, sc)
        flat.graph_mode = True                                   # device-side schedule only (host scheduler not stepped)
        fidx = {id(p): j for j, p in enumerate(flat.flat.params)}
        worst = {"param_vs_ref": 0.0, "fused_vs_ref": 0.0, "param_vs_fused": 0.0, "m_vs_fused": 0.0, "v_vs_fused": 0.0, "norm_vs_ref_rel": 0.0,
                 "norm_vs_fused_rel": 0.0, "lr_abs": 0.0, "lr_rel": 0.0}
        dead_before = pc[DEAD_INDEX].detach().clone()
        checkpoints = {}
        for step in range(STEPS):
            gs = grads_for(step)
            for i in range(len(SHAPES)):
                if i == DEAD_INDEX:
                    continue
                pa[i].grad = gs[i].clone().to(memory_format=torch.channels_last) if i == CL_INDEX else gs[i].clone()
                pb[i].grad = gs[i].clone().to(memory_format=torch.channels_last) if i == CL_INDEX else gs[i].clone()
                if flat.adopted:
                    pc[i].grad.add_(gs[i].view(pc[i].shape).to(memory_format=torch.channels_last) if i == CL_INDEX else gs[i])
                else:
                    pc[i].grad = gs[i].clone().to(memory_format=torch.channels_last) if i == CL_INDEX else gs[i].clone()
            lr_host = [g["lr"] for g in oa.param_groups]
            norm_a = clipped_adamw_reference_step(oa.param_groups, state_a, MAX_NORM)
            norm_b = torch.nn.utils.clip_grad_norm_([p for p in pb if p.grad is not None], MAX_NORM, error_if_nonfinite=False)
            ob.step()
            norm_c = flat.step()
            sa.step(); sb.step()
            if step == 0:
                assert flat.dead[fidx[id(pc[DEAD_INDEX])]] and sum(flat.dead) == 1, "dead (no-grad) parameter not detected"
            lr_dev = flat.lr_table()
            for g, (idx, _, _) in enumerate(GROUPS):
                a, b = lr_dev[g], float(lr_host[g])
                worst["lr_abs"] = max(worst["lr_abs"], abs(a - b))
                worst["lr_rel"] = max(worst["lr_rel"], abs(a - b) / max(abs(b), 1e-12))
            worst["norm_vs_ref_rel"] = max(worst["norm_vs_ref_rel"], abs(float(norm_c) - float(norm_a)) / float(norm_a))
            worst["norm_vs_fused_rel"] = max(worst["norm_vs_fused_rel"], abs(float(norm_c) - float(norm_b)) / float(norm_b))
            for i in range(len(SHAPES)):
                if i == DEAD_INDEX:
                    continue
                worst["param_vs_ref"] = max(worst["param_vs_ref"], _maxdiff(pc[i], pa[i]))
                worst["param_vs_fused"] = max(worst["param_vs_fused"], _maxdiff(pc[i], pb[i]))
                worst["fused_vs_ref"] = max(worst["fused_vs_ref"], _maxdiff(pb[i], pa[i]))
                worst["max_abs_param"] = max(worst.get("max_abs_param", 0.0), float(pa[i].abs().max()))
                mc, vc = flat.param_state(fidx[id(pc[i])], clone=False)
                worst["m_vs_fused"] = max(worst["m_vs_fused"], _maxdiff(mc, ob.state[pb[i]]["exp_avg"]))
                worst["v_vs_fused"] = max(worst["v_vs_fused"], _maxdiff(vc, ob.state[pb[i]]["exp_avg_sq"]))
            if step + 1 in (1, 10, 50, 100):
                checkpoints[step + 1] = {k: v for k, v in worst.items()}
            if step + 1 == 50:
                sd_flat = copy.deepcopy(oc.state_dict())
                sd_torch = copy.deepcopy(ob.state_dict())
                snap50 = ([p.detach().clone() for p in pc], [p.detach().clone() for p in pb])
            if step + 1 == 60:
                snap60 = ([p.detach().clone() for p in pc], [p.detach().clone() for p in pb])
        res["trajectory_worst"] = worst
        res["trajectory_checkpoints"] = checkpoints
        # frozen parameter untouched, grads zeroed
        res["dead_param_unchanged"] = bool(torch.equal(pc[DEAD_INDEX], dead_before))
        res["grad_buffer_zero_after_step"] = bool(float(flat.flat.grad.abs().max()) == 0.0)
        res["grad_views_are_flat"] = all(pc[i].grad.data_ptr() >= flat.flat.grad.data_ptr() and pc[i].grad.data_ptr() < flat.flat.grad.data_ptr() + 4 * flat.flat.total
                                         for i in range(len(SHAPES)) if i != DEAD_INDEX)
        res["param_views_are_flat"] = all(pc[i].data_ptr() >= flat.flat.master.data_ptr() and pc[i].data_ptr() < flat.flat.master.data_ptr() + 4 * flat.flat.total
                                          for i in range(len(SHAPES)))
        # state_dict format: same keys as torch AdamW, dead param absent, step tensor
        sd = oc.state_dict()
        sdt = ob.state_dict()
        res["state_dict_keys_match"] = (sorted(sd.keys()) == sorted(sdt.keys())
                                       and [sorted(k for k in g if k != "params") for g in sd["param_groups"]] == [sorted(k for k in g if k != "params") for g in sdt["param_groups"]])
        res["state_entries"] = (len(sd["state"]), len(sdt["state"]))
        res["state_entry_keys"] = sorted(next(iter(sd["state"].values())).keys())
        res["state_dict_lr_equal_host"] = max(abs(float(a["lr"]) - float(b["lr"])) for a, b in zip(sd["param_groups"], sdt["param_groups"]))
        res["state_dict_step"] = (float(next(iter(sd["state"].values()))["step"]), float(next(iter(sdt["state"].values()))["step"]))
        res["scheduler_last_epoch"] = (sc.last_epoch, sb.last_epoch)

        # ---------------- checkpoint compatibility (10 more steps from a step-50 checkpoint)
        def run_continue(sd_opt, p_init, use_flat: bool):
            ps = [torch.nn.Parameter(t.clone()) for t in p_init]
            o, s = make_opt(ps, fused=not use_flat)
            s.last_epoch = 50
            s._last_lr = s.get_lr()
            fl = None
            if use_flat:
                fl = build_flat(o, s, {id(p): f"p{i}" for i, p in enumerate(ps)}, MAX_NORM)
                install_checkpoint_compat(fl, o, s)
                fl.graph_mode = True
            o.load_state_dict(sd_opt)                      # flat: through the compat layer; torch: native
            if not use_flat:
                for g, lr in zip(o.param_groups, s.get_lr()):
                    g["lr"] = lr
            for step in range(50, 60):
                gs = grads_for(step)
                for i in range(len(SHAPES)):
                    if i == DEAD_INDEX:
                        continue
                    gi = gs[i].clone().to(memory_format=torch.channels_last) if i == CL_INDEX else gs[i].clone()
                    if use_flat and fl.adopted:
                        ps[i].grad.add_(gi)
                    else:
                        ps[i].grad = gi
                if use_flat:
                    fl.step()
                else:
                    torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], MAX_NORM, error_if_nonfinite=False)
                    o.step(); s.step()
            return ps
        # state restore is exact: flat ckpt -> torch AdamW state, torch ckpt -> flat buffers
        _ps = [torch.nn.Parameter(t.clone()) for t in snap50[0]]
        _o, _s = make_opt(_ps, fused=True)
        _o.load_state_dict(sd_flat)
        res["ckpt_flat_to_torch_state_bitwise"] = all(torch.equal(_o.state[_ps[i]]["exp_avg"], sd_flat["state"][_pid]["exp_avg"])
                                                       and float(_o.state[_ps[i]]["step"]) == 50.0
                                                       for _pid, i in enumerate([j for g in GROUPS for j in g[0]]) if _pid in sd_flat["state"])
        _ps2 = [torch.nn.Parameter(t.clone()) for t in snap50[1]]
        _o2, _s2 = make_opt(_ps2)
        _s2.last_epoch = 50
        _f2 = build_flat(_o2, _s2, None, MAX_NORM)
        install_checkpoint_compat(_f2, _o2, _s2)
        _o2.load_state_dict(sd_torch)
        _fid = {id(p): j for j, p in enumerate(_f2.flat.params)}
        _ok = True
        for _pid, i in enumerate([j for g in GROUPS for j in g[0]]):
            if _pid in sd_torch["state"]:
                _m, _v = _f2.param_state(_fid[id(_ps2[i])], clone=False)
                _ok = _ok and torch.equal(_m, sd_torch["state"][_pid]["exp_avg"]) and torch.equal(_v, sd_torch["state"][_pid]["exp_avg_sq"])
        res["ckpt_torch_to_flat_state_bitwise"] = bool(_ok and _f2.device_counters() == (50, 50))
        ps_t = run_continue(sd_flat, snap50[0], use_flat=False)       # flat checkpoint -> torch AdamW(fused)
        ps_f = run_continue(sd_torch, snap50[1], use_flat=True)       # torch checkpoint -> flat
        res["ckpt_flat_to_torch_maxdiff_vs_live_flat"] = max(_maxdiff(a, b) for a, b in zip(ps_t, snap60[0]))
        res["ckpt_torch_to_flat_maxdiff_vs_live_torch"] = max(_maxdiff(a, b) for a, b in zip(ps_f, snap60[1]))
        # ---------------- determinism: two identical runs, bitwise
        outs = []
        for _ in range(2):
            ps = make_params(0)
            o, s = make_opt(ps)
            fl = build_flat(o, s, None, MAX_NORM)
            fl.graph_mode = True
            norms = []
            for step in range(20):
                gs = grads_for(step)
                for i in range(len(SHAPES)):
                    if i == DEAD_INDEX:
                        continue
                    gi = gs[i].clone().to(memory_format=torch.channels_last) if i == CL_INDEX else gs[i].clone()
                    if fl.adopted:
                        ps[i].grad.add_(gi)
                    else:
                        ps[i].grad = gi
                norms.append(float(fl.step()))
            outs.append((norms, fl.flat.master.clone(), fl.flat.m.clone(), fl.flat.v.clone()))
        res["deterministic_bitwise"] = bool(outs[0][0] == outs[1][0] and torch.equal(outs[0][1], outs[1][1]) and torch.equal(outs[0][2], outs[1][2]) and torch.equal(outs[0][3], outs[1][3]))
    # ---------------- verdict
    w = res["trajectory_worst"]
    res["pass"] = bool(w["param_vs_ref"] <= 1e-6 and w["param_vs_fused"] <= 1.25 * w["fused_vs_ref"] + 1e-6 and w["m_vs_fused"] <= 1e-6 and w["v_vs_fused"] <= 1e-6
                       and w["norm_vs_ref_rel"] <= 1e-6 and w["norm_vs_fused_rel"] <= 1e-6 and w["lr_rel"] <= 1e-6
                       and res["dead_param_unchanged"] and res["grad_buffer_zero_after_step"] and res["deterministic_bitwise"]
                       and res["state_dict_keys_match"] and res["ckpt_flat_to_torch_maxdiff_vs_live_flat"] <= 1e-5
                       and res["ckpt_torch_to_flat_maxdiff_vs_live_torch"] <= 1e-5
                       and res["ckpt_flat_to_torch_state_bitwise"] and res["ckpt_torch_to_flat_state_bitwise"])
    return res


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=1, default=str))
    out_path = Path(__file__).resolve().parents[1] / "profile" / "synthetic_gate.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    sys.exit(0 if out["pass"] else 1)
