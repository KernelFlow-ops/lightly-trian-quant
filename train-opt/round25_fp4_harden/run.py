"""round25_fp4_harden entrypoint: hardening measurements of the phase-3 final stack P3 (FINAL + round21 FP4 head + round22 FP4 backbone); NO new kernels.
Adapted from round23_fp4_final/run.py (same bench / profile / gate_s / graph_equiv / e2e actions of round16 / round23, outputs redirected to round25_fp4_harden/profile).

Presets (``--stack`` / ``--ref``; see impl/presets25.py): everything of round23 (P3, P3H, P3B, P3_RN, P3_q8, FINAL, R0, ...) plus
    P3_hr16 / P3_hr32 / P3_hr64   P3 with backbone dY headroom 16 / 32 / 64 (default 8)       P3_ex   P3 with amax=exact in both patches
    P3_H2 / P3_H2f                P3 backbone + head ``head.dw=mx8`` / ``head.f=mx8,head.dw=mx8`` (ViT-B conservative tier candidates)

Actions (every GPU action holds the dual lock):
    bench          0.2 / 0.6 / 0.2 weighted early / mid / late timing under StepGraphRunner (round23 logic)
    profile / graph_equiv / e2e   as round23
    gate_s         round23's S gate (one --stack under graph vs --ref R0 eager and vs FINAL graph)
    gate_s_sweep   S gate for SEVERAL stacks (--stacks) in ONE process on the same 2 batches: R0 eager once per scenario, then every stack under graph; per-site FP4 saturation stats
    sat_study      saturation diagnosis: same process, same 2 batches, same seed: per-step per-site amax time series + per-site exceed counts of each (stack, repeat)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["QUACK_CACHE_DIR"] = "/root/autodl-tmp/train-opt-tmp/round25_fp4_harden/quack_cache"   # seeded from round23's cache; must precede any quack import
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
from common.check import gate_c  # noqa: E402
from common.config import B_AUTO  # noqa: E402
from common.harness import apply_patches, build, preload_batches, set_phase, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round19_final import run as r19run  # noqa: E402  (registers r19's presets and redirects to r19: undone by register() below)
from round16_full_graph import run as r16  # noqa: E402

NAME = "round25_fp4_harden"
R21, R22 = p23.R21, p23.R22


def _san(o):
    """JSON-safe copy: non-finite floats (inf amax ratios, NaN losses of a diverged run) become strings instead of making ``allow_nan=False`` raise."""
    if isinstance(o, float):
        return o if math.isfinite(o) else str(o)
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    return o


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(_san(obj), indent=2, default=str, allow_nan=False) + "\n")


def register(model: str) -> None:
    p23.register(ROUND, model)
    r19run.ROUND = ROUND                                   # r19's own actions write to ``ROUND / profile``


# ------------------------------------------------------------------ FP4 statistics of both patches
def summarize_head_stats(stats: dict) -> dict:
    """Group round21's ``FP4State.stats()`` by site class: sites 4c+2j = X column copy, 4c+2j+1 = dZ (row copy for dgrad, col copy for wgrad), the last two = weights (row slot)."""
    n = stats["n_sites"]
    n_calls = (n - 2) // 8
    groups = {"X_col": [(s, "col") for s in range(4 * n_calls) if s % 2 == 0], "dZ_row": [(s, "row") for s in range(4 * n_calls) if s % 2 == 1],
              "dZ_col": [(s, "col") for s in range(4 * n_calls) if s % 2 == 1], "W_row": [(4 * n_calls + j, "row") for j in range(2)]}
    out = {"step": stats["step"], "total_calls": stats["total_calls"], "total_exceed": stats["total_exceed"], "exceed_rate": stats["exceed_rate"]}
    for name, sl in groups.items():
        calls = sum(stats[c]["calls"][s] for s, c in sl)
        exc = sum(stats[c]["exceed"][s] for s, c in sl)
        ratios = [stats[c]["max_ratio"][s] for s, c in sl]
        out[name] = {"calls": calls, "exceed": exc, "exceed_rate": exc / max(1, calls), "max_ratio": max(ratios) if ratios else 0.0, "headroom": stats["headroom"][sl[0][0]] if sl else None}
    return out


def collect_fp4(ctx) -> dict:
    """Host-side saturation statistics of both FP4 states (one device sync each; never inside a graph)."""
    out: dict = {}
    h21, h22 = ctx.meta.get(R21), ctx.meta.get(R22)
    if h21 is not None:
        out["head"] = {"policy": h21.policy.describe(), **summarize_head_stats(h21.state.stats())}
    if h22 is not None:
        out["backbone"] = {"policy": h22.describe(), **h22.stats()}
    return out


def release_fp4(ctx) -> None:
    """Break the ctx <-> handle cycles and drop the per-step weight copies so the context is freed between scenarios."""
    h21, h22 = ctx.meta.get(R21), ctx.meta.get(R22)
    if h21 is not None:
        h21.wq = []
        if hasattr(h21, "ctx"):
            h21.ctx = None
    if h22 is not None:
        h22.ctx = None
        h22.weights = {}
        h22.layers = {}


# ------------------------------------------------------------------ gate S curves (round16 ``curve_run`` logic + the FP4 summary of both patches)
def curve_run23(cfg, batches, steps: int, scen: str, mode: str, args) -> dict:
    mem0 = torch.cuda.memory_allocated() / 2**30
    ctx = build(cfg)
    apply_patches(ctx)
    ctx.rng = RngTape(mode="off")
    r16.set_scenario_state(ctx, scen)
    runner = None
    if mode == "graph":
        runner = r16.make_runner(ctx, batches, args)
        runner.prepare(batches[0])
        r16.set_scenario_state(ctx, scen)
        runner.sync_probs_state()
    losses, norms, ema, value = [], [], [], None
    for step in range(steps):
        if runner is not None:
            out = runner.step(batches[step % len(batches)], step, batches[(step + 1) % len(batches)])
        else:
            out = train_step(ctx, batches[step % len(batches)], step)
        loss = float(out.loss.float().item())
        losses.append(loss)
        norms.append(None if out.grad_norm is None else float(out.grad_norm.float().item()))
        value = loss if value is None else 0.9 * value + 0.1 * loss
        ema.append(value)
    finite = all(math.isfinite(v) for v in losses) and all(v is None or math.isfinite(v) for v in norms)
    extra: dict = {}
    if runner is not None:
        extra["graph"] = runner.report.as_dict()
        runner.release()
    fp4 = collect_fp4(ctx)
    if fp4:
        extra["fp4"] = fp4
    release_fp4(ctx)
    del ctx, runner
    r16.free_gpu()
    extra["mem_gb_before_after"] = [round(mem0, 2), round(torch.cuda.memory_allocated() / 2**30, 2)]
    print(f"  [{mode} {cfg.patches[-1] if cfg.patches else 'R0'}] allocated GB before/after: {extra['mem_gb_before_after']}", flush=True)
    return {"steps": steps, "loss": losses, "loss_ema": ema, "grad_norm": norms, "finite": finite, **extra}


def action_gate_s(args) -> None:
    from common.harness import build as _build

    cfg_c = r16.make_cfg(args.model, args.stack, seed=0)
    cfg_r = r16.make_cfg(args.model, args.ref, seed=0)
    cfg_f = r16.make_cfg(args.model, "FINAL", seed=0)
    report = {"command": [sys.executable, *sys.argv], "ref": args.ref, "stack": args.stack, "runner": args.runner, "q2": p23.Q2_ACTIVE, "runs": {}}
    out = r16.unique_path(ROUND / "profile" / f"gateS_{args.model}_{r16.safe_name(args.stack)}_{args.runner}_vs_{r16.safe_name(args.ref)}.json")
    with bench.gpu_lock():
        r16.wait_exclusive()
        loader_ctx = _build(cfg_c)
        batches = preload_batches(loader_ctx, k=2)
        del loader_ctx
        r16.free_gpu()
        for scen in args.scenarios:
            t0 = time.time()
            ref = curve_run23(cfg_r, batches, args.steps, scen, "eager", args)
            cand = curve_run23(cfg_c, batches, args.steps, scen, args.runner, args)
            gate = gate_c(ref, cand, interval=10, max_rel=0.05)
            row = {"gate": gate, "ref": ref, "cand": cand}
            if not args.no_final:
                fin = curve_run23(cfg_f, batches, args.steps, scen, "graph", args)
                row["final"] = fin
                row["gate_vs_final"] = gate_c(fin, cand, interval=10, max_rel=0.05)
            report["runs"][scen] = row
            ema_all = max(abs(c - r) / abs(r) for c, r in zip(cand["loss_ema"], ref["loss_ema"]))
            msg = {"max_rel": gate["curve300_max_rel"], "pass": gate["pass"], "finite": gate["finite"], "ema_max_rel_all_steps": round(ema_all, 5), "checkpoints": gate["checkpoints"],
                   "grad_norm_ratios": gate["grad_norm_ratios"], "loss_first": [ref["loss"][0], cand["loss"][0]], "loss_last": [ref["loss"][-1], cand["loss"][-1]],
                   "captures": cand.get("graph", {}).get("captures"), "seconds": round(time.time() - t0, 1)}
            if "gate_vs_final" in row:
                g2 = row["gate_vs_final"]
                msg["vs_final"] = {"max_rel": g2["curve300_max_rel"], "pass": g2["pass"], "grad_norm_ratios": g2["grad_norm_ratios"]}
            print(scen, json.dumps(msg), flush=True)
            for k, v in cand.get("fp4", {}).items():
                brief = {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("step", "total_calls", "total_exceed", "exceed_rate")}
                print(f"  fp4 {k}", json.dumps(brief), json.dumps(v.get("per_kind", {k2: v[k2] for k2 in ("dZ_row", "dZ_col") if k2 in v})), flush=True)
            dump(out, report)
    print(f"wrote {out}", flush=True)


# ------------------------------------------------------------------ round25: multi-stack S gate (headroom sweep) and the saturation study
def _r5(v: float) -> float:
    return float(f"{v:.5g}")


def raw_site_stats(ctx) -> dict:
    """Per-site raw saturation counters of the backbone state (host; one sync): exceed / max cur-prev ratio of the row and column copy of every site, + the per-site headroom."""
    h22 = ctx.meta.get(R22)
    if h22 is None:
        return {}
    st = h22.state.stats()
    return {"site_names": h22.site_names, "headroom": st["headroom"], "calls": st["row"]["calls"], "exceed_row": st["row"]["exceed"], "exceed_col": st["col"]["exceed"],
            "maxr_row": st["row"]["max_ratio"], "maxr_col": st["col"]["max_ratio"], "active_sites": h22.active_sites()}


def curve_run25(cfg, batches, steps: int, scen: str, mode: str, args, trace: bool = False) -> dict:
    """round23's ``curve_run23`` + (optionally) the per-step amax table of the backbone FP4 state (``trace``: the amax of every site's row / col copy after each step, and the
    initial one = the exact amax of the eager warm-up) + the raw per-site saturation counters."""
    mem0 = torch.cuda.memory_allocated() / 2**30
    ctx = build(cfg)
    apply_patches(ctx)
    ctx.rng = RngTape(mode="off")
    r16.set_scenario_state(ctx, scen)
    runner = None
    if mode == "graph":
        runner = r16.make_runner(ctx, batches, args)
        runner.prepare(batches[0])
        r16.set_scenario_state(ctx, scen)
        runner.sync_probs_state()
    h22 = ctx.meta.get(R22)
    tr = None
    if trace and h22 is not None:
        torch.cuda.synchronize()
        tr = {"init": [[_r5(a), _r5(b)] for a, b in h22.state.amax[0].cpu().tolist()], "amax": []}
    losses, norms, ema, value = [], [], [], None
    for step in range(steps):
        if runner is not None:
            out = runner.step(batches[step % len(batches)], step, batches[(step + 1) % len(batches)])
        else:
            out = train_step(ctx, batches[step % len(batches)], step)
        loss = float(out.loss.float().item())
        losses.append(loss)
        norms.append(None if out.grad_norm is None else float(out.grad_norm.float().item()))
        value = loss if value is None else 0.9 * value + 0.1 * loss
        ema.append(value)
        if tr is not None:
            tr["amax"].append([[_r5(a), _r5(b)] for a, b in h22.state.amax[0].cpu().tolist()])
    finite = all(math.isfinite(v) for v in losses) and all(v is None or math.isfinite(v) for v in norms)
    extra: dict = {}
    if runner is not None:
        extra["graph"] = runner.report.as_dict()
        runner.release()
    fp4 = collect_fp4(ctx)
    if fp4:
        extra["fp4"] = fp4
    raw = raw_site_stats(ctx)
    if raw:
        extra["raw"] = raw
    if tr is not None:
        extra["trace"] = tr
    release_fp4(ctx)
    del ctx, runner, h22
    r16.free_gpu()
    extra["mem_gb_before_after"] = [round(mem0, 2), round(torch.cuda.memory_allocated() / 2**30, 2)]
    return {"steps": steps, "loss": losses, "loss_ema": ema, "grad_norm": norms, "finite": finite, **extra}


def brief_fp4(cand: dict) -> dict:
    """one-line saturation summary of a candidate run: backbone dY (exceed, calls, max ratio, sites), head (exceed, calls), x / w exceed."""
    bb = cand.get("fp4", {}).get("backbone", {}).get("per_kind", {})
    hd = cand.get("fp4", {}).get("head", {})
    dy = bb.get("dy", {})
    return {"dy_exceed": dy.get("exceed"), "dy_calls": dy.get("calls"), "dy_max_ratio": dy.get("max_ratio"), "dy_sites_exceeding": dy.get("sites_exceeding"),
            "x_exceed": bb.get("x", {}).get("exceed"), "w_exceed": bb.get("w", {}).get("exceed"), "head_exceed": hd.get("total_exceed"), "head_calls": hd.get("total_calls")}


def action_gate_s_sweep(args) -> None:
    """S gate for several stacks in one process: the same 2 batches for all, R0 (eager) once per scenario, every stack under ``--runner`` (graph); the report is rewritten after every run."""
    cfg_r = r16.make_cfg(args.model, args.ref, seed=0)
    cfgs = {s: r16.make_cfg(args.model, s, seed=0) for s in args.stacks}
    report = {"command": [sys.executable, *sys.argv], "model": args.model, "ref": args.ref, "stacks": list(args.stacks), "runner": args.runner, "n_batches": args.n_batches, "runs": {}}
    out = r16.unique_path(ROUND / "profile" / f"gateS_sweep_{args.model}_{args.tag}.json")
    with bench.gpu_lock():
        r16.wait_exclusive()
        loader_ctx = build(cfgs[args.stacks[0]])
        batches = preload_batches(loader_ctx, k=args.n_batches)
        report["batch_sum"] = [float(b["image"].double().sum()) for b in batches]
        del loader_ctx
        r16.free_gpu()
        for scen in args.scenarios:
            t0 = time.time()
            ref = curve_run25(cfg_r, batches, args.steps, scen, "eager", args)
            row = {"ref": {k: ref[k] for k in ("loss", "loss_ema", "grad_norm", "finite")}, "stacks": {}}
            report["runs"][scen] = row
            for s in args.stacks:
                t1 = time.time()
                cand = curve_run25(cfgs[s], batches, args.steps, scen, args.runner, args, trace=args.trace)
                gate = gate_c(ref, cand, interval=10, max_rel=0.05)
                ema_all = max(abs(c - r) / abs(r) for c, r in zip(cand["loss_ema"], ref["loss_ema"]))
                gn = [c / r for c, r in zip(cand["grad_norm"], ref["grad_norm"]) if c is not None and r]
                row["stacks"][s] = {"gate": gate, "cand": cand, "ema_max_rel_all_steps": ema_all, "grad_norm_ratio_all_steps": [min(gn), max(gn)], "brief": brief_fp4(cand)}
                msg = {"stack": s, "max_rel": round(gate["curve300_max_rel"], 5), "pass": gate["pass"], "finite": gate["finite"], "ema_all": round(ema_all, 5),
                       "gn_ratio_ckpt": gate["grad_norm_ratios"], "gn_all": [round(min(gn), 3), round(max(gn), 3)], "sat": row["stacks"][s]["brief"], "loss_last": cand["loss"][-1],
                       "s": round(time.time() - t1, 1)}
                print(scen, json.dumps(_san(msg)), flush=True)
                dump(out, report)
            print(f"== {scen} done {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {out}", flush=True)


def action_sat_study(args) -> None:
    """Saturation diagnosis: the same 2 (or ``--n-batches``) batches, same seed; each entry of ``--seq`` (a stack name, repeats allowed) is run for every scenario with the per-step amax trace."""
    cfgs = {s: r16.make_cfg(args.model, s, seed=0) for s in set(args.seq)}
    report = {"command": [sys.executable, *sys.argv], "model": args.model, "seq": list(args.seq), "n_batches": args.n_batches, "runs": {}}
    out = r16.unique_path(ROUND / "profile" / f"sat_study_{args.model}_{args.tag}.json")
    with bench.gpu_lock():
        r16.wait_exclusive()
        loader_ctx = build(cfgs[args.seq[0]])
        batches = preload_batches(loader_ctx, k=args.n_batches)
        report["batch_sum"] = [float(b["image"].double().sum()) for b in batches]
        del loader_ctx
        r16.free_gpu()
        for scen in args.scenarios:
            for i, s in enumerate(args.seq):
                t1 = time.time()
                cand = curve_run25(cfgs[s], batches, args.steps, scen, args.runner, args, trace=True)
                key = f"{scen}|{i}|{s}"
                report["runs"][key] = {"scen": scen, "idx": i, "stack": s, "cand": cand, "brief": brief_fp4(cand)}
                print(key, json.dumps(_san({"sat": report["runs"][key]["brief"], "loss_last": cand["loss"][-1], "s": round(time.time() - t1, 1)})), flush=True)
                dump(out, report)
    print(f"wrote {out}", flush=True)


# ------------------------------------------------------------------ graph_equiv with BOTH FP4 states
def snap_fp4(s) -> dict:
    return {"amax": s.amax.clone(), "seed": s.seed.clone(), "step": s.step.clone(), "inited": list(s.inited), "headroom": s.headroom.clone(),
            "mon": [getattr(s, n).clone() for n in ("mon_cnt", "mon_maxr") if hasattr(s, n)]}


def restore_fp4(s, snap: dict) -> None:
    s.amax.copy_(snap["amax"])
    s.seed.copy_(snap["seed"])
    s.step.copy_(snap["step"])
    s.headroom.copy_(snap["headroom"])
    for n, v in zip(("mon_cnt", "mon_maxr"), snap["mon"]):
        getattr(s, n).copy_(v)
    s.inited[:] = snap["inited"]


def action_graph_equiv(args) -> None:
    from round16_full_graph.impl.graph_step import padded_targets

    cfg = r16.make_cfg(args.model, args.stack, seed=args.seed)
    results: dict = {"command": [sys.executable, *sys.argv], "stack": args.stack, "q2": p23.Q2_ACTIVE, "scenarios": {}}
    out = r16.unique_path(ROUND / "profile" / f"graph_equiv_{args.model}_{r16.safe_name(args.stack)}.json")
    with bench.gpu_lock():
        r16.wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        ctx.rng = RngTape(mode="off")
        batches = preload_batches(ctx, k=2)
        flat = ctx.meta["round15_optimizer_flat"]
        h21, h22 = ctx.meta.get(R21), ctx.meta.get(R22)
        states = {k: h.state for k, h in (("head", h21), ("backbone", h22)) if h is not None}
        padded = [{"image": b["image"], "binary_masks": padded_targets(b, args.max_targets)} for b in batches]
        init = r16._snapshot(ctx, flat)
        init_cpu = init["master"].cpu()
        K = args.equiv_steps
        for scen in args.scenarios:
            row: dict = {}
            # (a) warm every signature of the scenario eagerly (exact init of the delayed-amax sites of both patches), then snapshot both FP4 states
            r16._restore(flat, init)
            for s in states.values():
                s.reset()
            if h21 is not None:
                h21.reset_state(full=True)
            r16.set_scenario_state(ctx, scen)
            runner = r16.make_runner(ctx, batches, args)
            runner.prepare(batches[0])
            if h22 is not None:
                assert all(h22.state.inited[i] for i in h22.active_sites()), "warm-up did not initialise every delayed-amax site of the backbone"
                row["backbone_active_sites"] = len(h22.active_sites())
            if h21 is not None:
                row["head_sites_inited"] = [bool(x) for x in h21.state.inited].count(True)
                assert all(h21.state.inited), "warm-up did not initialise every delayed-amax site of the head"
            snaps = {k: snap_fp4(s) for k, s in states.items()}

            def reset_all():
                r16._restore(flat, init)
                for k, s in states.items():
                    restore_fp4(s, snaps[k])
                if h21 is not None:
                    h21.calls = 0
                    h21.wq = []
                r16.set_scenario_state(ctx, scen)

            def advance():
                return {k: int(s.step.item()) - int(snaps[k]["step"].item()) for k, s in states.items()}

            def run_eager():
                reset_all()
                ls, gs = [], []
                for k in range(K):
                    torch.cuda.manual_seed(7000 + k)
                    o = train_step(ctx, padded[k % 2], k)
                    ls.append(o.loss.clone())
                    gs.append(o.grad_norm.clone())
                torch.cuda.synchronize()
                return [float(x) for x in ls], [float(x) for x in gs], flat.flat.master.cpu(), advance()

            e1 = run_eager()
            e2 = run_eager()
            r16.free_gpu()
            reset_all()
            runner.sync_probs_state()
            gl, gg = [], []
            for k in range(K):
                torch.cuda.manual_seed(7000 + k)
                o = runner.step(batches[k % 2], k, batches[(k + 1) % 2])
                gl.append(o.loss.clone())
                gg.append(o.grad_norm.clone())
            torch.cuda.synchronize()
            gl, gg = [float(x) for x in gl], [float(x) for x in gg]
            gmaster = flat.flat.master.cpu()
            adv_graph = advance()
            row["steps"] = K
            row["fp4_state_advance"] = {"eager1": e1[3], "eager2": e2[3], "graph": adv_graph, "expected_each": K,
                                        "ok": bool(all(v == K for d in (e1[3], e2[3], adv_graph) for v in d.values()) and len(adv_graph) == len(states))}
            row["eager_vs_eager"] = r16._delta_stats(e1[2], e2[2], init_cpu)
            row["graph_vs_eager"] = r16._delta_stats(gmaster, e1[2], init_cpu)
            row["loss_eager"], row["loss_eager2"], row["loss_graph"] = e1[0], e2[0], gl
            row["gnorm_eager"], row["gnorm_graph"] = e1[1], gg
            row["loss_rel_max_graph_vs_eager"] = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(gl, e1[0]))
            row["loss_rel_max_eager_vs_eager"] = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(e2[0], e1[0]))
            row["gnorm_rel_max_graph_vs_eager"] = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(gg, e1[1]))
            row["loss_bitwise_first_step"] = bool(gl[0] == e1[0][0])
            row["graph"] = runner.report.as_dict()
            runner.release()
            del runner
            r16.free_gpu()
            results["scenarios"][scen] = row
            print(scen, json.dumps({k: row[k] for k in ("steps", "fp4_state_advance", "eager_vs_eager", "graph_vs_eager", "loss_rel_max_graph_vs_eager", "loss_rel_max_eager_vs_eager",
                                                        "gnorm_rel_max_graph_vs_eager", "loss_bitwise_first_step")}), flush=True)
            print("  captures", row["graph"]["captures"], "capture_s", [round(x, 2) for x in row["graph"]["capture_s"]], "keys", row["graph"]["keys"], flush=True)
            dump(out, results)
        release_fp4(ctx)
    print(f"wrote {out}", flush=True)


# ------------------------------------------------------------------ bench (round16 ``action_bench`` + reserved memory)
def device_mem() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {"device_used_gb": (total - free) / 2**30, "allocated_gb": torch.cuda.memory_allocated() / 2**30, "reserved_gb": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}


def action_bench(args) -> None:
    cfg = r16.make_cfg(args.model, args.stack, seed=args.seed)
    phases = ("early", "mid", "late") if args.phases == "all" else (args.phases,)
    with bench.gpu_lock():
        r16.wait_exclusive()
        t0 = time.time()
        ctx = build(cfg)
        apply_patches(ctx)
        batches = preload_batches(ctx, k=args.preload)
        build_s = time.time() - t0
        step_fn, runner = r16.make_step(ctx, batches, args)
        graph_prep_s = time.time() - t0 - build_s
        mem_setup = device_mem()
        measured = bench.time_steps(ctx, batches, phases=phases, warmup=args.warmup, iters=args.iters, repeats=args.repeats, segments=False, step_fn=step_fn)
        mem_end = device_mem()
        env = bench.env_info()
        graph_report = runner.report.as_dict() if runner is not None else None
        if runner is not None:
            runner.sync_probs_state()
        fp4 = collect_fp4(ctx)
    name = args.name or f"{args.stack}|{args.runner}"
    summary = {
        "model": args.model, "name": name, "stack": args.stack, "runner": args.runner, "q2": p23.Q2_ACTIVE, "build_s": build_s, "graph_prep_s": graph_prep_s,
        "patches": list(cfg.patches), "patch_kwargs": cfg.patch_kwargs,
        "step_ms_mean": {p: measured[p]["step_ms"]["mean"] for p in phases}, "img_s": {p: measured[p]["img_s"] for p in phases}, "weighted_img_s": measured["weighted_img_s"],
        "repeat_ms_mean": {p: [r["step_ms"]["mean"] for r in measured[p]["repeats"]] for p in phases},
        "mem": {"after_setup": mem_setup, "after_bench": mem_end}, "peak_reserved_gb": mem_end["peak_reserved_gb"], "graph": graph_report,
        "fp4_stats": {k: {kk: v.get(kk) for kk in ("policy", "step", "total_calls", "total_exceed", "exceed_rate")} for k, v in fp4.items()} if fp4 else None,
    }
    out = r16.unique_path(ROUND / "profile" / f"bench_{args.model}_{r16.safe_name(name)}_{args.phases}.json")
    dump(out, {"summary": summary, "measurement": measured, "env": env, "command": [sys.executable, *sys.argv]})
    print(json.dumps({k: summary[k] for k in ("model", "name", "step_ms_mean", "img_s", "weighted_img_s", "peak_reserved_gb", "repeat_ms_mean")}, default=str), flush=True)
    print(f"wrote {out}", flush=True)


# ------------------------------------------------------------------ end to end with the real loader
def action_e2e(args) -> None:
    from lightly_train._data.infinite_cycle_iterator import InfiniteCycleIterator

    from round25_fp4_harden import run_final as rf

    cfg = rf.final_cfg(args.model, batch=args.batch or None, epochs=20, workers=max(args.worker_list), seed=args.seed, stack=args.stack)
    settings = [dict(zip(("workers", "prefetch", "pin", "persistent"), s)) for s in (tuple(x.split(":")) for x in args.loader_settings)]
    for s in settings:
        s["workers"], s["prefetch"], s["pin"], s["persistent"] = int(s["workers"]), int(s["prefetch"]), s["pin"] == "1", s["persistent"] == "1"
    report: dict = {"command": [sys.executable, *sys.argv], "model": args.model, "stack": args.stack, "q2": p23.Q2_ACTIVE, "batch": cfg.batch_size, "steps": args.steps, "runs": []}
    with bench.gpu_lock():
        r16.wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        batches = preload_batches(ctx, k=2)
        runner = rf.make_graph_runner(ctx, batches[0], run_steps=args.steps + args.warmup + 200)
        # GPU-only reference in the same process: preloaded batches, early phase (the first steps of the real schedule are all 'one')
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
            stats = rf.train_loop(ctx, runner, loader, iterator, steps=args.steps, start_step=0, warm_steps=args.warmup, log_every=0, workers=s["workers"], quiet=True)
            row = {"setting": s, **{k: v for k, v in stats.items() if k != "losses"}, "seconds": time.time() - t0}
            row["steady_vs_gpu_only"] = row.get("img_s_steady", 0) / report["gpu_only_early"]["img_s"]
            report["runs"].append(row)
            print("e2e", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            del loader, iterator
            r16.free_gpu()
        report["graph"] = runner.report.as_dict()
        report["mem"] = device_mem()
        report["fp4_stats"] = {k: {kk: v.get(kk) for kk in ("policy", "step", "total_calls", "total_exceed", "exceed_rate")} for k, v in collect_fp4(ctx).items()}
        runner.release()
    out = r16.unique_path(ROUND / "profile" / f"e2e_{args.model}_{args.tag}.json")
    dump(out, report)
    print(f"wrote {out}", flush=True)


def action_profile(args) -> None:
    r16.action_profile(args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=tuple(B_AUTO), required=True)
    ap.add_argument("--stack", default="P3")
    ap.add_argument("--name")
    ap.add_argument("--action", choices=("bench", "profile", "gate_s", "graph_equiv", "e2e", "gate_s_sweep", "sat_study"), required=True)
    ap.add_argument("--runner", choices=("eager", "graph"), default="graph")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--ref", default="R0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--scenarios", nargs="*", default=["schedule", "compressed", "fixed_mid", "fixed_late"])
    ap.add_argument("--no-final", action="store_true", help="gate_s: skip the FINAL graph curves (the vs-FINAL deviation)")
    ap.add_argument("--equiv-steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--preload", type=int, default=2)
    ap.add_argument("--batch", type=int, default=0, help="e2e: batch size (0 = B_AUTO)")
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--max-resident", type=int, default=8)
    ap.add_argument("--metrics", choices=("step", "lazy", "off"), default="lazy")
    ap.add_argument("--no-side-stream", action="store_true")
    ap.add_argument("--cpu-batches", action="store_true")
    ap.add_argument("--no-segments", action="store_true")
    ap.add_argument("--no-results", action="store_true", default=True)
    ap.add_argument("--loader-settings", nargs="*", default=["16:2:1:1"], help="e2e: workers:prefetch:pin(0/1):persistent(0/1)")
    ap.add_argument("--tag", default="x")
    ap.add_argument("--stacks", nargs="*", default=["P3"], help="gate_s_sweep: stacks run under --runner on the same batches")
    ap.add_argument("--seq", nargs="*", default=["P3B", "P3", "P3B", "P3"], help="sat_study: sequence of stacks (repeats allowed)")
    ap.add_argument("--n-batches", type=int, default=2, help="gate_s_sweep / sat_study: number of preloaded batches cycled by the steps")
    ap.add_argument("--trace", action="store_true", help="gate_s_sweep: also store the per-step backbone amax table of each stack")
    args = ap.parse_args()
    args.worker_list = [int(s.split(":")[0]) for s in args.loader_settings]
    register(args.model)
    if args.action != "bench" and args.phases == "all":
        args.phases = "mid"
    {"bench": action_bench, "profile": action_profile, "gate_s": action_gate_s, "graph_equiv": action_graph_equiv, "e2e": action_e2e, "gate_s_sweep": action_gate_s_sweep,
     "sat_study": action_sat_study}[args.action](args)


if __name__ == "__main__":
    sys.path.insert(0, str(ROUND))
    main()
