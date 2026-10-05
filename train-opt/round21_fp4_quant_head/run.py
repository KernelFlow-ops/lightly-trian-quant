"""round21_fp4_quant_head entrypoint (R21b): bench / profile / gate_s / graph_equiv for FINAL + the real-kernel FP4 head.  Thin wrapper around round19_final/run.py (which wraps
round16_full_graph/run.py); every output is redirected to round21_fp4_quant_head/ (profile/, tmp).

Presets (``--stack`` / ``--ref``), all FINAL (= common.config.BEST_STACK_V2) + the round21 patch with a round20b policy string (``impl/head_fp4.py``):
    P3H        head.f=mx8,head.dw=nv4+sr     (CONTRACTS 6b 13.2 strategy, H3)
    P3H_H1     head.dw=nv4+sr                (forward stays bf16)
    P3H_H2     head.dw=mx8                   (r14's MXFP8 backward, the control)
    P3H_RN     head.f=mx8,head.dw=nv4+rn     (no SR)
    P3H_EX     head.f=mx8,head.dw=nv4+sr,amax=exact   (exact per-tensor amax instead of delayed)
    P3H_HR2    P3H with dy_headroom 2 (the pre-13.5 default; for the saturation study)
    P3H_RNEX   head.f=mx8,head.dw=nv4+rn,amax=exact   (bit-aligned with the round20b simulated head)
plus everything of r19 (R0, TB1/P1, P2, P2R15, P2R14, FINAL, ...).

Actions (every GPU action holds the dual lock):  bench | profile | gate_s | graph_equiv   (same as r19 / r16; graph_equiv here resets the FP4 state between runs)
    python run.py --model vitl16 --stack P3H --action bench --runner graph --phases all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QUACK_CACHE_DIR", "/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/quack_cache")
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROUND = Path(__file__).resolve().parent

import torch  # noqa: E402

from common import bench  # noqa: E402
from common.config import B_AUTO, P2R14R15_PATCH_KWARGS, P2R14R15_STACK  # noqa: E402
from common.harness import apply_patches, build, preload_batches, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round19_final import run as r19run  # noqa: E402  (registers r19's presets, redirects to r19: undone by register() below)
from round19_final.impl import presets as p19  # noqa: E402
from round16_full_graph import run as r16  # noqa: E402

NAME = "round21_fp4_quant_head"
H3 = "head.f=mx8,head.dw=nv4+sr"
H1 = "head.dw=nv4+sr"
H2 = "head.dw=mx8"


def register() -> None:
    p19.register(ROUND)
    r19run.ROUND = ROUND                                   # r19's own actions write to ``ROUND / profile``
    r16.SHORT["r21"] = NAME
    r16.DEFAULT_KW[NAME] = {"variant": H3}

    def stack(policy, **kw):
        return P2R14R15_STACK + (NAME,), {**P2R14R15_PATCH_KWARGS, NAME: {"variant": policy, **kw}}

    r16.PRESET_CFG.update({
        "P3H": stack(H3), "P3H_H1": stack(H1), "P3H_H2": stack(H2),
        "P3H_RN": stack("head.f=mx8,head.dw=nv4+rn"), "P3H_EX": stack(H3 + ",amax=exact"), "P3H_HR2": stack(H3, dy_headroom=2.0),
        "P3H_RNEX": stack("head.f=mx8,head.dw=nv4+rn,amax=exact"),
    })


register()

# ------------------------------------------------------------------ gate S with the FP4 saturation statistics of the candidate
_orig_curve_run = r16.curve_run


def summarize_stats(stats: dict) -> dict:
    """Group ``FP4State.stats()`` by site class: sites 4c+2j = X column copy (col slot), 4c+2j+1 = dZ (row copy for dgrad, col copy for wgrad), the last two = weights (row slot)."""
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


def curve_run_stats(cfg, batches, steps, scen, mode, args):
    from round21_fp4_quant_head.impl import patch as p21

    p21.STATES.clear()
    res = _orig_curve_run(cfg, batches, steps, scen, mode, args)
    if p21.STATES:
        policy, st = p21.STATES[-1]
        res["fp4_policy"] = policy
        res["fp4_stats"] = summarize_stats(st.stats())
        p21.STATES.clear()
    return res


r16.curve_run = curve_run_stats


# ------------------------------------------------------------------ graph_equiv with the FP4 state reset (eager runs: full reset; graph run: keep the exact init made by the warm-up)
def action_graph_equiv(args) -> None:
    from round16_full_graph.impl.graph_step import padded_targets

    cfg = r16.make_cfg(args.model, args.stack, seed=args.seed)
    results: dict = {"command": [sys.executable, *sys.argv], "scenarios": {}}
    with bench.gpu_lock():
        r16.wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        ctx.rng = RngTape(mode="off")
        batches = preload_batches(ctx, k=2)
        flat = ctx.meta["round15_optimizer_flat"]
        handle = ctx.meta.get(NAME)
        padded = [{"image": b["image"], "binary_masks": padded_targets(b, args.max_targets)} for b in batches]
        init = r16._snapshot(ctx, flat)
        for scen in args.scenarios:
            K = args.equiv_steps
            row: dict = {}

            def run_eager():
                r16._restore(flat, init)
                if handle is not None:
                    handle.reset_state(full=True)
                r16.set_scenario_state(ctx, scen)
                ls, gs = [], []
                for k in range(K):
                    torch.cuda.manual_seed(7000 + k)
                    o = train_step(ctx, padded[k % 2], k)
                    ls.append(o.loss.clone())
                    gs.append(o.grad_norm.clone())
                torch.cuda.synchronize()
                return [float(x) for x in ls], [float(x) for x in gs], flat.flat.master.cpu()

            e1 = run_eager()
            e2 = run_eager()
            r16.free_gpu()
            runner = r16.make_runner(ctx, batches, args)
            r16._restore(flat, init)
            if handle is not None:
                handle.reset_state(full=True)
            r16.set_scenario_state(ctx, scen)
            runner.prepare(batches[0])
            r16._restore(flat, init)
            if handle is not None:
                handle.reset_state(full=False)       # keep ``inited`` + the exact init amax of the warm-up; clear the current slot, restart the SR counter
            r16.set_scenario_state(ctx, scen)
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
            row["steps"] = K
            init_cpu = init["master"].cpu()
            row["eager_vs_eager"] = r16._delta_stats(e1[2], e2[2], init_cpu)
            row["graph_vs_eager"] = r16._delta_stats(gmaster, e1[2], init_cpu)
            row["loss_eager"], row["loss_eager2"], row["loss_graph"] = e1[0], e2[0], gl
            row["gnorm_eager"], row["gnorm_graph"] = e1[1], gg
            row["loss_rel_max_graph_vs_eager"] = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(gl, e1[0]))
            row["loss_rel_max_eager_vs_eager"] = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(e2[0], e1[0]))
            row["gnorm_rel_max_graph_vs_eager"] = max(abs(a - b) / max(abs(a), 1e-8) for a, b in zip(gg, e1[1]))
            row["loss_bitwise_first_step"] = bool(gl[0] == e1[0][0])
            row["graph"] = runner.report.as_dict()
            if handle is not None:
                st = handle.state.stats()
                row["fp4_stats"] = summarize_stats(st)
            runner.release()
            del runner
            r16.free_gpu()
            results["scenarios"][scen] = row
            print(scen, json.dumps({k: row[k] for k in ("steps", "eager_vs_eager", "graph_vs_eager", "loss_rel_max_graph_vs_eager", "loss_rel_max_eager_vs_eager",
                                                          "gnorm_rel_max_graph_vs_eager", "loss_bitwise_first_step")}), flush=True)
            print("  captures", row["graph"]["captures"], "capture_s", [round(x, 2) for x in row["graph"]["capture_s"]], "keys", row["graph"]["keys"], flush=True)
    out = r16.unique_path(ROUND / "profile" / f"graph_equiv_{args.model}_{r16.safe_name(args.stack)}.json")
    out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=tuple(B_AUTO), required=True)
    ap.add_argument("--stack", default="P3H")
    ap.add_argument("--name")
    ap.add_argument("--action", choices=("bench", "profile", "gate_s", "graph_equiv"), required=True)
    ap.add_argument("--runner", choices=("eager", "graph"), default="graph")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--ref", default="R0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--scenarios", nargs="*", default=["schedule", "compressed", "fixed_mid", "fixed_late"])
    ap.add_argument("--equiv-steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--preload", type=int, default=2)
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--max-resident", type=int, default=8)
    ap.add_argument("--metrics", choices=("step", "lazy", "off"), default="lazy")
    ap.add_argument("--no-side-stream", action="store_true")
    ap.add_argument("--cpu-batches", action="store_true")
    ap.add_argument("--no-segments", action="store_true")
    ap.add_argument("--no-results", action="store_true", default=True)
    args = ap.parse_args()
    if args.action != "bench" and args.phases == "all":
        args.phases = "mid"
    {"bench": r16.action_bench, "profile": r16.action_profile, "gate_s": r19run.action_gate_s, "graph_equiv": action_graph_equiv}[args.action](args)


if __name__ == "__main__":
    main()
