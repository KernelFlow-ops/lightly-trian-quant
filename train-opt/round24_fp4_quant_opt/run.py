"""round24_fp4_quant_opt entrypoint: round22's run.py (unchanged logic) with the FP4 quantizer swapped for round24's ``fp4q2`` for the presets whose name contains "q2".

ALIAS (must precede every import of round22 / round21 code): when a stack of the command line contains "q2", ``sys.modules["round21_fp4_quant_head.impl.fp4q"]`` and the
``impl`` package attribute are replaced by ``round24_fp4_quant_opt.impl.fp4q2``; round22's unmodified ``fp4_block22`` then does ``from round21_fp4_quant_head.impl import fp4q``
and gets fp4q2.  ``fp4q2.WEIGHT_ROW = False`` (round22 drops the weight row copy right after the call).  Stacks without "q2" are exactly round22's.

Presets added:  P3B_q2 (= P3B: ViT-L policy B4), P3B_B_q2 (= P3B_B: ViT-B policy B3), P3B_q2p7 / P3B_B_q2p7 (same with Philox4x32-7 SR noise),
P3B_q2d / P3B_B_q2d (fp4q2d superset module, sr_mode="dither"), P3B_q2dp7 / P3B_B_q2dp7 (dither + Philox-7).

Original round22 docstring follows.  Adapted from round19_final/run.py (same bench /
profile / graph_equiv / gate_s actions; round16's outputs are redirected to this round's directory by ``presets.register``).

Stacks (--stack, --ref): everything of round19 (R0, P1/TB1, FINAL, ...) plus
    P3B          FINAL + round22  ViT-L policy B4:  all.d=nv4+sr, all.w=nv4+sr, nofirst        (block 0 keeps FINAL's mx8 d / w)
    P3B_B        FINAL + round22  ViT-B policy B3:  all.d=nv4+sr, all.w=nv4+sr                 (no nofirst)
    P3B_noFirst  FINAL + round22  B3 + nofirst  (= P3B's policy; named model-agnostically for the ViT-B nofirst arm)
    P3B_exact    P3B / B3 policies with amax=exact (control)  ->  P3B_exactB
or any "name[:variant],..." spec, e.g. ``FINAL,round22_fp4_backbone:all.d=nv4+sr+all.w=nv4+sr+nofirst`` (variants use ``+``, never ``,`` or ``:``).

Actions (every GPU action holds the dual lock): bench, profile, graph_equiv (round16 actions), gate_s (this file: --stack under --runner vs --ref eager R0), gate_b is in tests/gate_b_real.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["QUACK_CACHE_DIR"] = "/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/quack_cache"  # seeded from r13 / r14 / r17 / r20 / r21 caches; must precede any quack import
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROUND = Path(__file__).resolve().parent


def install_q2_alias(philox_rounds: int = 10, dither: bool = False) -> None:
    import importlib

    if dither:  # superset module: fp4q2 + sr_mode="dither" (impl/fp4q2d.py, impl/fp4_kernels2d.cu)
        from round24_fp4_quant_opt.impl import fp4q2d as fp4q2

        fp4q2.SR_MODE = "dither"
    else:
        from round24_fp4_quant_opt.impl import fp4q2

    pkg = importlib.import_module("round21_fp4_quant_head.impl")
    sys.modules["round21_fp4_quant_head.impl.fp4q"] = fp4q2
    pkg.fp4q = fp4q2
    fp4q2.WEIGHT_ROW = False
    fp4q2.PHILOX_ROUNDS = philox_rounds


if any("q2" in a for a in sys.argv[1:]):
    install_q2_alias(7 if any(("q2p7" in a or "q2dp7" in a) for a in sys.argv[1:]) else 10, dither=any(("q2d" in a) for a in sys.argv[1:]))

import torch  # noqa: E402

from common import bench  # noqa: E402
from common.check import gate_c  # noqa: E402
from common.config import B_AUTO, BEST_STACK_V2, BEST_STACK_V2_KWARGS  # noqa: E402
from common.harness import build, preload_batches  # noqa: E402
from round19_final.impl import presets  # noqa: E402

presets.register(ROUND)
from round16_full_graph import run as r16  # noqa: E402

R22 = "round22_fp4_backbone"  # the patch stays round22's; the quantizer module behind it is swapped by install_q2_alias()
B3 = "all.d=nv4+sr+all.w=nv4+sr"
B4 = B3 + "+nofirst"


def register_presets() -> None:
    r16.SHORT["r22"] = R22

    def preset(variant: str):
        return BEST_STACK_V2 + (R22,), {**BEST_STACK_V2_KWARGS, R22: {"variant": variant}}

    r16.PRESET_CFG.update({
        "P3B": preset(B4), "P3B_B": preset(B3), "P3B_noFirst": preset(B4),
        "P3B_exact": preset(B4 + "+amax=exact"), "P3B_exactB": preset(B3 + "+amax=exact"),
        "P3B_rn": preset("all.d=nv4+rn+all.w=nv4+rn+nofirst"), "P3B_rnB": preset("all.d=nv4+rn+all.w=nv4+rn"),
        "P3B_q2": preset(B4), "P3B_B_q2": preset(B3), "P3B_q2p7": preset(B4), "P3B_B_q2p7": preset(B3),
        "P3B_q2d": preset(B4), "P3B_B_q2d": preset(B3), "P3B_q2dp7": preset(B4), "P3B_B_q2dp7": preset(B3),
    })


register_presets()


def curve_run22(cfg, batches, steps: int, scen: str, mode: str, args) -> dict:
    """round16 ``curve_run`` (verbatim logic) + the FP4 saturation summary of the round22 handle (read after the last step, outside the graph)."""
    import math

    from common.harness import apply_patches, train_step
    from common.rng import RngTape

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
    extra = {}
    if runner is not None:
        extra = {"graph": runner.report.as_dict()}
        runner.release()
    h = ctx.meta.get("round22_fp4_backbone")
    if h is not None:
        extra["fp4"] = {"policy": h.describe(), **h.stats()}
        h.ctx = None  # break the ctx <-> handle cycle
        h.weights = {}
        h.layers = {}
    del ctx, runner, h
    r16.free_gpu()
    extra["mem_gb_before_after"] = [round(mem0, 2), round(torch.cuda.memory_allocated() / 2**30, 2)]
    print(f"  [{mode} {cfg.patches[-1] if cfg.patches else 'R0'}] allocated GB before/after: {extra['mem_gb_before_after']}", flush=True)
    return {"steps": steps, "loss": losses, "loss_ema": ema, "grad_norm": norms, "finite": finite, **extra}


# ------------------------------------------------------------------ gate S: --stack (graph) vs --ref (eager, default R0)   (adapted from round19_final/run.py)
def action_gate_s(args) -> None:
    cfg_c = r16.make_cfg(args.model, args.stack, seed=0)
    cfg_r = r16.make_cfg(args.model, args.ref, seed=0)
    report = {"command": [sys.executable, *sys.argv], "ref": args.ref, "stack": args.stack, "runner": args.runner, "runs": {}}
    out = r16.unique_path(ROUND / "profile" / f"gateS_{args.model}_{r16.safe_name(args.stack)}_{args.runner}_vs_{r16.safe_name(args.ref)}.json")
    with bench.gpu_lock():
        r16.wait_exclusive()
        loader_ctx = build(cfg_c)
        batches = preload_batches(loader_ctx, k=2)
        del loader_ctx
        r16.free_gpu()
        for scen in args.scenarios:
            t0 = time.time()
            ref = curve_run22(cfg_r, batches, args.steps, scen, "eager", args)
            cand = curve_run22(cfg_c, batches, args.steps, scen, args.runner, args)
            gate = gate_c(ref, cand, interval=10, max_rel=0.05)
            report["runs"][scen] = {"gate": gate, "ref": ref, "cand": cand}
            print(scen, json.dumps({"max_rel": gate["curve300_max_rel"], "pass": gate["pass"], "finite": gate["finite"], "checkpoints": gate["checkpoints"],
                                    "grad_norm_ratios": gate["grad_norm_ratios"], "loss_first": [ref["loss"][0], cand["loss"][0]], "loss_last": [ref["loss"][-1], cand["loss"][-1]],
                                    "captures": cand.get("graph", {}).get("captures"), "seconds": round(time.time() - t0, 1)}), flush=True)
            if "fp4" in cand:
                f4 = cand["fp4"]
                print("  fp4", json.dumps({k: f4.get(k) for k in ("step", "total_calls", "total_exceed", "exceed_rate")}), json.dumps(f4.get("per_kind")), flush=True)
            out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=tuple(B_AUTO), required=True)
    ap.add_argument("--stack", default="FINAL")
    ap.add_argument("--name")
    ap.add_argument("--action", choices=("bench", "profile", "gate_s", "graph_equiv"), required=True)
    ap.add_argument("--runner", choices=("eager", "graph"), default="graph")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--ref", default="R0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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
    if args.action not in ("bench",) and args.phases == "all":
        args.phases = "mid"
    {"bench": r16.action_bench, "profile": r16.action_profile, "graph_equiv": r16.action_graph_equiv, "gate_s": action_gate_s}[args.action](args)


if __name__ == "__main__":
    sys.path.insert(0, str(ROUND))
    main()
