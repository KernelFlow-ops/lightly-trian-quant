"""R22 graph_equiv: StepGraphRunner replay vs eager on identical RNG, for a stack with a delayed-amax FP4 state.  Adapted from round16_full_graph/run.py::action_graph_equiv.

The FP4 state (amax table, SR seed counter, step counter) is part of the training state, so it is snapshotted AFTER the runner's eager warm-up (which performs the exact
init of every delayed-amax site) and restored in place (the captured graph holds the tensor addresses) before every eager / graph run.  Each scenario then runs K steps
twice eagerly (eager vs eager: the run-to-run noise floor: round13's atomic dx accumulation) and K steps through graph replay; the parameter update (flat master) of the graph
run is compared with the first eager run.  Also checks that the state counter advanced by exactly K per run (one roll per step).

    python tests/graph_equiv22.py --model vitb16 --stack P3B_B --equiv-steps 8 --out profile/graph_equiv_vitb16_P3B_B.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND))
sys.path.insert(0, str(ROUND.parent))

import torch  # noqa: E402

import run as r22run  # noqa: E402,F401  (registers the presets, QUACK cache dir, redirects round16's outputs)
from common import bench  # noqa: E402
from common.harness import apply_patches, build, preload_batches, train_step  # noqa: E402
from common.rng import RngTape  # noqa: E402
from round16_full_graph import run as r16  # noqa: E402
from round16_full_graph.impl.graph_step import padded_targets  # noqa: E402

R22 = "round22_fp4_backbone"


def snap_fp4(h) -> dict:
    s = h.state
    return {"amax": s.amax.clone(), "seed": s.seed.clone(), "step": s.step.clone(), "inited": list(s.inited), "headroom": s.headroom.clone(),
            "mon": [getattr(s, n).clone() for n in ("mon_cnt", "mon_maxr") if hasattr(s, n)]}


def restore_fp4(h, snap: dict) -> None:
    s = h.state
    s.amax.copy_(snap["amax"])
    s.seed.copy_(snap["seed"])
    s.step.copy_(snap["step"])
    s.headroom.copy_(snap["headroom"])
    for n, v in zip(("mon_cnt", "mon_maxr"), snap["mon"]):
        getattr(s, n).copy_(v)
    s.inited[:] = snap["inited"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--equiv-steps", type=int, default=8)
    ap.add_argument("--scenarios", nargs="*", default=["schedule", "compressed", "fixed_mid", "fixed_late"])
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--max-resident", type=int, default=8)
    ap.add_argument("--metrics", default="lazy")
    ap.add_argument("--no-side-stream", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = r16.make_cfg(args.model, args.stack, seed=args.seed)
    results: dict = {"command": [sys.executable, *sys.argv], "stack": args.stack, "scenarios": {}}
    with bench.gpu_lock():
        r16.wait_exclusive()
        ctx = build(cfg)
        apply_patches(ctx)
        ctx.rng = RngTape(mode="off")
        batches = preload_batches(ctx, k=2)
        flat = ctx.meta["round15_optimizer_flat"]
        h = ctx.meta[R22]
        padded = [{"image": b["image"], "binary_masks": padded_targets(b, args.max_targets)} for b in batches]
        init = r16._snapshot(ctx, flat)
        init_cpu = init["master"].cpu()
        K = args.equiv_steps
        for scen in args.scenarios:
            row: dict = {}
            # (a) warm every signature of the scenario eagerly (exact init of the delayed-amax sites), then snapshot the FP4 state
            r16._restore(flat, init)
            h.state.reset()
            r16.set_scenario_state(ctx, scen)
            runner = r16.make_runner(ctx, batches, args)
            runner.prepare(batches[0])
            assert all(h.state.inited[i] for i in h.active_sites()), "warm-up did not initialise every delayed-amax site"
            row["active_sites"] = len(h.active_sites())
            fp = snap_fp4(h)

            def run_eager():
                r16._restore(flat, init)
                restore_fp4(h, fp)
                r16.set_scenario_state(ctx, scen)
                ls, gs = [], []
                for k in range(K):
                    torch.cuda.manual_seed(7000 + k)
                    o = train_step(ctx, padded[k % 2], k)
                    ls.append(o.loss.clone())
                    gs.append(o.grad_norm.clone())
                torch.cuda.synchronize()
                return [float(x) for x in ls], [float(x) for x in gs], flat.flat.master.cpu(), int(h.state.step.item()) - int(fp["step"].item())

            e1 = run_eager()
            e2 = run_eager()
            r16.free_gpu()
            r16._restore(flat, init)
            restore_fp4(h, fp)
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
            state_adv_graph = int(h.state.step.item()) - int(fp["step"].item())
            row["steps"] = K
            row["fp4_state_advance"] = {"eager1": e1[3], "eager2": e2[3], "graph": state_adv_graph, "expected": K, "ok": bool(e1[3] == e2[3] == state_adv_graph == K)}
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
            args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
