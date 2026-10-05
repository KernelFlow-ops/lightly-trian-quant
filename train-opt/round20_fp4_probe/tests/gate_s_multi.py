"""R20b gate S (CONTRACTS 11.5): 50-step trajectories on 2 real batches, 4 scenarios (schedule, compressed, fixed_mid, fixed_late); R0, FINAL and the candidate
policies run in ONE process, all eager (the simulated FP4 quantizers are torch ops).  Candidates use ``amax=delayed`` (the policy default): the per-tensor amax is
2 x the amax of the previous step at the same call site, so unlike gate B this exercises the delayed state over 50 optimizer steps.

Pass: EMA(0.9) loss deviation <= 5% at the checkpoints (every 10 steps; the maximum over all 50 steps is recorded too) against R0 AND against FINAL, no NaN / Inf,
grad-norm ratios within [0.1, 10] (common.check.gate_c); grad-norm ratios are recorded.

  python tests/gate_s_multi.py --model vitl16 --cands B5 H3 --out profile/gateS_r20b_vitl16.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["QUACK_CACHE_DIR"] = "/root/autodl-tmp/train-opt-tmp/round20_fp4_probe/quack_cache_b"
os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "tests"))

import torch  # noqa: E402

import cands as CANDS  # noqa: E402
from common import bench  # noqa: E402
from common.check import _rel, gate_c  # noqa: E402
from common.harness import build, preload_batches  # noqa: E402
from gate_b_multi import r20_cfg  # noqa: E402
from round16_full_graph import run as r16  # noqa: E402
from round20_fp4_probe.impl import fp4_sim as FS  # noqa: E402


def max_rel_all(a, b):
    return max(_rel(x, y) for x, y in zip(a["loss_ema"], b["loss_ema"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["schedule", "compressed", "fixed_mid", "fixed_late"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    table = CANDS.all_screen_cands()
    cands = {}
    for c in args.cands:
        if c in table:
            cands[c] = table[c]
        else:
            n, _, p = c.partition("=")
            assert p, f"unknown candidate {c!r}"
            cands[n] = p
    ns = SimpleNamespace(max_targets=2, max_resident=8, metrics="lazy", no_side_stream=False)
    cfg_r = r16.make_cfg(args.model, "R0", seed=0)
    cfg_f = r16.make_cfg(args.model, "FINAL", seed=0)
    cfg_c = {n: r20_cfg(args.model, 0, "mid", p) for n, p in cands.items()}
    outp = Path(args.out)
    report = {"command": [sys.executable, *sys.argv], "model": args.model, "steps": args.steps, "cands": cands, "mode": "all eager, 2 batches, amax=delayed", "runs": {}}
    if outp.exists():
        report["runs"] = json.loads(outp.read_text()).get("runs", {})
    with bench.gpu_lock():
        r16.wait_exclusive()
        lctx = build(cfg_r)
        batches = preload_batches(lctx, k=2)
        del lctx
        r16.free_gpu()
        report.setdefault("batch_sum", float(sum(float(b["image"].double().sum()) for b in batches)))
        for scen in args.scenarios:
            run = report["runs"].setdefault(scen, {"cands": {}})
            t0 = time.time()
            if "R0" not in run:
                run["R0"] = r16.curve_run(cfg_r, batches, args.steps, scen, "eager", ns)
                run["FINAL"] = r16.curve_run(cfg_f, batches, args.steps, scen, "eager", ns)
                print(scen, "R0 / FINAL done", f"{time.time() - t0:.0f}s", "final-vs-R0 max_rel(all)", round(max_rel_all(run["R0"], run["FINAL"]), 4), flush=True)
            for name, cfg in cfg_c.items():
                if name in run["cands"]:
                    continue
                tc = time.time()
                cur = r16.curve_run(cfg, batches, args.steps, scen, "eager", ns)
                diag = FS.diag_snapshot()  # delayed-amax statistics of this candidate's 50 steps (empty for amax=exact / no nv4)
                g_r0, g_fn = gate_c(run["R0"], cur, interval=10, max_rel=0.05), gate_c(run["FINAL"], cur, interval=10, max_rel=0.05)
                row = {"policy": cands[name], "curve": cur, "vs_r0": g_r0, "vs_final": g_fn, "max_rel_all_vs_r0": max_rel_all(run["R0"], cur), "max_rel_all_vs_final": max_rel_all(run["FINAL"], cur),
                       "pass": bool(g_r0["pass"] and g_fn["pass"] and cur["finite"]), "seconds": time.time() - tc, "delayed_diag": diag}
                run["cands"][name] = row
                print(scen, name, json.dumps({"vsR0_ckpt": round(g_r0["curve300_max_rel"], 4), "vsFINAL_ckpt": round(g_fn["curve300_max_rel"], 4), "vsR0_all": round(row["max_rel_all_vs_r0"], 4),
                                              "vsFINAL_all": round(row["max_rel_all_vs_final"], 4), "finite": cur["finite"], "pass": row["pass"], "gn_ratio_r0": g_r0["grad_norm_ratios"],
                                              "loss_first_last": [cur["loss"][0], cur["loss"][-1]], "diag": {k: v for k, v in diag.items() if k != "worst_sites"}, "ref_first_last": [run["R0"]["loss"][0], run["R0"]["loss"][-1]], "s": round(row["seconds"])}), flush=True)
                outp.write_text(json.dumps(report, indent=1, allow_nan=False))
            outp.write_text(json.dumps(report, indent=1, allow_nan=False))
    report["summary"] = {n: {"policy": p, "all_scenarios_pass": all(report["runs"][s]["cands"].get(n, {}).get("pass", False) for s in args.scenarios),
                             "max_rel_ckpt_vs_r0": max(report["runs"][s]["cands"][n]["vs_r0"]["curve300_max_rel"] for s in args.scenarios if n in report["runs"][s]["cands"]),
                             "max_rel_ckpt_vs_final": max(report["runs"][s]["cands"][n]["vs_final"]["curve300_max_rel"] for s in args.scenarios if n in report["runs"][s]["cands"]),
                             "max_rel_all_vs_r0": max(report["runs"][s]["cands"][n]["max_rel_all_vs_r0"] for s in args.scenarios if n in report["runs"][s]["cands"]),
                             "max_rel_all_vs_final": max(report["runs"][s]["cands"][n]["max_rel_all_vs_final"] for s in args.scenarios if n in report["runs"][s]["cands"])}
                         for n, p in cands.items()}
    outp.write_text(json.dumps(report, indent=1, allow_nan=False))
    print("SUMMARY", json.dumps({n: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "policy"} for n, s in report["summary"].items()}), flush=True)
    print("wrote", outp)


if __name__ == "__main__":
    main()
