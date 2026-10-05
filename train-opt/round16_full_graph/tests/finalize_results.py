"""Consolidate round15 / round16 results.json from the profile jsons and nsys exports on disk (no GPU).  Idempotent.
  python finalize_results.py
Fills: kernels_per_step / gpu_busy_ratio (from the raw kernel trace, tests/trace_busy.py) / syncs_per_step (nsys_analyze), graph capture info,
gate S / graph-vs-eager equivalence (round16), optimizer gates (round15)."""
import glob
import json
import sys
from pathlib import Path

ROOT = Path("/root/lightly-train/train-opt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "round16_full_graph" / "tests"))
from common import bench  # noqa: E402
from trace_busy import analyze  # noqa: E402

R15, R16 = ROOT / "round15_optimizer_flat", ROOT / "round16_full_graph"
NSYS = Path("/root/autodl-tmp/train-opt-tmp/round16_full_graph/nsys")
# results.json variant -> nsys capture prefix (ViT mid, 5 steps)
NSYS_OF = {
    "P2_eager_r7": "step_{m}_P2_eager_v1", "P2r15_eager": "step_{m}_P2_r15_eager_v1",
    "P2r15_graph": "step_{m}_P2_r15_graph_v1", "P2r15_graph_lazy": "step_{m}_P2_r15_graph_v2",
}


def nsys_fields(model: str, variant: str):
    tmpl = NSYS_OF.get(variant)
    if tmpl is None:
        return None
    prefix = NSYS / tmpl.format(m=model)
    if not Path(str(prefix) + "_cuda_gpu_trace.csv").exists():
        return None
    a = analyze(str(prefix))
    j = json.load(open(str(prefix) + "_analysis.json"))
    return {"kernels_per_step": a["kernels"], "gpu_busy_ratio": a["busy_ratio"], "syncs_per_step": j.get("syncs_per_step"),
            "nsys": {"window_ms_per_step": a["span_ms_per_step"], "kernel_ms_per_step": a["busy_ms_per_step"], "gaps_gt20us_per_step": a["gaps_gt20us_per_step"],
                     "gap_ms_per_step": a["gap_ms_per_step"], "prefix": prefix.name, "note": "mid phase, 5-step window, from the raw kernel trace"}}


def main() -> None:
    r16 = json.load(open(R16 / "results.json"))
    r16_payload: dict = {}
    r15_payload: dict = {}
    for model in ("vitl16", "vitb16"):
        for variant, entry in r16.get(model, {}).items():
            add = nsys_fields(model, variant) or {}
            upd = {k: v for k, v in add.items()}
            # accuracy: gate S (graph vs eager) on the graph variants
            if variant in ("P2r15_graph_lazy", "P2r15_graph"):
                gs = glob.glob(str(R16 / "profile" / f"gateS_{model}_P2_r15_graph_vs_eager*.json"))
                if gs:
                    runs = json.load(open(gs[0]))["runs"]
                    upd["accuracy"] = {"level_c": {"curve300_max_rel": max(r["gate"]["curve300_max_rel"] for r in runs.values()),
                                                   "pass": all(r["gate"]["pass"] for r in runs.values()),
                                                   "scenarios": {k: {"max_rel": r["gate"]["curve300_max_rel"], "pass": r["gate"]["pass"], "captures": r["graph"]["graph"]["captures"]} for k, r in runs.items()},
                                                   "desc": "gate S: 50 steps on 2 real batches, graph vs eager (same stack P2+r15), gate_c(interval=10, max_rel=0.05)"}}
                eq = sorted(glob.glob(str(R16 / "profile" / f"graph_equiv_{model}_P2_r15*.json")))
                eqs = {}
                for f in eq:
                    for scen, row in json.load(open(f))["scenarios"].items():
                        eqs[scen] = {k: row[k] for k in ("steps", "loss_bitwise_first_step", "loss_rel_max_graph_vs_eager", "loss_rel_max_eager_vs_eager",
                                                         "gnorm_rel_max_graph_vs_eager", "graph_vs_eager", "eager_vs_eager")}
                if eqs:
                    upd["graph_equivalence"] = eqs
            if variant.startswith("P2r15_graph") and entry.get("graph"):
                # torch.cuda.max_memory_allocated() after the capture excludes the graph's private pool: use the reserved memory after capture
                upd["peak_mem_gb"] = max(entry["graph"]["reserved_gb_after_capture"])
                upd["peak_mem_note"] = "max torch reserved GB after capture (graph private pool included); max_memory_allocated during replay is not representative"
            if upd:
                r16_payload.setdefault(model, {})[variant] = upd
        # round15 results: the optimizer-only table is already there; add the step-level comparison and gates
        for variant in ("P2_eager_r7", "P2r15_eager"):
            if variant in r16.get(model, {}):
                e = json.loads(json.dumps(r16[model][variant]))
                e["round"] = "round15_optimizer_flat"
                e["desc"] = e.get("desc", "") + "  (measured with round16_full_graph/run.py, same file as round15/run.py)"
                e.update(nsys_fields(model, variant) or {})
                r15_payload.setdefault(model, {})[variant] = e
        gb = R15 / "profile" / f"gateB_{model}_P1_r15_vs_P1_mid.json"
        if gb.exists():
            res = json.load(open(gb))["results"]
            lv = {"loss_rel": max(r["loss_rel"] for r in res), "grad_cos_min": min(r["grad_cos_min"] for r in res),
                  "grad_cos_global_min": min(r["grad_cos_global"] for r in res), "grad_norm_rel": max(r["grad_norm_rel"] for r in res),
                  "update_cos_min": min(r["update_cos_min"] for r in res), "pass": all(r["pass"] for r in res), "seeds": [r["seed"] for r in res],
                  "note": "step 0: warmup lr = 0 in every group, so the update is exactly 0 in both stacks and update_cos is vacuous; see real_step_lr"}
            r15_payload.setdefault(model, {}).setdefault("P1_r15_flat", {})["accuracy"] = {"level_b": lv}
        rs = R15 / "profile" / f"real_step_lr_{model}.json"
        if rs.exists():
            r15_payload.setdefault(model, {}).setdefault("P1_r15_flat", {})["real_step_lr"] = json.load(open(rs))
        cr = R15 / "profile" / f"check_real_flat_{model}.json"
        if cr.exists():
            r15_payload.setdefault(model, {}).setdefault("P1_r15_flat", {})["real_model_check"] = json.load(open(cr))
    sg = R15 / "profile" / "synthetic_gate.json"
    if sg.exists():
        d = json.load(open(sg))
        r15_payload["synthetic_gate"] = {"pass": d["pass"], "worst_100_steps": d["trajectory_worst"], "deterministic_bitwise": d["deterministic_bitwise"],
                                         "ckpt_flat_to_torch_state_bitwise": d["ckpt_flat_to_torch_state_bitwise"], "ckpt_torch_to_flat_state_bitwise": d["ckpt_torch_to_flat_state_bitwise"]}
    fb = R16 / "profile" / "fallback_and_swap_vitb16.json"
    if fb.exists():
        r16_payload["fallback_and_swap_vitb16"] = json.load(open(fb))
    bench.write_results(R16 / "results.json", r16_payload)
    bench.write_results(R15 / "results.json", r15_payload)
    print("round16 variants:", {m: sorted(v) for m, v in json.load(open(R16 / "results.json")).items() if isinstance(v, dict)})
    print("round15 keys:", sorted(json.load(open(R15 / "results.json"))))


if __name__ == "__main__":
    main()
