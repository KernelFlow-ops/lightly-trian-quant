"""Screen NVFP4 W4A4 MLP candidates against real fp32 semantic-video references.

All GPU work, including model loading and the 8-frame gate, uses the shared
infer-opt GPU lock. This script measures accuracy, not fake-quant performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "impl"))
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parent / "round7_head_postprocess"))

import sens  # noqa: E402
import nvfp4sim  # noqa: E402
import head as hd  # noqa: E402
from bench import gpu_lock, write_results, env_info  # noqa: E402


def load_units(path):
    return [tuple(u) for u in json.loads((HERE / path).read_text())["chosen_units"]]


def candidates(name):
    chosen = [u for u in load_units(f"units_{name}_n{6 if name == 'vitl16' else 14}.json")
              if u[1] in ("fc1", "fc2")]
    if name == "vitl16":
        return {
            "L5.fc1": [(5, "fc1")],
            "L20.fc2": [(20, "fc2")],
            "L23.fc2": [(23, "fc2")],
            "least_sensitive_3": chosen[:3],
            "row_recommended_6": chosen,
        }
    return {
        "L1.fc2": [(1, "fc2")],
        "L11.fc1": [(11, "fc1")],
        "L8.fc2": [(8, "fc2")],
        "least_sensitive_3": chosen[:3],
        "row_recommended_mlp_6": chosen[:6],
        "row_recommended_mlp_11": chosen,
    }


def zero_gemm_bound(name, units, n_layers, gemm):
    # Optimistic: every selected bf16 GEMM becomes free; quantization is ignored.
    return sum(gemm[f"{name}/{t}/M{2258 if i >= n_layers - 3 else 2058}"]["bf16_ms"]
               for i, t in units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("vitl16", "vitb16"), required=True)
    args = ap.parse_args()
    gemm = json.loads((HERE / "results.json").read_text())["gemm_microbench"]
    with gpu_lock():
        ev = sens.Evaluator(args.model)
        hd.enable(ev.model)
        baseline = ev.run()
        print("baseline", baseline, flush=True)
        rows = {}
        for tag, units in candidates(args.model).items():
            with nvfp4sim.patched(ev.model, units):
                accuracy = ev.run()
            rows[tag] = {
                "units": [list(u) for u in units], "accuracy": accuracy,
                "zero_gemm_upper_bound_ms": zero_gemm_bound(args.model, units, ev.L, gemm),
            }
            print(tag, "n", len(units), "gate", accuracy, "zero-GEMM bound ms",
                  round(rows[tag]["zero_gemm_upper_bound_ms"], 3), flush=True)
        write_results(HERE / "results.json", {
            "nvfp4_screen": {args.model: {"baseline": baseline, "candidates": rows}},
            "env_nvfp4": env_info(),
        })


if __name__ == "__main__":
    main()
