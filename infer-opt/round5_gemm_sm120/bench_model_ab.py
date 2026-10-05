"""Interleaved whole-model CUDA-Graph A/B for Round 3 and tuned Round 5 v1.

Usage: python bench_model_ab.py vitl16|vitb16
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
from bench import write_results  # noqa: E402
from kt import ab  # noqa: E402
import model_factory as mf  # noqa: E402
from r5_model_impls import make_model_fn_r5  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("vitl16", "vitb16"))
    args = parser.parse_args()
    model = mf.load_model(args.model)
    f3 = make_model_fn_r5(args.model, model, "r3")
    f5 = make_model_fn_r5(args.model, model, "r5_v1")
    x = torch.randn(2, 3, 512, 512, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        fns = {"r3": lambda: f3(x), "r5_v1_tuned": lambda: f5(x)}
        first = ab(fns, n=1, rounds=9, reps=3)
        second = ab(dict(reversed(list(fns.items()))), n=1, rounds=9, reps=3)
    result = {name: {"first_ms": first[name]["p50"] / 1000,
                     "reverse_ms": second[name]["p50"] / 1000,
                     "paired_ms": (first[name]["p50"] + second[name]["p50"]) / 2000}
              for name in fns}
    print(args.model, result, flush=True)
    write_results(HERE / "results.json", {"model_graph_ab": {args.model: result}})


if __name__ == "__main__":
    main()
