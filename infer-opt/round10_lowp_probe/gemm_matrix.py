"""GEMM speed matrix runner (graph replay, interleaved candidates, median of 3 rounds).  Run under the dual GPU lock.

  python gemm_matrix.py --cands r5_v1,cublas_bf16,cublas_fp16,cublas_fp16acc --Ms 2058,2258 [--models vitl16] [--kinds qkv,fc1] [--out results_gemm.json]

Results are merged into round10_lowp_probe/results_gemm.json  { "<model>|<kind>|<M>": { "<cand>": {us, tflops, rounds_us, note} } }.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl"))

import torch  # noqa: E402

import bench10 as b  # noqa: E402

REGISTRY = ["cands_base", "cands_f16", "cands_i8", "cands_mx8", "cands_fp4", "cands_epi"]


def all_cands(names):
    out = []
    for m in REGISTRY:
        try:
            mod = importlib.import_module(m)
        except ModuleNotFoundError as ex:
            if ex.name == m:
                continue
            raise
        out += mod.cands(names)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--Ms", default="2058,2258")
    ap.add_argument("--models", default="vitl16,vitb16")
    ap.add_argument("--kinds", default="qkv,proj,fc1,fc2,patch,ct1,ct2,einsum")
    ap.add_argument("--extra", action="store_true", help="patch-embed / head ConvT / mask-einsum shapes instead of the backbone shapes")
    ap.add_argument("--out", default=str(HERE / "results_gemm.json"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=1e9, help="stop starting new shapes after this many seconds (rerun to resume: finished shapes are skipped)")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    names = a.cands.split(",")
    cands = all_cands(names)
    got = {c.name for c in cands}
    assert got == set(names), f"unknown candidates {set(names) - got}"
    shapes = [s for s in (b.extra_shapes() if a.extra else b.gemm_shapes([int(x) for x in a.Ms.split(",")]))
              if s["model"] in a.models.split(",") and s["kind"] in a.kinds.split(",")]
    out = Path(a.out)
    print("other gpu procs:", b.other_gpu_procs(), flush=True)
    import json
    done = json.loads(out.read_text()) if out.exists() else {}
    T0 = time.time()
    for s in shapes:
        t0 = time.time()
        if not a.redo and all(c.name in done.get(b.shape_key(s), {}) for c in cands):
            continue
        if t0 - T0 > a.time_limit:
            print("time limit reached; rerun to resume", flush=True)
            break
        res = b.time_shape(s, cands, rounds=a.rounds)
        for c in cands:
            if c.name in res:
                res[c.name]["note"] = c.note
        b.merge_results(out, {b.shape_key(s): res, "_meta": {"other_gpu_procs": b.other_gpu_procs(), "time": time.strftime("%F %T")}})
        line = "  ".join(f"{n}={v['us']:.1f}us/{v['tflops']:.0f}TF" if "us" in v else f"{n}=ERR" for n, v in res.items())
        print(f"{b.shape_key(s):22s} {line}   ({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
