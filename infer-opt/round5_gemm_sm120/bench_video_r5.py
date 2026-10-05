"""Whole-video FPS in the Round-9 bench convention (bench9.run_variant: graph pipeline, 126-frame video looped 5x, inflight=2, accuracy vs the
original script and vs fp32 gate), with round5 model_fns.  round9 files are only imported (results go to round5's results.json).
usage: python bench_video_r5.py --model vitl16 --variants r3,r5_v1,r5_v2"""
import argparse, sys, traceback
from pathlib import Path
R9 = Path("/root/lightly-train/infer-opt/round9_integration")
HERE = Path("/root/lightly-train/infer-opt/round5_gemm_sm120")
sys.path.insert(0, str(R9)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "impl")); sys.path.insert(0, "/root/lightly-train/infer-opt/common")
import bench9  # noqa: E402  (round9's bench: imports video_infer + round8 helpers)
import model_factory as mf  # noqa: E402
import torch  # noqa: E402
from bench import gpu_lock  # noqa: E402
import r5_model_impls as r5m  # noqa: E402

bench9.OUT = HERE / "results_video.json"
NAMES = {"r3": "round3 fused_block + graph + fused-prepost + fused-head (round9 'fused')",
         "r5_v1": "round5 v1 (tuned tiles, LN kernels) + graph + fused-prepost + fused-head",
         "r5_v2": "round5 v2 (LN-fused GEMMs, stream-K, PDL) + graph + fused-prepost + fused-head",
         "r5_v2_nopdl": "round5 v2 without PDL + graph + fused-prepost + fused-head"}
bench9.NAMES.update(NAMES)


def build(name, variant):
    model = mf.load_model(name)
    return model, r5m.make_model_fn_r5(name, model, variant)


bench9.build = build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(mf.WEIGHTS))
    ap.add_argument("--variants", default="r3,r5_v1,r5_v2")
    ap.add_argument("--loops", type=int, default=5)
    args = ap.parse_args()
    orig = bench9.vi.load_orig_script()
    m0 = mf.load_model(args.model)
    with gpu_lock():
        bench9.baseline_loop(orig, m0, 1, False)
        base_res, bmasks, bconfs = bench9.baseline_loop(orig, m0, 2, True)
    print("baseline", base_res, flush=True)
    del m0
    torch.cuda.empty_cache()
    for v in args.variants.split(","):
        try:
            bench9.run_variant(args.model, v, args.loops, bmasks, bconfs, base_res)
        except Exception:  # noqa: BLE001
            print(f"VARIANT {v} FAILED:\n{traceback.format_exc()}", flush=True)
