"""R21a follow-up: cost of the `headroom` argument and of the roll() saturation monitor.   -> profile/bench_headroom.json
 (1) quant kernels: default call vs headroom=8.0 vs headroom=2.0 explicit, same process, interleaved CUDA-graph replays, cold inputs (rotating buffers), median of 3 rounds.  (The quantizer
     kernels are unchanged: headroom only changes a kernel scalar argument -- this is the noise floor check.)
 (2) roll: old `amax_roll` (prev <- next, next <- 0, seed += 1) vs new `amax_roll_mon` (+ saturation monitor) for n_sites in (64, 300, 1000), graph replay, us per launch.
 (3) after the change vs the pre-change full benchmark (profile/bench_fp4q.json) is done by tests/compare_bench_headroom.py (needs `bench_fp4q.py --tag headroom` first).
"""
import json
import sys
import time
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parent))
sys.path.insert(0, str(ROUND / "impl"))
sys.path.insert(0, str(ROUND / "tests"))
import torch

import fp4q
from bench_fp4q import capture, interleave
from common.bench import assert_gpu_exclusive, gpu_lock

dev = "cuda"
L2 = 96 * 2 ** 20
SHAPES = [("X[7203,1024]", 7203, 1024), ("dY[7203,4096]", 7203, 4096), ("dZ[28672,4096]", 28672, 4096), ("ViT-B X[14406,768]", 14406, 768), ("ViT-B dZ[57344,3072]", 57344, 3072)]


def main():
    out = {"round": "round21_fp4_quant_head", "quant": {}, "roll": {}}
    with gpu_lock():
        for _ in range(30):
            try:
                assert_gpu_exclusive()
                break
            except RuntimeError as e:
                print("retry:", str(e)[:100], flush=True)
                time.sleep(20)
        state = fp4q.FP4State(4)
        ext = fp4q.extension()
        for name, R, C in SHAPES:
            n_buf = max(2, int(-(-4.2 * L2 // (R * C * 2 * 1.4))))
            xs = [torch.randn(R, C, device=dev, dtype=torch.bfloat16)]
            xs += [xs[0].clone() for _ in range(n_buf - 1)]
            st = state
            seed = st.seed
            cands = {}
            for tag, kw in (("default", {}), ("hr8", {"headroom": 8.0}), ("hr2_explicit", {"headroom": 2.0})):
                cands[f"dual/{tag}"] = capture(lambda x, kw=kw: fp4q.quant_act(x, "nv4", amax="delayed", state=st, site=0, **kw), xs)
                cands[f"dual_sr/{tag}"] = capture(lambda x, kw=kw: fp4q.quant_act(x, "nv4", sr_row=True, sr_col=True, seed=seed, sr_offset=1, amax="delayed", state=st, site=0, **kw), xs)
                cands[f"dual_rhtcol_sr/{tag}"] = capture(lambda x, kw=kw: fp4q.quant_act(x, "nv4", rht_col=True, sr_row=True, sr_col=True, seed=seed, sr_offset=1, amax="delayed", state=st, site=0, **kw), xs)
            us = interleave(cands)
            rec = {}
            for mode in ("dual", "dual_sr", "dual_rhtcol_sr"):
                d, h8, h2 = us[f"{mode}/default"], us[f"{mode}/hr8"], us[f"{mode}/hr2_explicit"]
                rec[mode] = {"default_us": round(d, 2), "hr8_us": round(h8, 2), "hr2_explicit_us": round(h2, 2), "hr8_vs_default": round(h8 / d, 4), "hr2_vs_default": round(h2 / d, 4)}
            out["quant"][name] = rec
            print(f"[quant] {name} " + " ".join(f"{m}: {r['default_us']}/{r['hr8_us']} ({r['hr8_vs_default']}x)" for m, r in rec.items()), flush=True)
            del xs, cands
            torch.cuda.empty_cache()
        for n_sites in (64, 300, 1000):
            s = fp4q.FP4State(n_sites)
            g_old = capture(lambda _: ext.amax_roll(s.amax, s.seed, s.step) or s.seed, [None], inner_reps=20)
            g_new = capture(lambda _: s.roll() or s.seed, [None], inner_reps=20)
            us = interleave({"amax_roll(old)": g_old, "amax_roll_mon(new)": g_new}, reps=50)
            out["roll"][str(n_sites)] = {k: round(v, 3) for k, v in us.items()}
            print(f"[roll] sites={n_sites}", {k: round(v, 3) for k, v in us.items()}, flush=True)
        (ROUND / "profile" / "bench_headroom.json").write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
