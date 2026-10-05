"""Smoke test: r12 model_fn vs r11 on real eval frames (logit agreement, finiteness), plus model_fn GPU time under graph replay.

  python tests/smoke12.py vitb16 [variant ...]        (default variants: r12+nopatch r12)
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
R11 = HERE.parent / "round11_attn_tier0"
for p in (R11, R11 / "impl", HERE / "impl", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402
import r11_common as c  # noqa: E402
import bench12  # noqa: F401,E402  (registers the r12 variants)
from lock2 import gpu_lock2  # noqa: E402

name = sys.argv[1]
variants = sys.argv[2:] or ["r12+nopatch", "r12"]


def graph_time(fn, x, reps=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(x)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps, out


with gpu_lock2(), torch.inference_mode():
    model = c.mf.load_model(name)
    frames = c.mf.load_frames()
    xs = []
    for f in (frames[0], frames[7]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            b, _ = c.mf.base_prep(model, f)
        xs.append(b.to(torch.bfloat16).contiguous())
    fns = {}
    for v in ["r5_v1", "r11"] + variants:
        t0 = time.time()
        fns[v] = c.make_variant(name, model, v)[0]
        fns[v](xs[0])
        torch.cuda.synchronize()
        print(f"built {v} in {time.time() - t0:.0f}s", flush=True)
    ref = [[t.float().clone() for t in fns["r5_v1"](x)] for x in xs]
    for v in ["r11"] + variants:
        for i, x in enumerate(xs):
            ml, cl = [t.float() for t in fns[v](x)]
            d = (ml - ref[i][0]).abs()
            cos = torch.nn.functional.cosine_similarity(ml.flatten().double(), ref[i][0].flatten().double(), dim=0).item()
            print(f"{v:14s} frame{(0, 7)[i]}: finite {bool(torch.isfinite(ml).all() and torch.isfinite(cl).all())} "
                  f"ml max_abs {d.max().item():.3f} mean_abs {d.mean().item():.4f} cos {cos:.8f} "
                  f"cl max_abs {(cl - ref[i][1]).abs().max().item():.4f}", flush=True)
    for v in ["r5_v1", "r11"] + variants:
        ms, _ = graph_time(fns[v], xs[0])
        print(f"{v:14s} model_fn graph replay {ms:.3f} ms", flush=True)
