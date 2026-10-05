"""nsys target: whole-model graph (model_fn only) of one variant, replayed under cudaProfiler capture.
Run under the dual-lock wrapper (tests/prof13.sh):   python tests/prof13.py --model vitl16 --variant r13_mx8a --iters 20
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "round12_lowp_stack"))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import speed12  # noqa: E402
import plug13  # noqa: E402
from lock2 import gpu_lock2  # noqa: E402

speed12.PLUGINS.append(plug13)
c = speed12.c
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--variant", required=True)
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()
with gpu_lock2(), torch.inference_mode():
    model = c.mf.load_model(a.model)
    fn, pcls = speed12.make_variant(a.model, model, a.variant)
    frame = c.mf.load_frames()[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xb, _ = c.mf.base_prep(model, frame)
    x = xb.to(torch.bfloat16).contiguous()
    g, out = speed12.capture(fn, x)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(a.iters):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
