"""nsys target: build a variant's whole-frame graph (prep + model_fn + post) and replay it under cudaProfiler capture.

Run through the dual-lock shell wrapper (see profile11.sh).
  python prof_model.py --model vitl16 --variant r5_v1 --iters 30
"""
import argparse
import torch
import r11_common as c
from lock2 import gpu_lock2

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--variant", default="r5_v1")
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()
with gpu_lock2(), torch.inference_mode():
    model = c.mf.load_model(a.model)
    fn, pcls = c.make_variant(a.model, model, a.variant)
    pipe = c.make_pipeline(model, fn, pcls, slots=1)
    g = pipe.graphs[0]
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(a.iters):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
