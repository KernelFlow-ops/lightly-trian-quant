"""Round 14: nsys capture of one variant's model_fn CUDA graph (N replays inside the profiler range).
  prof14.sh <out_prefix> prof14.py <model> <variant> [nrep]      (dual GPU lock; see prof14.sh)"""
import sys
from pathlib import Path

R12 = Path("/root/lightly-train/infer-opt/round12_lowp_stack")
R14 = Path("/root/lightly-train/infer-opt/round14_fp4_full")
sys.path.insert(0, str(R12))
sys.path.insert(0, str(R14))
import torch
import speed12 as S          # sets up sys.path (round 11 / 12), registers r12; make_variant dispatches to PLUGINS
import plug14
S.PLUGINS.append(plug14)
c = S.c

model_name, variant = sys.argv[1], sys.argv[2]
nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 5
with torch.inference_mode():
    frame = c.mf.load_frames()[0]
    model = c.mf.load_model(model_name)
    fn, _ = S.make_variant(model_name, model, variant)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xb, _ = c.mf.base_prep(model, frame)
    x = xb.to(torch.bfloat16).contiguous()
    g, out = S.capture(fn, x)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(nrep):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
print("profiled", variant, "replays", nrep)
