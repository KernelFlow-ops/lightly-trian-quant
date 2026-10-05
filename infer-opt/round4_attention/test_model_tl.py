"""Whole-model correctness with the TileLang attention (eager and under CUDA-graph capture) vs SDPA path and fp32 eager."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "round1_torch_cleanup"; R2 = HERE.parent / "round2_cuda_graph"
for p in (HERE / "impl", R1, R2, HERE.parent / "common"):
    sys.path.insert(0, str(p))
import torch, model_factory as mf
from check import tensor_diff
from graphed import GraphedModel
from opt_model import OptEoMT

for name in ("vitl16", "vitb16"):
    m = mf.load_model(name)
    fr = mf.load_frames()[0]
    with torch.inference_mode():
        b, _ = mf.base_prep(m, fr)
        rm, rc = mf.base_model(m, b)
        base = OptEoMT(m, sdpa_backend="cudnn").forward_backend(b)
        for v in ("tl", "tl_rope"):
            o = OptEoMT(m, attn_impl=v)
            em, ec = o.forward_backend(b)
            g = GraphedModel(o.forward_backend, (2, 3, 512, 512))
            gm, gc = g(b)
            print(name, v, "eager vs fp32: mask", {k: round(x, 5) for k, x in tensor_diff(em, rm).items() if k in ("max_abs", "cos")},
                  "cls", {k: round(x, 5) for k, x in tensor_diff(ec, rc).items() if k in ("max_abs", "cos")})
            print(name, v, "eager vs opt-sdpa: mask", {k: round(x, 5) for k, x in tensor_diff(em, base[0]).items() if k in ("max_abs", "cos")},
                  "| graph vs eager identical:", bool(torch.equal(gm, em)) and bool(torch.equal(gc, ec)))
    print(name, "opt-sdpa vs fp32: mask", {k: round(x, 5) for k, x in tensor_diff(base[0], rm).items() if k in ("max_abs", "cos")})
