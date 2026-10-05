"""Direct numerical comparison of the r11 model_fn variants vs r5_v1 on real frames (mask logits / class logits), before any gate.

  python tests/test_r11_model.py --model vitl16 --variants r11,r11+patch,r11+head,r11+tail --frames 0,7
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import r11_common as c
from lock2 import gpu_lock2

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vitl16")
ap.add_argument("--variants", default="r11")
ap.add_argument("--frames", default="0,7")
a = ap.parse_args()
frames = c.mf.load_frames()
variants = a.variants.split(",")
with gpu_lock2(), torch.inference_mode():
    base_model = c.mf.load_model(a.model)
    base_fn, pcls = c.make_variant(a.model, base_model, "r5_v1")
    pipe = c.make_pipeline(base_model, base_fn, pcls, slots=1)
    outs = {}
    for fi in [int(x) for x in a.frames.split(",")]:
        pipe.pinned_np[0][...] = frames[fi]
        pipe.frame_gpu[0].copy_(pipe.pinned[0])
        pipe._prep_launch(0)
        outs[fi] = [t.clone() for t in base_fn(pipe.batch)]
        print("r5_v1 ml absmean", outs[fi][0].float().abs().mean().item(), flush=True)
    for v in variants:
        m = c.mf.load_model(a.model)
        fn, _ = c.make_variant(a.model, m, v)
        for fi, ref in outs.items():
            pipe.pinned_np[0][...] = frames[fi]
            pipe.frame_gpu[0].copy_(pipe.pinned[0])
            pipe._prep_launch(0)
            ml, cl = fn(pipe.batch)
            d = (ml.float() - ref[0].float()).abs()
            dc = (cl.float() - ref[1].float()).abs()
            cos = torch.nn.functional.cosine_similarity(ml.double().flatten(), ref[0].double().flatten(), dim=0).item()
            print(f"{v:28s} frame {fi}: ml max_abs {d.max().item():.3g} mean_abs {d.mean().item():.3g} cos {cos:.8f} | cl max_abs {dc.max().item():.3g} mean_abs {dc.mean().item():.3g}  identical_ml {torch.equal(ml, ref[0])}", flush=True)
