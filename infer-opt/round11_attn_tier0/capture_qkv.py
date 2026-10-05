"""Capture real r5 qkv_phys [3,2,N,H,64] bf16 (RoPE applied) of selected layers on real frames -> qkv_real_<model>.pt  (input of the attention matrix).

  python capture_qkv.py --model vitl16 --frames 0,3,7
"""
import argparse
import torch
import r11_common as c
import r11_model as rm
from lock2 import gpu_lock2

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--frames", default="0,7")
a = ap.parse_args()
frames = c.mf.load_frames()


class Rec:
    def __init__(self, layers):
        self.layers, self.i, self.store = set(layers), 0, {}

    def __call__(self, qkv, B, N, d, layer=None):
        if self.i in self.layers:
            self.store[self.i] = qkv.clone()
        self.i += 1
        return rm.attn_flash(qkv, B, N, d)


with gpu_lock2(), torch.inference_mode():
    model = c.mf.load_model(a.model)
    nb = len(model.backbone.blocks)
    layers = sorted({0, nb // 2, nb - 4, nb - 3, nb - 1})
    rec = Rec(layers)
    on, flags = rm.parse("r11")
    flags["attn"] = rec
    fn = rm.make_model_fn_r11(a.model, model, on, flags)
    fn_pipe = c.make_pipeline(model, rm.make_model_fn_r11(a.model, model, on, dict(flags, attn="flash")), rm.Pipeline11, slots=1)
    out = {}
    for fi in [int(x) for x in a.frames.split(",")]:
        fn_pipe.pinned_np[0][...] = frames[fi]
        fn_pipe.frame_gpu[0].copy_(fn_pipe.pinned[0])
        fn_pipe._prep_launch(0)
        rec.i, rec.store = 0, {}
        fn(fn_pipe.batch)
        for l, t in rec.store.items():
            out[f"L{l}_f{fi}"] = t.cpu()
    torch.save(out, c.TMP / f"qkv_real_{a.model}.pt")
    print("saved", list(out.keys()), {k: tuple(v.shape) for k, v in list(out.items())[:2]})
