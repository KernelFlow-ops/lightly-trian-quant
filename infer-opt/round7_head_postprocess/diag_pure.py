"""Which of prep / post causes the bf16-pure ViT-L accuracy difference? (2x2 swap)"""
import sys
from pathlib import Path
import torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import model_factory as mf, pipeline as pl

model = mf.load_model("vitl16", torch.bfloat16); pl.init(model)
res = {}
for prep in ("base", "fast"):
    for post in ("base", "fast"):
        acc = {}
        for i, fr in enumerate(mf.load_frames()):
            with torch.inference_mode():
                if prep == "base": batch, meta = mf.base_prep(model, fr)
                else: batch, meta = pl.fast_prep(fr)
                ml, cl = pl.model_lowres(model, batch, "32-true")
                if post == "base":
                    m512 = F.interpolate(ml, (512, 512), mode="bilinear")
                    logits = mf.base_post_logits(model, (m512, cl), meta).float()
                    mask, _, conf = mf.host_finalize(model, logits.cpu().numpy())
                else:
                    mask, conf, logits = pl.fast_post(ml, cl, meta, return_logits=True)
            acc[f"frame{i}"] = mf.compare_to_ref("vitl16", i, logits, mask, conf)
        g = mf.gate(acc); print(prep, post, {k: round(v, 4) if not isinstance(v, bool) else v for k, v in g.items()}, flush=True)
