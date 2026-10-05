"""fp32 eager reference on all 126 frames of the test video (gate H), cached to disk.

  python ref126.py --model vitl16
Saves  $TMP/ref126_<model>.pt : {"head_bits": packed (head mask) per frame, "conf": [126] float64, "lr": fp32 [126,2,512,675] (cpu)}
using round7 fast prep (fp32) + fp32 eager forward_train + round7 post (same path as round6 greedy/heldout_real).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import r11_common as c
from lock2 import gpu_lock2

sys.path.insert(0, str(c.ROOT / "round7_head_postprocess"))
import pipeline as pl  # noqa: E402  (round7)


def ref_path(model):
    return c.TMP / f"ref126_{model}.pt"


def decode_all():
    import cv2
    cap = cv2.VideoCapture(c.mf.VIDEO_PATH)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    frames = decode_all()
    print("frames", len(frames), flush=True)
    with gpu_lock2(), torch.inference_mode():
        m32 = c.mf.load_model(a.model)
        pl.init(m32)
        heads, confs, lrs = [], [], []
        for i, f in enumerate(frames):
            b, meta = pl.fast_prep(f)
            ml, cl = pl.model_lowres(m32, b, "32-true")
            mask, conf = pl.fast_post(ml, cl, meta)
            st = pl._POST._state(meta)
            lrs.append(st["lr"].clone().cpu())
            heads.append(np.packbits(mask == 1))
            confs.append(conf)
        torch.save({"head_bits": np.stack(heads), "conf": np.array(confs, dtype=np.float64), "lr": torch.stack(lrs),
                    "shape": tuple(mask.shape)}, ref_path(a.model))
        print("saved", ref_path(a.model), "conf range", min(confs), max(confs), flush=True)


if __name__ == "__main__":
    main()
