"""Correctness of P1 (fused prep) and P2 (fused post) against the baseline stages.

  python test_prepost.py --model vitb16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "impl"))

import model_factory as mf  # noqa: E402
from check import tensor_diff  # noqa: E402
import postfuse as pf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitb16")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--tall", action="store_true", help="rotate frames to verify vertical tiling")
    args = ap.parse_args()
    model = mf.load_model(args.model, torch.float32)
    frames = mf.load_frames()[: args.frames]
    if args.tall:
        frames = [np.ascontiguousarray(np.rot90(frame)) for frame in frames]
    norm = model.image_normalize

    print("== aa weight table vs F.interpolate(antialias=True) on random input")
    h, w = frames[0].shape[:2]
    x = torch.rand(1, 1, h, w, device="cuda")
    rh, rw = pf.resized_size(h, w, 512)
    ref = F.interpolate(x, size=(rh, rw), mode="bilinear", antialias=True, align_corners=False)
    ym, ys, yw = (torch.from_numpy(a).cuda() for a in pf.aa_table(h, rh))
    xm, xs, xw = (torch.from_numpy(a).cuda() for a in pf.aa_table(w, rw))
    # apply tables in torch (gather) -> same math as the kernel
    def apply(x, mn, sz, wt, dim):
        K = wt.shape[1]
        idx = (mn[:, None] + torch.arange(K, device="cuda")[None]).clamp(max=x.shape[dim] - 1)
        g = x.index_select(dim, idx.flatten())
        shp = list(x.shape); shp[dim:dim + 1] = [idx.shape[0], K]
        g = g.reshape(shp)
        wshape = [1] * len(shp); wshape[dim] = idx.shape[0]; wshape[dim + 1] = K
        return (g * wt.reshape(wshape)).sum(dim + 1)
    got = apply(apply(x, ym, ys, yw, 2), xm, xs, xw, 3)
    table_diff = tensor_diff(got, ref)
    print("table-vs-aten", table_diff)
    assert table_diff["max_abs"] < 1e-4, table_diff

    for dtype in (torch.float32, torch.bfloat16):
        fp = pf.FastPrep(norm["mean"], norm["std"], 512, dtype)
        m2 = mf.load_model(args.model, dtype) if dtype != torch.float32 else model
        print(f"== P1 prep {dtype}")
        for i, fr in enumerate(frames):
            with torch.inference_mode():
                bb, bmeta = mf.base_prep(m2, fr)
                fb, fmeta = fp(fr)
            assert fmeta["origins"] == bmeta["origins"], (fmeta, bmeta)
            assert (fmeta["resized_h"], fmeta["resized_w"]) == (bmeta["resized_h"], bmeta["resized_w"])
            d = tensor_diff(fb, bb)
            print(f"frame{i}", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()})
            assert not d["nan"] and d["cos"] > 0.9999, d
            if dtype == torch.float32:
                assert d["max_abs"] < 1e-3, d
        if m2 is not model:
            del m2

    print("== P2 post: kernel vs baseline path on identical low-res inputs")
    post = pf.FastPost(model.internal_class_to_class.tolist(), 1, 512)
    for prec in ("bf16-mixed", "32-true"):
        print(f"-- input precision {prec}")
        for i, fr in enumerate(frames):
            with torch.inference_mode(), mf.autocast_ctx(prec):
                batch, meta = mf.base_prep(model, fr)
                ml, cl = model.forward_train(batch, return_logits_per_layer=False)
                ml, cl = ml[-1], cl[-1]
                m512 = F.interpolate(ml, (512, 512), mode="bilinear")
                logits_b = mf.base_post_logits(model, (m512, cl), meta)
            mask_b, _, conf_b = mf.host_finalize(model, logits_b.float().cpu().numpy())
            with torch.inference_mode():
                mask_k, conf_k, logits_k = post(ml, cl, meta, return_logits=True)
            d = tensor_diff(logits_k, logits_b.float())
            agree = float((mask_k == mask_b).mean())
            hb, hk = mask_b == 1, mask_k == 1
            iou = float((hb & hk).sum() / max((hb | hk).sum(), 1))
            print(f"frame{i} logits max_abs {d['max_abs']:.3e} mean_abs {d['mean_abs']:.3e} cos {d['cos']:.8f} "
                  f"mask_agree {agree:.6f} head_iou {iou:.6f} conf_base {conf_b:.6f} conf_k {conf_k:.6f} "
                  f"|d| {abs(conf_b-conf_k):.2e} head_px {int(hb.sum())}")
            assert agree > 0.9999 and abs(conf_b - conf_k) < 0.01, (agree, conf_b, conf_k)
            if prec == "32-true":
                assert d["max_abs"] < 1e-5 and iou == 1.0, (d, iou)


if __name__ == "__main__":
    main()
