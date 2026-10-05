"""One-frame portrait-video check for the Round-9 whole-frame CUDA graph.

Usage: python check_tall_graph.py --model vitb16 --video /path/to/portrait.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "impl"))
sys.path.insert(0, str(HERE.parent / "common"))
import model_factory as mf  # noqa: E402
from bench import gpu_lock  # noqa: E402
import model_impls as mi  # noqa: E402
import video_infer as vi  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(mf.WEIGHTS), required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--impl", choices=list(mi.IMPLS), default="lightly")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Video contains no decodable frames: {args.video}")
    h, w = frame.shape[:2]
    if h <= w:
        ap.error(f"Expected a portrait video, got {w}x{h}")

    def collect(pipe, graph):
        pipe.use_graph = graph
        outputs = []

        def consumer(idx, mask, conf, frame_bgr, slot, release):
            outputs.append((mask.copy(), conf))
            return False

        stats = pipe.run([str(args.video)], consumer)
        if stats["frames"] != 1:
            raise ValueError(f"Expected one frame, got {stats['frames']}")
        return outputs[0]

    with gpu_lock(), torch.inference_mode():
        model = mf.load_model(args.model)
        orig = vi.load_orig_script()
        ref_mask, _, ref_conf = orig.predict_head(model, frame, torch.device("cuda"), "bf16-mixed")
        fn = mi.make_model_fn(args.model, model, args.impl)
        pipe = vi.VideoPipeline(model, fn, h, w, slots=2, inflight=1)
        graph_mask, graph_conf = collect(pipe, True)
        eager_mask, eager_conf = collect(pipe, False)
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            pipe.use_graph = True
            output = vi.infer_video_fast(pipe, orig, args.video, args.out_dir, 0.45)
        else:
            output = None

    ref_head, graph_head = ref_mask == 1, graph_mask == 1
    union = int(np.logical_or(ref_head, graph_head).sum())
    inter = int(np.logical_and(ref_head, graph_head).sum())
    result = {
        "impl": args.impl,
        "shape": [h, w],
        "graph_vs_eager_mask_identical": bool(np.array_equal(graph_mask, eager_mask)),
        "graph_vs_eager_conf_abs": abs(graph_conf - eager_conf),
        "vs_original_pixel_agree": float((graph_mask == ref_mask).mean()),
        "original_head_pixels": int(ref_head.sum()),
        "graph_head_pixels": int(graph_head.sum()),
        "head_inter": inter,
        "head_union": union,
        "vs_original_head_iou": inter / union if union else 1.0,
        "vs_original_conf_rel": abs(graph_conf - ref_conf) / max(abs(ref_conf), 1e-6),
        "output": output,
    }
    print(json.dumps(result, indent=2))
    if not result["graph_vs_eager_mask_identical"] or result["graph_vs_eager_conf_abs"] != 0:
        raise AssertionError("CUDA graph changed the portrait-frame output")
    if result["vs_original_head_iou"] < mf.GATE_HEAD_IOU or result["vs_original_conf_rel"] > mf.GATE_CONF_REL:
        raise AssertionError("Portrait-frame output missed the accuracy gate")


if __name__ == "__main__":
    main()
