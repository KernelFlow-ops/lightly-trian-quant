"""Model / frame loading and the baseline (PyTorch eager) pipeline.

The baseline mirrors `scripts/infer_dinov3_eomt_head_video.py::predict_head`
exactly, split into stages so each stage can be timed separately:

    prep  : BGR uint8 frame -> cvtColor -> PIL -> preprocess_image -> preprocess_batch
    model : forward_backend(batch)  -> (mask_logits (B,Q,512,512), class_logits (B,Q,K+1))
    post  : per-pixel logits -> untile -> drop void class -> resize to frame -> host argmax

The checkpoints are DINOv3 EoMT *semantic* segmentation models
(classes {0: background, 1: head}, 100 queries, 3 joint blocks, 512x512 crops).
Test frames come from the real video /root/autodl-tmp/1.mp4 (1900x1440): resized to
short side 512 -> 675x512 -> 2 overlapping crops (starts 0 and 163), batch=2.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("LIGHTLY_TRAIN_EVENTS_DISABLED", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

WEIGHTS = {
    "vitl16": "/root/autodl-tmp/vitl-best.pth",
    "vitb16": "/root/autodl-tmp/vitb-best.pth",
}
# Real test video: 1900x1440 @60fps, 126 frames. Short side 1440 -> 512 gives
# 675x512, tiled into 2 crops (starts 0 and 163) -> batch=2.
VIDEO_PATH = "/root/autodl-tmp/1.mp4"
NUM_EVAL_FRAMES = 8
FRAME_DIR = Path("/root/autodl-tmp/infer-opt-tmp/frames")
REF_DIR = Path("/root/autodl-tmp/infer-opt-tmp/ref")
HEAD_CLASS_ID = 1
# Accuracy gate agreed with the user (+-5% vs fp32 eager reference).
GATE_HEAD_IOU = 0.95  # pooled over all eval frames
GATE_CONF_REL = 0.05  # per frame |conf - ref| / |ref|


def load_model(model: str, dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    import lightly_train

    m = lightly_train.load_model(WEIGHTS[model], device="cuda")
    m.eval()
    if dtype != torch.float32:
        m.to(dtype)
    return m


def extract_video_frames(n: int = NUM_EVAL_FRAMES) -> list[Path]:
    """Decode `n` evenly spaced frames of the test video to lossless PNGs (once)."""
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    paths, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            p = FRAME_DIR / f"video1_f{idx:03d}.png"
            cv2.imwrite(str(p), frame)
            paths.append(p)
        idx += 1
    cap.release()
    assert len(paths) == n, (len(paths), wanted)
    return paths


def load_frames() -> list[np.ndarray]:
    """Evenly spaced frames of the real test video, BGR uint8 as cv2 decodes them.

    Frame 0 is used for timing; all frames are used for accuracy.
    """
    paths = sorted(FRAME_DIR.glob("video1_f*.png"))
    if len(paths) != NUM_EVAL_FRAMES:
        paths = extract_video_frames()
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    assert all(f is not None for f in frames)
    return frames


def autocast_ctx(precision: str):
    if precision == "bf16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# ---------------------------------------------------------------- baseline stages
def base_prep(model, frame_bgr: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    x, meta = model.preprocess_image(image)
    batch = model.preprocess_batch([x])
    return batch, meta


def base_model(model, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return model.forward_backend(batch)


def base_post_logits(model, raw, meta) -> torch.Tensor:
    """GPU part of post: returns (K, H, W) logits at frame resolution."""
    mask_logits, class_logits = raw
    crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
    logits = model.untile(
        crop_logits=crop_logits,
        origins=meta["origins"],
        image_sizes=[(meta["resized_h"], meta["resized_w"])],
    )[0]
    logits = logits[:-1].unsqueeze(0)
    return F.interpolate(logits, size=(meta["orig_h"], meta["orig_w"]), mode="bilinear")[0]


def base_post(model, raw, meta) -> tuple[np.ndarray, np.ndarray, float]:
    """Full post exactly as the video script: host copy + numpy argmax + confidence."""
    scores = base_post_logits(model, raw, meta).float().cpu().numpy()
    return host_finalize(model, scores)


def host_finalize(model, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    internal_mask = scores.argmax(axis=0).astype(np.int64)
    class_ids = model.internal_class_to_class.cpu().numpy()
    mask = class_ids[internal_mask].astype(np.uint8)
    head_index = int(np.nonzero(class_ids == HEAD_CLASS_ID)[0][0])
    head_score = scores[head_index]
    head_pixels = mask == HEAD_CLASS_ID
    if np.any(head_pixels):
        confidence = float(head_score[head_pixels].mean())
    else:
        confidence = float(head_score.max())
    return mask, head_score, confidence


def base_e2e(model, frame_bgr: np.ndarray, precision: str):
    with torch.inference_mode(), autocast_ctx(precision):
        batch, meta = base_prep(model, frame_bgr)
        raw = base_model(model, batch)
        return base_post(model, raw, meta)


# ---------------------------------------------------------------- reference I/O
def ref_path(model: str, idx: int) -> Path:
    return REF_DIR / f"{model}_frame{idx}.pt"


def make_reference(model_name: str) -> None:
    """fp32 eager reference outputs for every frame (final logits + raw of frame 0)."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(model_name, torch.float32)
    for i, frame in enumerate(load_frames()):
        with torch.inference_mode():
            batch, meta = base_prep(model, frame)
            raw = base_model(model, batch)
            logits = base_post_logits(model, raw, meta).float()
            mask, _, conf = host_finalize(model, logits.cpu().numpy())
        payload = {
            "logits": logits.cpu(),  # (K, H, W) fp32
            "mask": torch.from_numpy(mask),
            "confidence": conf,
            "batch_shape": tuple(batch.shape),
            "class_logits": raw[1].float().cpu(),
        }
        if i == 0:
            payload["mask_logits_512_fp16"] = raw[0].half().cpu()
        torch.save(payload, ref_path(model_name, i))
    del model
    torch.cuda.empty_cache()


def compare_to_ref(model_name: str, idx: int, logits: torch.Tensor, mask: np.ndarray,
                   confidence: float) -> dict[str, float]:
    ref = torch.load(ref_path(model_name, idx))
    # float64: fp32 cosine over ~4M elements has ~1e-3 accumulation error.
    rl = ref["logits"].double()
    ll = logits.double().cpu()
    rm = ref["mask"].numpy() == HEAD_CLASS_ID
    pm = mask == HEAD_CLASS_ID
    inter = int(np.logical_and(rm, pm).sum())
    union = int(np.logical_or(rm, pm).sum())
    conf_abs = abs(confidence - ref["confidence"])
    return {
        "logits_max_abs": (ll - rl).abs().max().item(),
        "logits_cos": torch.nn.functional.cosine_similarity(
            ll.flatten(), rl.flatten(), dim=0).item(),
        "head_iou": inter / union if union else 1.0,
        "head_inter": inter,
        "head_union": union,
        "pixel_agree": float((mask == ref["mask"].numpy()).mean()),
        "confidence_abs_diff": conf_abs,
        "confidence_rel_diff": conf_abs / max(abs(ref["confidence"]), 1e-6),
        "ref_head_pixels": int(rm.sum()),
    }


def gate(acc_by_frame: dict[str, dict]) -> dict[str, float | bool]:
    """User's accuracy gate (+-5%): pooled head IoU >= 0.95, per-frame conf rel diff <= 5%."""
    frames = [v for v in acc_by_frame.values() if isinstance(v, dict) and "head_union" in v]
    inter = sum(v["head_inter"] for v in frames)
    union = sum(v["head_union"] for v in frames)
    pooled = inter / union if union else 1.0
    conf_rel = max((v["confidence_rel_diff"] for v in frames), default=0.0)
    return {
        "pooled_head_iou": pooled,
        "min_frame_head_iou": min((v["head_iou"] for v in frames), default=1.0),
        "max_conf_rel_diff": conf_rel,
        "pass": pooled >= GATE_HEAD_IOU and conf_rel <= GATE_CONF_REL,
    }
