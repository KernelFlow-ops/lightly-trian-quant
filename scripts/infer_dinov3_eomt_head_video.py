#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import lightly_train


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/root/autodl-tmp/runs/20260720_220833_dinov3_vitl16_eomt/"
            "exported_models/exported_best.pt"
        ),
    )
    parser.add_argument("--video-dir", type=Path, default=Path("test/video"))
    parser.add_argument("--out-dir", type=Path, default=Path("test/infer_video"))
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def list_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def predict_head(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    with torch.inference_mode(), autocast_context(device, precision):
        x, meta = model.preprocess_image(image)  # type: ignore[attr-defined]
        batch = model.preprocess_batch([x])  # type: ignore[attr-defined]
        raw_outputs = model.forward_backend(batch)  # type: ignore[attr-defined]
        mask_logits, class_logits = raw_outputs

        crop_logits = model.to_per_pixel_logits_semantic(  # type: ignore[attr-defined]
            mask_logits, class_logits
        )
        logits = model.untile(  # type: ignore[attr-defined]
            crop_logits=crop_logits,
            origins=meta["origins"],
            image_sizes=[(meta["resized_h"], meta["resized_w"])],
        )[0]
        logits = logits[:-1].unsqueeze(0)
        logits = F.interpolate(
            logits,
            size=(meta["orig_h"], meta["orig_w"]),
            mode="bilinear",
        )[0]
        scores = logits.float().cpu().numpy()

    internal_mask = scores.argmax(axis=0).astype(np.int64)
    class_ids = model.internal_class_to_class.cpu().numpy()  # type: ignore[attr-defined]
    mask = class_ids[internal_mask].astype(np.uint8)

    head_index = None
    for internal_id, class_id in enumerate(class_ids.tolist()):
        if int(class_id) == 1:
            head_index = internal_id
            break
    if head_index is None:
        raise ValueError("Model does not contain class id 1 for head.")

    head_score = scores[head_index]
    head_pixels = mask == 1
    if np.any(head_pixels):
        confidence = float(head_score[head_pixels].mean())
    else:
        confidence = float(head_score.max())
    return mask, head_score, confidence


def draw_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    confidence: float,
    alpha: float,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    head_pixels = mask == 1
    if np.any(head_pixels):
        overlay = annotated.copy()
        overlay[head_pixels] = (0, 0, 255)
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0)

        contours, _ = cv2.findContours(
            head_pixels.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(annotated, contours, -1, (0, 255, 255), 2)

    text = f"head_conf={confidence:.4f}"
    cv2.rectangle(annotated, (18, 18), (430, 72), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        text,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def infer_video(
    model: torch.nn.Module,
    video_path: Path,
    out_dir: Path,
    device: torch.device,
    precision: str,
    alpha: float,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = video_path.stem

    video_out = out_dir / f"{stem}_infer.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create video writer: {video_out}")

    best: dict[str, Any] = {
        "confidence": -1.0,
        "frame_index": -1,
        "frame": None,
        "mask": None,
        "score": None,
    }
    frame_index = 0
    rows: list[dict[str, Any]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask, score, confidence = predict_head(
            model=model,
            frame_bgr=frame,
            device=device,
            precision=precision,
        )
        annotated = draw_overlay(frame, mask, confidence, alpha)
        writer.write(annotated)

        rows.append(
            {
                "video": video_path.name,
                "frame_index": frame_index,
                "confidence": f"{confidence:.8f}",
                "head_pixels": int((mask == 1).sum()),
            }
        )
        if confidence > best["confidence"]:
            best = {
                "confidence": confidence,
                "frame_index": frame_index,
                "frame": annotated.copy(),
                "mask": mask.copy(),
                "score": score.copy(),
            }
        frame_index += 1

    cap.release()
    writer.release()

    if best["frame"] is not None:
        cv2.imwrite(str(out_dir / f"{stem}_best_frame.jpg"), best["frame"])
        cv2.imwrite(str(out_dir / f"{stem}_best_mask.png"), best["mask"] * 255)
        score_u8 = np.clip(best["score"] * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}_best_score.png"), score_u8)

    with (out_dir / f"{stem}_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f, fieldnames=["video", "frame_index", "confidence", "head_pixels"]
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    return {
        "video": video_path.name,
        "frames": frame_index,
        "output_video": str(video_out),
        "best_frame_index": int(best["frame_index"]),
        "best_confidence": float(best["confidence"]),
    }


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    videos = list_videos(args.video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under: {args.video_dir}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise ValueError(f"Output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lightly_train.load_model(args.model)
    model.to(device)
    model.eval()

    summary = []
    for video_path in videos:
        result = infer_video(
            model=model,
            video_path=video_path,
            out_dir=args.out_dir,
            device=device,
            precision=args.precision,
            alpha=args.alpha,
        )
        summary.append(result)
        print(
            f"{video_path.name}: best frame {result['best_frame_index']} "
            f"confidence {result['best_confidence']:.4f}"
        )

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved inference outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
