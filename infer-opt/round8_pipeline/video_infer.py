#!/usr/bin/env python3
"""Round 8: pipelined DINOv3-EoMT head-segmentation video inference.

Drop-in replacement for scripts/infer_dinov3_eomt_head_video.py (same outputs: overlay video,
frames CSV, best frame / mask) built from:
  * decode thread: cv2.VideoCapture.read() straight into a pinned ring buffer (no host memcpy)
  * per frame: H2D (copy stream) -> [prep graph -> model_fn -> post graph] (compute stream)
    -> D2H of uint8 mask + 4 int64 stats (copy stream), chained with CUDA events
  * result thread: waits the D2H event, computes confidence, hands (idx, mask, conf) to a consumer
  * optional overlay/video-writer thread
`model_fn(batch) -> (mask_logits_lr (B,Q,h,w), class_logits (B,Q,3))` is pluggable and runs on the
compute stream (prep/post kernels are CUDA-graph captured, model_fn is called eagerly).

Usage (drop-in):
  python video_infer.py --model /root/autodl-tmp/vitl-best.pth --video-dir test/video --out-dir out
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("LIGHTLY_TRAIN_EVENTS_DISABLED", "1")

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R7 = HERE.parent / "round7_head_postprocess"
sys.path.insert(0, str(R7 / "impl"))
sys.path.insert(0, str(HERE.parent / "common"))

import head as hd  # noqa: E402
import postfuse as pf  # noqa: E402

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
HEAD_CLASS_ID = 1


def load_orig_script():
    spec = importlib.util.spec_from_file_location(
        "orig_head_video", "/root/lightly-train/scripts/infer_dinov3_eomt_head_video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- pluggable model
def make_eager_model_fn(model, precision: str = "bf16-mixed", fused_head: bool = True):
    """Default model_fn: eager backbone under autocast + P3 fused head. Returns low-res logits."""
    if fused_head:
        hd.enable(model)

    def model_fn(batch: torch.Tensor):
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16-mixed"
               else torch.autocast("cuda", enabled=False))
        with ctx:
            ml, cl = model.forward_train(batch, return_logits_per_layer=False)
        return ml[-1], cl[-1]

    return model_fn


# ---------------------------------------------------------------- pipeline
class VideoPipeline:
    def __init__(self, model, model_fn: Callable, height: int, width: int, slots: int = 4,
                 inflight: int = 2, use_graph: bool = True, device: str = "cuda"):
        self.model_fn, self.H, self.W, self.S = model_fn, height, width, slots
        self.inflight = inflight
        self.dev = device
        crop = min(model.image_size)
        dtype = next(model.parameters()).dtype
        norm = model.image_normalize
        fp = pf.FastPrep(norm["mean"], norm["std"], crop, dtype)
        st = fp._state(height, width)
        self.meta, self.crop = st["meta"], crop
        self.tables, self.starts, self.batch = st["tables"], st["starts"], st["out"]
        self.mean, self.std = fp.mean, fp.std
        class_ids = model.internal_class_to_class.tolist()
        assert len(class_ids) == 2
        self.class_ids = class_ids
        self.head_ch = class_ids.index(HEAD_CLASS_ID)
        rh, rw = self.meta["resized_h"], self.meta["resized_w"]
        self.lr = torch.empty((2, rh, rw), dtype=torch.float32, device=device)
        # pinned input ring / output ring (same slot index)
        self.pinned = [torch.empty((height, width, 3), dtype=torch.uint8).pin_memory()
                       for _ in range(slots)]
        self.pinned_np = [p.numpy() for p in self.pinned]
        self.frame_gpu = [torch.empty((height, width, 3), dtype=torch.uint8, device=device)
                          for _ in range(slots)]
        self.mask_g = [torch.empty((height, width), dtype=torch.uint8, device=device) for _ in range(slots)]
        self.stats_g = [torch.zeros(4, dtype=torch.int64, device=device) for _ in range(slots)]
        self.mask_h = [torch.empty((height, width), dtype=torch.uint8).pin_memory() for _ in range(slots)]
        self.stats_h = [torch.zeros(4, dtype=torch.int64).pin_memory() for _ in range(slots)]
        self.mask_h_np = [m.numpy() for m in self.mask_h]
        self.stats_h_np = [s.numpy() for s in self.stats_h]
        self.h2d = torch.cuda.Stream()
        self.comp = torch.cuda.Stream()
        self.d2h = torch.cuda.Stream()
        self.ev_h2d = [torch.cuda.Event() for _ in range(slots)]
        self.ev_comp = [torch.cuda.Event() for _ in range(slots)]
        self.ev_d2h = [torch.cuda.Event() for _ in range(slots)]
        self.ml_buf = self.cl_buf = None
        self.prep_g: list[Any] = [None] * slots
        self.post_g: list[Any] = [None] * slots
        self.use_graph = use_graph
        self._warmup()

    # -- kernel launches (graph-capturable, current stream, no host sync)
    def _prep_launch(self, s: int):
        pf.ext().prep_launch(self.frame_gpu[s], *self.tables, self.starts, self.crop,
                             self.mean, self.std, self.batch)

    def _post_launch(self, s: int):
        e = pf.ext()
        e.post_a_launch(self.ml_buf, self.cl_buf, self.lr, self.starts, self.crop)
        e.post_b_launch(self.lr, self.mask_g[s], self.stats_g[s], None, self.class_ids[0],
                        self.class_ids[1], self.head_ch, HEAD_CLASS_ID)

    def _warmup(self):
        with torch.inference_mode(), torch.cuda.stream(self.comp):
            for s in range(self.S):
                self.frame_gpu[s].zero_()
            self._prep_launch(0)
            for _ in range(3):
                ml, cl = self.model_fn(self.batch)
            self.ml_buf, self.cl_buf = ml.contiguous().clone(), cl.contiguous().clone()
            for s in range(self.S):
                self._post_launch(s)
        self.comp.synchronize()
        if self.use_graph:
            for s in range(self.S):
                for name, fn in (("prep_g", self._prep_launch), ("post_g", self._post_launch)):
                    g = torch.cuda.CUDAGraph()
                    with torch.inference_mode(), torch.cuda.graph(g, stream=self.comp):
                        fn(s)
                    getattr(self, name)[s] = g
            torch.cuda.synchronize()

    # -- one frame
    def submit(self, s: int):
        """Enqueue H2D -> prep -> model -> post -> D2H for the frame in pinned slot s."""
        with torch.inference_mode():
            with torch.cuda.stream(self.h2d):
                self.frame_gpu[s].copy_(self.pinned[s], non_blocking=True)
                self.ev_h2d[s].record(self.h2d)
            self.comp.wait_event(self.ev_h2d[s])
            with torch.cuda.stream(self.comp):
                if self.use_graph:
                    self.prep_g[s].replay()
                else:
                    self._prep_launch(s)
                ml, cl = self.model_fn(self.batch)
                self.ml_buf.copy_(ml)
                self.cl_buf.copy_(cl)
                if self.use_graph:
                    self.post_g[s].replay()
                else:
                    self._post_launch(s)
                self.ev_comp[s].record(self.comp)
            self.d2h.wait_event(self.ev_comp[s])
            with torch.cuda.stream(self.d2h):
                self.mask_h[s].copy_(self.mask_g[s], non_blocking=True)
                self.stats_h[s].copy_(self.stats_g[s], non_blocking=True)
                self.ev_d2h[s].record(self.d2h)

    def confidence(self, s: int) -> float:
        st = self.stats_h_np[s]
        total, count = float(st[:1].view(np.float64)[0]), int(st[1])
        if count > 0:
            return total / count
        key = int(st[2]) & 0xFFFFFFFF
        bits = key ^ (0xFFFFFFFF if not (key >> 31) else 0x80000000)
        return float(np.array([bits], dtype=np.uint32).view(np.float32)[0])

    # -- streaming driver
    def run(self, video_paths: list[str], consumer: Callable, loops: int = 1,
            on_frame_count: Callable[[int], None] | None = None) -> dict[str, Any]:
        """consumer(idx, mask, conf, frame_bgr, slot, release) -> None. It must call release() when
        done with mask/frame (or return True-ish `keep`=False to auto release). For simplicity the
        consumer returns True if it takes ownership (and will call release(slot) later)."""
        S = self.S
        free: queue.Queue = queue.Queue()
        for s in range(S):
            free.put(s)
        decoded: queue.Queue = queue.Queue()
        pending: queue.Queue = queue.Queue()
        inflight = threading.Semaphore(self.inflight)
        stats = {"lat_ms": [], "svc_ms": [], "t_first": None, "t_last": None, "err": None, "t_res": []}

        def decode():
            try:
                idx = 0
                for _ in range(loops):
                    for vp in video_paths:
                        cap = cv2.VideoCapture(str(vp))
                        while True:
                            s = free.get()
                            ok, img = cap.read(self.pinned_np[s])
                            if not ok:
                                free.put(s)
                                break
                            assert img.ctypes.data == self.pinned_np[s].ctypes.data
                            decoded.put((s, idx, time.perf_counter()))
                            idx += 1
                        cap.release()
            except Exception as e:  # noqa: BLE001
                stats["err"] = e
            decoded.put(None)

        def results():
            try:
                while True:
                    item = pending.get()
                    if item is None:
                        return
                    s, idx, t_in, t_sub = item
                    self.ev_d2h[s].synchronize()
                    t_out = time.perf_counter()
                    inflight.release()
                    stats["lat_ms"].append((t_out - t_in) * 1e3)
                    stats["svc_ms"].append((t_out - t_sub) * 1e3)
                    stats["t_res"].append(t_out)
                    conf = self.confidence(s)
                    taken = consumer(idx, self.mask_h_np[s], conf, self.pinned_np[s], s, free.put)
                    if not taken:
                        free.put(s)
            except Exception as e:  # noqa: BLE001
                stats["err"] = e
                while True:  # drain so the producer does not deadlock
                    item = pending.get()
                    if item is None:
                        return
                    inflight.release()
                    free.put(item[0])

        td = threading.Thread(target=decode, daemon=True)
        tr = threading.Thread(target=results, daemon=True)
        td.start()
        tr.start()
        t0 = time.perf_counter()
        n = 0
        while True:
            item = decoded.get()
            if item is None:
                break
            s, idx, t_in = item
            inflight.acquire()
            t_sub = time.perf_counter()
            self.submit(s)
            pending.put((s, idx, t_in, t_sub))
            n += 1
        pending.put(None)
        tr.join()
        td.join()
        if stats["err"] is not None:
            raise stats["err"]
        stats["t_total"] = time.perf_counter() - t0
        stats["frames"] = n
        return stats


# ---------------------------------------------------------------- overlay / outputs
class OverlaySink:
    """Consumer: overlay + video writer thread (+ CSV rows, best frame)."""

    def __init__(self, orig, out_video: Path | None, fps: float, size: tuple[int, int], alpha: float,
                 keep_best: bool = True):
        self.orig, self.alpha = orig, alpha
        self.q: queue.Queue = queue.Queue()
        self.rows: list[dict[str, Any]] = []
        self.best = {"confidence": -1.0, "frame_index": -1, "frame": None, "mask": None}
        self.writer = None
        if out_video is not None:
            self.writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            assert self.writer.isOpened()
        self.release_fn = None
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def __call__(self, idx, mask, conf, frame, slot, release):
        self.q.put((idx, mask, conf, frame, slot, release))
        return True  # we own the slot until written

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            idx, mask, conf, frame, slot, release = item
            annotated = self.orig.draw_overlay(frame, mask, conf, self.alpha)
            if self.writer is not None:
                self.writer.write(annotated)
            self.rows.append({"frame_index": idx, "confidence": f"{conf:.8f}",
                              "head_pixels": int((mask == 1).sum())})
            if conf > self.best["confidence"]:
                self.best = {"confidence": conf, "frame_index": idx, "frame": annotated.copy(),
                             "mask": mask.copy()}
            release(slot)

    def close(self):
        self.q.put(None)
        self.th.join()
        if self.writer is not None:
            self.writer.release()


def infer_video_fast(pipe: VideoPipeline, orig, video_path: Path, out_dir: Path, alpha: float):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    stem = video_path.stem
    sink = OverlaySink(orig, out_dir / f"{stem}_infer.mp4", fps, (w, h), alpha)
    stats = pipe.run([str(video_path)], sink)
    sink.close()
    rows = sorted(sink.rows, key=lambda r: r["frame_index"])
    for r in rows:
        r["video"] = video_path.name
    b = sink.best
    if b["frame"] is not None:
        cv2.imwrite(str(out_dir / f"{stem}_best_frame.jpg"), b["frame"])
        cv2.imwrite(str(out_dir / f"{stem}_best_mask.png"), b["mask"] * 255)
    with (out_dir / f"{stem}_frames.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["video", "frame_index", "confidence", "head_pixels"])
        wr.writeheader()
        wr.writerows(rows)
    return {"video": video_path.name, "frames": stats["frames"], "best_frame_index": b["frame_index"],
            "best_confidence": b["confidence"], "seconds": stats["t_total"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, default=Path("test/video"))
    ap.add_argument("--out-dir", type=Path, default=Path("test/infer_video"))
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--slots", type=int, default=6)
    args = ap.parse_args()
    import lightly_train

    orig = load_orig_script()
    videos = orig.list_videos(args.video_dir)
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise ValueError(f"Output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = lightly_train.load_model(args.model)
    model.to("cuda").eval()
    summary = []
    pipes: dict[tuple[int, int], VideoPipeline] = {}
    model_fn = make_eager_model_fn(model, args.precision)
    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
        if hw not in pipes:
            pipes[hw] = VideoPipeline(model, model_fn, hw[0], hw[1], slots=args.slots, inflight=3)
        r = infer_video_fast(pipes[hw], orig, vp, args.out_dir, args.alpha)
        summary.append(r)
        print(f"{vp.name}: best frame {r['best_frame_index']} confidence {r['best_confidence']:.4f} "
              f"({r['frames']} frames, {r['seconds']:.1f}s)")
    import json
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
