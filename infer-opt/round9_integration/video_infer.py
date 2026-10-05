#!/usr/bin/env python3
"""Round 9: pipelined DINOv3-EoMT head-segmentation video inference.

Drop-in replacement for scripts/infer_dinov3_eomt_head_video.py (same outputs: overlay video,
frames CSV, best frame / mask) built from:
  * decode thread: cv2.VideoCapture.read() straight into a pinned ring buffer (no host memcpy)
  * per frame: H2D (copy stream) -> [prep + model_fn + post graph] (compute stream)
    -> D2H of uint8 mask + 4 int64 stats (copy stream), chained with CUDA events
  * result thread: waits the D2H event, computes confidence, hands (idx, mask, conf) to a consumer
  * optional overlay/video-writer thread
`model_fn(batch) -> (mask_logits_lr (B,Q,h,w), class_logits (B,Q,3))` is pluggable and captured
with pre/post kernels in a per-slot CUDA graph.

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
R7 = HERE.parent / "round7_head_postprocess"  # postfuse kernels
sys.path.insert(0, str(R7 / "impl"))
sys.path.insert(0, str(HERE.parent / "common"))

import head as hd  # noqa: E402
sys.path.insert(0, str(HERE / "impl"))
import model_impls as mi  # noqa: E402
import postfuse as pf  # noqa: E402

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
HEAD_CLASS_ID = 1


def load_orig_script():
    spec = importlib.util.spec_from_file_location(
        "orig_head_video", HERE.parents[1] / "scripts" / "infer_dinov3_eomt_head_video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_model_impl(requested: str, height: int, width: int, crop: int) -> tuple[str, str | None]:
    rh, rw = pf.resized_size(height, width, crop)
    n_crops = len(pf.compute_tiles(rh, rw))
    if n_crops > 8:
        raise ValueError(f"Frame requires {n_crops} crops; fused postprocess supports at most 8")
    if height > width:
        return "lightly", "portrait video"
    if n_crops != 2:
        return "lightly", f"{n_crops} crops (optimized kernels are tuned for 2)"
    return requested, None


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
        if slots < 1 or inflight < 1:
            raise ValueError("slots and inflight must be positive")
        self.model_fn, self.H, self.W, self.S = model_fn, height, width, slots
        self.inflight = inflight
        self.dev = device
        crop = min(model.image_size)
        dtype = torch.bfloat16  # static bf16 model input (model_fn contract)
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
        self.graphs: list[Any] = [None] * slots
        self.use_graph = use_graph
        self._warmup()

    # -- kernel launches (graph-capturable, current stream, no host sync)
    def _prep_launch(self, s: int):
        pf.ext().prep_launch(self.frame_gpu[s], *self.tables, self.starts, self.crop,
                             self.mean, self.std, self.batch)

    def _post_launch(self, s: int, ml, cl):
        e = pf.ext()
        e.post_a_launch(ml.contiguous(), cl.contiguous(), self.lr, self.starts, self.crop)
        e.post_b_launch(self.lr, self.mask_g[s], self.stats_g[s], None, self.class_ids[0],
                        self.class_ids[1], self.head_ch, HEAD_CLASS_ID)

    def _frame(self, s: int):
        """prep -> model_fn -> post for slot s (everything that goes into the per-slot CUDA graph)."""
        self._prep_launch(s)
        ml, cl = self.model_fn(self.batch)
        self._post_launch(s, ml, cl)

    def _warmup(self):
        with torch.inference_mode(), torch.cuda.stream(self.comp):
            for s in range(self.S):
                self.frame_gpu[s].zero_()
            for _ in range(3):  # also triggers JIT compiles / autotune / workspace allocation
                self._frame(0)
        self.comp.synchronize()
        if self.use_graph:
            pool = None
            for s in range(self.S):
                g = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(g, pool=pool, stream=self.comp):
                    self._frame(s)
                pool = g.pool() if pool is None else pool
                self.graphs[s] = g
            torch.cuda.synchronize()

    # -- one frame
    def submit(self, s: int):
        """Enqueue H2D -> [one CUDA graph: prep+model+post] -> D2H for the frame in pinned slot s."""
        with torch.inference_mode():
            with torch.cuda.stream(self.h2d):
                self.frame_gpu[s].copy_(self.pinned[s], non_blocking=True)
                self.ev_h2d[s].record(self.h2d)
            self.comp.wait_event(self.ev_h2d[s])
            with torch.cuda.stream(self.comp):
                if self.use_graph:
                    self.graphs[s].replay()
                else:
                    self._frame(s)
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
        stop = threading.Event()
        stats = {"lat_ms": [], "svc_ms": [], "t_first": None, "t_last": None, "err": None, "t_res": []}

        def decode():
            try:
                idx = 0
                for _ in range(loops):
                    for vp in video_paths:
                        if stop.is_set():
                            return
                        cap = cv2.VideoCapture(str(vp))
                        try:
                            if not cap.isOpened():
                                raise RuntimeError(f"Could not open video: {vp}")
                            seen = 0
                            while not stop.is_set():
                                try:
                                    s = free.get(timeout=0.1)
                                except queue.Empty:
                                    continue
                                if stop.is_set():
                                    free.put(s)
                                    break
                                ok, img = cap.read(self.pinned_np[s])
                                if not ok:
                                    free.put(s)
                                    break
                                assert img.ctypes.data == self.pinned_np[s].ctypes.data
                                decoded.put((s, idx, time.perf_counter()))
                                idx += 1
                                seen += 1
                        finally:
                            cap.release()
                        if seen == 0 and not stop.is_set():
                            raise RuntimeError(f"Video contains no decodable frames: {vp}")
            except Exception as e:  # noqa: BLE001
                stats["err"] = e
                stop.set()
            finally:
                decoded.put(None)

        def results():
            current = None
            current_inflight = False
            try:
                while True:
                    item = pending.get()
                    if item is None:
                        return
                    s, idx, t_in, t_sub = item
                    current, current_inflight = s, True
                    self.ev_d2h[s].synchronize()
                    t_out = time.perf_counter()
                    inflight.release()
                    current_inflight = False
                    stats["lat_ms"].append((t_out - t_in) * 1e3)
                    stats["svc_ms"].append((t_out - t_sub) * 1e3)
                    stats["t_res"].append(t_out)
                    conf = self.confidence(s)
                    taken = consumer(idx, self.mask_h_np[s], conf, self.pinned_np[s], s, free.put)
                    if not taken:
                        free.put(s)
                    current = None
            except Exception as e:  # noqa: BLE001
                stats["err"] = e
                stop.set()
                if current is not None:
                    if current_inflight:
                        inflight.release()
                    free.put(current)
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
        try:
            while True:
                item = decoded.get()
                if item is None:
                    break
                s, idx, t_in = item
                if stop.is_set():
                    free.put(s)
                    continue
                inflight.acquire()
                if stop.is_set():
                    inflight.release()
                    free.put(s)
                    continue
                t_sub = time.perf_counter()
                try:
                    self.submit(s)
                except Exception:
                    inflight.release()
                    free.put(s)
                    raise
                pending.put((s, idx, t_in, t_sub))
                n += 1
        except Exception as e:  # noqa: BLE001
            stats["err"] = e
            stop.set()
        finally:
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
        self.error: BaseException | None = None
        if out_video is not None:
            self.writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            assert self.writer.isOpened()
        self.release_fn = None
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def __call__(self, idx, mask, conf, frame, slot, release):
        if self.error is not None:
            raise RuntimeError("overlay consumer failed") from self.error
        self.q.put((idx, mask, conf, frame, slot, release))
        return True  # we own the slot until written

    def _loop(self):
        current = None
        try:
            while True:
                current = self.q.get()
                if current is None:
                    return
                idx, mask, conf, frame, slot, release = current
                annotated = self.orig.draw_overlay(frame, mask, conf, self.alpha)
                if self.writer is not None:
                    self.writer.write(annotated)
                self.rows.append({"frame_index": idx, "confidence": f"{conf:.8f}",
                                  "head_pixels": int((mask == 1).sum())})
                if conf > self.best["confidence"]:
                    self.best = {"confidence": conf, "frame_index": idx, "frame": annotated.copy(),
                                 "mask": mask.copy()}
                release(slot)
                current = None
        except BaseException as e:  # surface writer/overlay errors to the caller
            self.error = e
            if current is not None:
                current[5](current[4])
            while True:
                item = self.q.get()
                if item is None:
                    return
                item[5](item[4])

    def close(self):
        self.q.put(None)
        self.th.join()
        if self.writer is not None:
            self.writer.release()
        if self.error is not None:
            raise RuntimeError("overlay consumer failed") from self.error


def infer_video_fast(pipe: VideoPipeline, orig, video_path: Path, out_dir: Path, alpha: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    stem = video_path.stem
    sink = OverlaySink(orig, out_dir / f"{stem}_infer.mp4", fps, (w, h), alpha)
    try:
        stats = pipe.run([str(video_path)], sink)
    finally:
        sink.close()
    if stats["frames"] == 0:
        raise RuntimeError(f"Video contains no decodable frames: {video_path}")
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
    ap.add_argument("--model-impl", default="r5_v1", choices=list(mi.IMPLS))
    ap.add_argument("--fp8", action="store_true", help="FP8 GEMMs (impl=lightly only)")
    args = ap.parse_args()
    if args.precision != "bf16-mixed":
        ap.error("Round 9 model implementations only support --precision bf16-mixed")
    if args.slots < 1:
        ap.error("--slots must be positive")
    if args.fp8 and args.model_impl != "lightly":
        ap.error("--fp8 requires --model-impl lightly")
    import lightly_train

    orig = load_orig_script()
    output_root = args.out_dir.resolve()
    videos = [vp for vp in orig.list_videos(args.video_dir)
              if not vp.resolve().is_relative_to(output_root)]
    if not videos:
        ap.error(f"No video files found under: {args.video_dir}")
    video_specs = []
    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {vp}")
        hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
        if min(hw) < 1:
            raise RuntimeError(f"Invalid video dimensions {hw} for: {vp}")
        video_specs.append((vp, hw))
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        ap.error(f"Output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = lightly_train.load_model(args.model)
    model.to("cuda").eval()
    crop = min(model.image_size)
    selected_specs = []
    for vp, hw in video_specs:
        impl, reason = select_model_impl(args.model_impl, hw[0], hw[1], crop)
        if args.fp8 and reason is not None:
            ap.error(f"{vp.name}: {reason}; FP8 requires the validated wide two-crop shape")
        selected_specs.append((vp, hw, impl, reason))
    summary = []
    pipes: dict[tuple[int, int], VideoPipeline] = {}
    name = "vitl16" if model.backbone.embed_dim == 1024 else "vitb16"
    needed = {impl for _, _, impl, _ in selected_specs}
    model_fns = {impl: mi.make_model_fn(name, model, impl, args.fp8 and impl == "lightly")
                 for impl in dict.fromkeys((args.model_impl, "lightly")) if impl in needed}
    for vp, hw, impl, reason in selected_specs:
        if reason is not None and impl != args.model_impl:
            print(f"{vp.name}: {reason}; using lightly for accuracy (requested {args.model_impl})", flush=True)
        else:
            print(f"{vp.name}: using {impl}", flush=True)
        if hw not in pipes:
            pipes[hw] = VideoPipeline(model, model_fns[impl], hw[0], hw[1], slots=args.slots, inflight=3)
        r = infer_video_fast(pipes[hw], orig, vp, args.out_dir, args.alpha)
        r["model_impl"] = impl
        summary.append(r)
        print(f"{vp.name}: best frame {r['best_frame_index']} confidence {r['best_confidence']:.4f} "
              f"({r['frames']} frames, {r['seconds']:.1f}s)")
    import json
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
