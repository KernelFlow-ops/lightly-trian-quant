"""Python wrappers for the Round 7 fused pre/post-processing kernels (postfuse_ext.cu).

Everything is parameterised by the frame shape (no hard-coded 1080p / 910); per-shape
tables and buffers are created once and cached, and all launches happen on the current
CUDA stream (CUDA-graph friendly except for the final host copy in `FastPost`).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        os.environ.setdefault("PATH", "")
        for p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
            if p not in os.environ["PATH"]:
                os.environ["PATH"] = p + ":" + os.environ["PATH"]
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from torch.utils.cpp_extension import load

        build = Path("/root/autodl-tmp/infer-opt-tmp/postfuse/build")
        build.mkdir(parents=True, exist_ok=True)
        _EXT = load(
            name="postfuse_ext",
            sources=[str(_HERE / "postfuse_ext.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_120a"],
            build_directory=str(build),
            verbose=False,
        )
    return _EXT


# ---------------------------------------------------------------- geometry helpers
def compute_tiles(resized_h: int, resized_w: int) -> list[tuple[int, int, int, bool]]:
    """Same arithmetic as task_model.tile (image_idx 0)."""
    h, w = resized_h, resized_w
    long_side, short_side = max(h, w), min(h, w)
    is_tall = h > w
    crop = short_side
    num_crops = math.ceil(long_side / crop)
    overlap = num_crops * crop - long_side
    overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0
    origins = []
    for j in range(num_crops):
        start = int(j * (crop - overlap_per_crop))
        origins.append((0, start, start + crop, is_tall))
    return origins


def resized_size(h: int, w: int, short: int) -> tuple[int, int]:
    """torchvision `_compute_resized_output_size` for size=[short]."""
    if w <= h:
        return int(short * h / w), short
    return short, int(short * w / h)


def aa_table(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ATen `_compute_indices_min_size_weights_aa` for bilinear + antialias, align_corners=False.

    Returns (xmin[int32], xsize[int32], weights[float32, (out, K)]) with float32 arithmetic.
    """
    f32 = np.float32
    scale = f32(in_size) / f32(out_size)
    support = scale * f32(1.0) if scale >= 1.0 else f32(1.0)  # interp_size(2) * 0.5 * scale
    invscale = f32(1.0) / scale if scale >= 1.0 else f32(1.0)
    mins = np.zeros(out_size, np.int32)
    sizes = np.zeros(out_size, np.int32)
    rows = []
    for i in range(out_size):
        center = scale * (f32(i) + f32(0.5))
        xmin = max(int(center - support + f32(0.5)), 0)
        xsize = min(int(center + support + f32(0.5)), in_size) - xmin
        w = np.array(
            [max(f32(0.0), f32(1.0) - abs((f32(j + xmin) - center + f32(0.5)) * invscale))
             for j in range(xsize)], dtype=np.float32)
        total = w.sum(dtype=np.float32)
        w = w / total if total != 0 else w
        mins[i], sizes[i] = xmin, xsize
        rows.append(w)
    K = int(sizes.max())
    wt = np.zeros((out_size, K), np.float32)
    for i, w in enumerate(rows):
        wt[i, : len(w)] = w
    return mins, sizes, wt


# ---------------------------------------------------------------- P1
class FastPrep:
    """BGR uint8 frame (host) -> (n_crops, 3, crop, crop) normalized batch on GPU."""

    def __init__(self, mean, std, crop: int, dtype: torch.dtype = torch.float32,
                 device: str = "cuda"):
        self.mean, self.std = [float(m) for m in mean], [float(s) for s in std]
        self.crop, self.dtype, self.device = crop, dtype, device
        self._cache: dict[tuple[int, int], dict[str, Any]] = {}

    def _state(self, h: int, w: int) -> dict[str, Any]:
        key = (h, w)
        st = self._cache.get(key)
        if st is not None:
            return st
        rh, rw = resized_size(h, w, self.crop)
        origins = compute_tiles(rh, rw)
        ymin, ysz, ywt = aa_table(h, rh)
        xmin, xsz, xwt = aa_table(w, rw)
        dev = self.device
        t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
        pinned = torch.empty((h, w, 3), dtype=torch.uint8).pin_memory()
        st = {
            "meta": {"orig_h": h, "orig_w": w, "resized_h": rh, "resized_w": rw,
                     "origins": origins, "num_crops": len(origins)},
            "starts": [o[1] for o in origins],
            "tables": (t(ymin), t(ysz), t(ywt), t(xmin), t(xsz), t(xwt)),
            "pinned": pinned, "pinned_np": pinned.numpy(),
            "frame_gpu": torch.empty((h, w, 3), dtype=torch.uint8, device=dev),
            "out": torch.empty((len(origins), 3, self.crop, self.crop), dtype=self.dtype,
                               device=dev),
            "ev_h2d": torch.cuda.Event(),
        }
        self._cache[key] = st
        return st

    def __call__(self, frame_bgr: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        """Returns (batch, meta). `batch` is a persistent buffer reused by the next call."""
        h, w = frame_bgr.shape[:2]
        st = self._state(h, w)
        st["ev_h2d"].synchronize()  # previous async H2D must have consumed the pinned buffer
        np.copyto(st["pinned_np"], frame_bgr)  # host staging (pinned) -> async H2D
        st["frame_gpu"].copy_(st["pinned"], non_blocking=True)
        st["ev_h2d"].record()
        self.launch(st)
        return st["out"], st["meta"]

    def launch(self, st: dict[str, Any]) -> None:
        ext().prep_launch(st["frame_gpu"], *st["tables"], st["starts"], self.crop,
                          self.mean, self.std, st["out"])


# ---------------------------------------------------------------- P2
class FastPost:
    """Low-res mask logits + class logits -> (uint8 mask on host, head confidence)."""

    def __init__(self, class_ids: list[int], head_class_id: int = 1, crop: int = 512,
                 device: str = "cuda"):
        assert len(class_ids) == 2, "kernel handles K=2 known classes (+1 void)"
        self.class_ids = [int(c) for c in class_ids]
        self.head_class_id = head_class_id
        self.head_ch = self.class_ids.index(head_class_id)
        self.crop, self.device = crop, device
        self._cache: dict[tuple, dict[str, Any]] = {}

    def _state(self, meta: dict[str, Any]) -> dict[str, Any]:
        key = (meta["orig_h"], meta["orig_w"], meta["resized_h"], meta["resized_w"])
        st = self._cache.get(key)
        if st is None:
            oh, ow = meta["orig_h"], meta["orig_w"]
            rh, rw = meta["resized_h"], meta["resized_w"]
            dev = self.device
            st = {
                "starts": [o[1] for o in meta["origins"]],
                "lr": torch.empty((2, rh, rw), dtype=torch.float32, device=dev),
                "mask": torch.empty((oh, ow), dtype=torch.uint8, device=dev),
                "stats": torch.zeros(4, dtype=torch.int64, device=dev),
                "mask_h": torch.empty((oh, ow), dtype=torch.uint8).pin_memory(),
                "stats_h": torch.zeros(4, dtype=torch.int64).pin_memory(),
                "logits": None,
            }
            self._cache[key] = st
        return st

    def __call__(self, mask_logits_lr: torch.Tensor, class_logits: torch.Tensor,
                 meta: dict[str, Any], return_logits: bool = False):
        """Returns (mask uint8 (H,W) numpy view on pinned memory, confidence[, logits (2,H,W)]).

        The mask view is only valid until the next call (persistent pinned buffer).
        `return_logits` is a debug switch: kernel B also writes fp32 logits (2,H,W).
        """
        st = self._state(meta)
        e = ext()
        e.post_a_launch(mask_logits_lr.contiguous(), class_logits.contiguous(), st["lr"],
                        st["starts"], self.crop)
        logits = None
        if return_logits:
            if st["logits"] is None:
                st["logits"] = torch.empty((2, meta["orig_h"], meta["orig_w"]),
                                           dtype=torch.float32, device=self.device)
            logits = st["logits"]
        e.post_b_launch(st["lr"], st["mask"], st["stats"], logits, self.class_ids[0],
                        self.class_ids[1], self.head_ch, self.head_class_id)
        st["mask_h"].copy_(st["mask"], non_blocking=True)
        st["stats_h"].copy_(st["stats"], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        s = st["stats_h"].numpy()
        total, count = float(s[:1].view(np.float64)[0]), int(s[1])
        if count > 0:
            conf = total / count
        else:
            key = int(s[2]) & 0xFFFFFFFF
            bits = key ^ (0xFFFFFFFF if not (key >> 31) else 0x80000000)
            conf = float(np.array([bits], dtype=np.uint32).view(np.float32)[0])
        mask = st["mask_h"].numpy()
        if return_logits:
            return mask, conf, logits
        return mask, conf
