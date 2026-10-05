"""CPU-only regression checks for video worker error propagation."""
from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import numpy as np

import video_infer as vi


class _Event:
    def synchronize(self):
        pass


class _Capture:
    opened = True

    def __init__(self, path):
        self.frame = 0

    def isOpened(self):
        return self.opened

    def read(self, buf):
        self.frame += 1
        if self.frame > 100:
            return False, None
        return True, buf

    def release(self):
        pass


def _pipe():
    pipe = object.__new__(vi.VideoPipeline)
    pipe.S = 2
    pipe.inflight = 2
    pipe.pinned_np = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
    pipe.mask_h_np = [np.zeros((2, 2), dtype=np.uint8) for _ in range(2)]
    pipe.ev_d2h = [_Event(), _Event()]
    pipe.submit = lambda slot: None
    pipe.confidence = lambda slot: 0.5
    return pipe


class VideoEntryTests(unittest.TestCase):
    def test_model_impl_falls_back_outside_tuned_shape(self):
        self.assertEqual(vi.select_model_impl("r5_v1", 1440, 1900, 512), ("r5_v1", None))
        self.assertEqual(vi.select_model_impl("r5_v1", 1900, 1440, 512)[0], "lightly")
        self.assertEqual(vi.select_model_impl("r5_v1", 720, 3840, 512)[0], "lightly")
        self.assertEqual(vi.select_model_impl("r5_v1", 512, 512, 512)[0], "lightly")
        with self.assertRaisesRegex(ValueError, "at most 8"):
            vi.select_model_impl("r5_v1", 512, 4608, 512)

    def test_consumer_failure_does_not_deadlock_workers(self):
        errors = []

        def fail(*args):
            raise RuntimeError("consumer failed")

        def run():
            try:
                _pipe().run(["input.mp4"], fail)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        with patch.object(vi.cv2, "VideoCapture", _Capture):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            worker.join(timeout=3)
        self.assertFalse(worker.is_alive(), "video workers remained blocked after consumer failure")
        self.assertEqual(str(errors[0]), "consumer failed")

    def test_open_failure_reaches_caller(self):
        class ClosedCapture(_Capture):
            opened = False

        with patch.object(vi.cv2, "VideoCapture", ClosedCapture):
            with self.assertRaisesRegex(RuntimeError, "Could not open video"):
                _pipe().run(["missing.mp4"], lambda *args: False)

    def test_overlay_failure_releases_slot_and_reaches_caller(self):
        class BadOverlay:
            def draw_overlay(self, *args):
                raise RuntimeError("overlay failed")

        released = []
        sink = vi.OverlaySink(BadOverlay(), None, 30, (2, 2), 0.45)
        sink(0, np.zeros((2, 2), np.uint8), 0.5, np.zeros((2, 2, 3), np.uint8), 1, released.append)
        with self.assertRaisesRegex(RuntimeError, "overlay consumer failed"):
            sink.close()
        self.assertEqual(released, [1])


if __name__ == "__main__":
    unittest.main()
