"""Optional epoch-level validation and checkpoint helpers for the fast loop."""
from __future__ import annotations

import copy
import queue
import threading
from pathlib import Path
from typing import Any, Callable

import torch


def validation_context():
    return torch.inference_mode()


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        source = value.detach()
        if source.device.type != "cuda":
            return source.clone()
        try:
            target = torch.empty_like(source, device="cpu", pin_memory=True)
            target.copy_(source, non_blocking=True)
            return target
        except RuntimeError:
            return source.cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _contains_cuda(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device.type == "cuda"
    if isinstance(value, dict):
        return any(_contains_cuda(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_cuda(item) for item in value)
    return False


class AsyncCheckpointWriter:
    """Copy one checkpoint snapshot off GPU, then serialize on a worker thread.

    The caller supplies the exact Lightly checkpoint payload and save function;
    this helper changes scheduling only. `flush()` propagates worker failures.
    """

    def __init__(self, save_fn: Callable[[dict[str, Any], str | Path], None] | None = None) -> None:
        self._save_fn = save_fn or torch.save
        self._tasks: queue.Queue[tuple[dict[str, Any], str | Path, torch.cuda.Event | None] | None] = queue.Queue(maxsize=1)
        self._error: BaseException | None = None
        self._closed = False
        self._worker = threading.Thread(target=self._run, daemon=False)
        self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._tasks.get()
            try:
                if item is None:
                    return
                state, path, ready = item
                if ready is not None:
                    ready.synchronize()
                self._save_fn(state, path)
            except BaseException as exc:
                self._error = exc
            finally:
                self._tasks.task_done()

    def submit(self, state: dict[str, Any], path: str | Path) -> None:
        if self._closed:
            raise RuntimeError("checkpoint writer is closed")
        self.flush()
        uses_cuda = _contains_cuda(state)
        snapshot = _cpu_snapshot(state)
        ready = None
        if uses_cuda:
            ready = torch.cuda.Event()
            ready.record()
        self._tasks.put((snapshot, path, ready))

    def flush(self) -> None:
        self._tasks.join()
        if self._error is not None:
            raise RuntimeError("asynchronous checkpoint save failed") from self._error

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._closed = True
            self._tasks.put(None)
            self._worker.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
