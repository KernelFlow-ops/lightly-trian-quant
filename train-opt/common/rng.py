"""Named random tensor recording/replay for step equivalence tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class RngTape:
    mode: str = "off"
    values: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"off", "record", "replay"}:
            raise ValueError(f"Unsupported RngTape mode: {self.mode}")
        self._used: set[str] = set()

    def sample(self, name: str, draw: Callable[[], torch.Tensor], device: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.mode == "replay":
            if name not in self.values:
                raise KeyError(f"RngTape replay missing call point {name!r}")
            # Keep later untaped random calls (notably per-block RoPE) aligned.
            draw()
            value = self.values[name].to(device=device, dtype=dtype)
        else:
            value = draw()
            if self.mode == "record":
                self.values[name] = value.detach().cpu()
        self._used.add(name)
        return value

    def rand(self, name: str, shape: tuple[int, ...], device: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        value = self.sample(name, lambda: torch.rand(shape, device=device, dtype=dtype), device, dtype)
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"RngTape shape mismatch at {name}: {tuple(value.shape)} != {shape}")
        return value

    def uniform(self, name: str, shape: tuple[int, ...], low: float, high: float, *, device: Any) -> torch.Tensor:
        return self.rand(name, shape, device=device).mul(high - low).add(low)

    def reset_used(self) -> None:
        self._used.clear()

    def assert_complete(self) -> None:
        missing = set(self.values) - self._used
        if missing:
            raise AssertionError(f"RngTape values not consumed: {sorted(missing)}")
