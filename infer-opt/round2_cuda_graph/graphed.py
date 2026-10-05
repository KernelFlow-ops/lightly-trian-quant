"""Reusable CUDA Graph wrapper.

    g = GraphedModel(model_fn, static_shape=(2, 3, 512, 512), dtype=torch.bfloat16)
    out = g(x)          # x is copied into the static input buffer, graph replayed
    # `out` is the static output buffer (tuple/tensor): consume it (or clone) before the next call.

`model_fn` must be capture-safe: no host sync, no dynamic shapes, no allocation-order
dependence beyond what the graph private pool handles (torch allocates from the graph pool).
Warmup runs on a side stream (also lets cuDNN/cuBLAS pick algorithms and allocate workspaces
before capture).
"""
from __future__ import annotations

from typing import Any, Callable

import torch


class GraphedModel:
    def __init__(self, model_fn: Callable[[torch.Tensor], Any], static_shape: tuple[int, ...],
                 dtype: torch.dtype = torch.bfloat16, device: str = "cuda", warmup: int = 3) -> None:
        self.static_in = torch.zeros(static_shape, dtype=dtype, device=device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(warmup):
                    model_fn(self.static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph):
                self.static_out = model_fn(self.static_in)
        torch.cuda.synchronize()

    def __call__(self, x: torch.Tensor):
        self.static_in.copy_(x)  # dtype cast (fp32 -> bf16) happens inside the copy kernel
        self.graph.replay()
        return self.static_out
