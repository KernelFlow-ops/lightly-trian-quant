from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, "train-opt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_adamw_matches_torch_one_step() -> None:
    from round7_optimizer.impl.cuda_adamw import fused_adamw_step

    torch.cuda.set_per_process_memory_fraction(0.125)
    torch.manual_seed(123)
    initial = torch.randn(257, device="cuda", dtype=torch.float32)
    grad = torch.randn_like(initial)
    ref = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.AdamW(
        [ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.05, foreach=False,
    )
    ref.grad = grad.detach().clone()
    torch.nn.utils.clip_grad_norm_([ref], 0.01)
    optimizer.step()

    actual = initial.detach().clone()
    actual_grad = grad.detach().clone()
    exp_avg = torch.zeros_like(actual)
    exp_avg_sq = torch.zeros_like(actual)
    step = torch.zeros((), device="cuda", dtype=torch.float32)
    lr = torch.tensor(1e-3, device="cuda", dtype=torch.float32)
    wd = torch.tensor(0.05, device="cuda", dtype=torch.float32)
    fused_adamw_step(
        [actual], [actual_grad], [exp_avg], [exp_avg_sq],
        [step], [lr], [wd], [], 0.01, 0.9, 0.999, 1e-8,
    )
    torch.cuda.synchronize()
    assert float((actual - ref).abs().max()) <= 3e-6
    assert float(step) == 1.0
    assert bool(torch.isfinite(actual).all())
