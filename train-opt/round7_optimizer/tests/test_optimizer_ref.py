from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round7_optimizer.impl.optimizer_ref import clipped_adamw_reference_step


def test_reference_matches_torch_adamw_and_clip_for_100_steps() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(19)
    shapes = ((37, 19), (3, 7), (1,))
    baseline_params = [torch.nn.Parameter(torch.randn(shape)) for shape in shapes]
    candidate_params = [
        torch.nn.Parameter(value.detach().clone()) for value in baseline_params
    ]
    group_settings = [
        {"lr": 1e-4, "weight_decay": 0.05},
        {"lr": 8e-5, "weight_decay": 0.0},
        {"lr": 6.4e-5, "weight_decay": 0.05},
    ]
    baseline_groups = [
        {"params": [param], **settings}
        for param, settings in zip(baseline_params, group_settings)
    ]
    candidate_groups = [
        {"params": [param], **settings}
        for param, settings in zip(candidate_params, group_settings)
    ]
    optimizer = torch.optim.AdamW(baseline_groups, foreach=False)
    candidate_state = {}
    for _ in range(100):
        gradients = [torch.randn_like(value) * 0.01 for value in baseline_params]
        for baseline, candidate, grad in zip(
            baseline_params, candidate_params, gradients
        ):
            baseline.grad = grad.clone()
            candidate.grad = grad.clone()
        reference_norm = torch.nn.utils.clip_grad_norm_(baseline_params, 0.01)
        optimizer.step()
        candidate_norm = clipped_adamw_reference_step(
            candidate_groups, candidate_state, 0.01
        )
        assert torch.allclose(reference_norm, candidate_norm, rtol=1e-6, atol=1e-6)
    for baseline, candidate in zip(baseline_params, candidate_params):
        assert torch.allclose(baseline, candidate, rtol=1e-6, atol=1e-6)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.allclose(
                optimizer.state[baseline][key], candidate_state[candidate][key],
                rtol=1e-6, atol=1e-6,
            ), key
