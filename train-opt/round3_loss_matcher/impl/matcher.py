"""Exact batched assignment for at most four targets per image."""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


@lru_cache(maxsize=16)
def _candidate_choices(num_targets: int, device: torch.device) -> Tensor:
    return torch.tensor(
        list(itertools.product(range(num_targets), repeat=num_targets)),
        dtype=torch.long,
        device=device,
    )


def assign_topk_exact(cost: Tensor, valid: Tensor) -> Tensor:
    """Return query indices in target-column order, with -1 for padded columns.

    cost is (problems, queries, targets) and valid is (problems, targets).
    Every real target is considered against its T cheapest queries. At most T-1
    other targets can occupy those T queries, so an optimal assignment is in
    this candidate set. Ties may select a different but equal-cost assignment.
    """
    if cost.ndim != 3 or valid.shape != (cost.shape[0], cost.shape[2]):
        raise ValueError("Expected cost (S,Q,T) and valid (S,T)")
    problems, queries, targets = cost.shape
    if targets > queries:
        raise ValueError("Assignment requires T <= Q")
    if targets == 0:
        return torch.empty((problems, 0), dtype=torch.long, device=cost.device)
    if targets > 4:
        # The dataset has T <= 2. One transfer keeps the general fallback usable.
        host_cost = cost.detach().cpu().numpy()
        host_valid = valid.detach().cpu().numpy()
        output = np.full((problems, targets), -1, dtype=np.int64)
        for problem in range(problems):
            columns = np.flatnonzero(host_valid[problem])
            if columns.size:
                rows, assigned = linear_sum_assignment(host_cost[problem][:, columns])
                output[problem, columns[assigned]] = rows
        return torch.from_numpy(output).to(cost.device)

    if cost.is_cuda and cost.dtype == torch.float32 and valid.dtype == torch.bool:
        try:
            from .cuda_extension import load_cuda_ops

            return load_cuda_ops().matcher_exact(cost.contiguous(), valid.contiguous())
        except (RuntimeError, OSError):
            # Keep the tested Torch path usable if a local nvcc build is absent.
            pass

    padded_cost = cost.masked_fill(~valid[:, None, :], torch.inf)
    candidates = padded_cost.topk(targets, dim=1, largest=False).indices
    choices = _candidate_choices(targets, cost.device)
    combinations = choices.shape[0]
    by_target = candidates.transpose(1, 2)
    chosen = by_target.gather(
        2, choices.T[None, :, :].expand(problems, targets, combinations)
    ).transpose(1, 2)
    chosen_cost = padded_cost.transpose(1, 2).gather(
        2, chosen.transpose(1, 2)
    ).transpose(1, 2)
    score = torch.where(valid[:, None, :], chosen_cost, 0).sum(-1)
    for i in range(targets):
        for j in range(i + 1, targets):
            collision = (chosen[:, :, i] == chosen[:, :, j]) & valid[:, None, i] & valid[:, None, j]
            score = score.masked_fill(collision, torch.inf)
    winner = score.argmin(dim=1)
    selected = chosen.gather(
        1, winner[:, None, None].expand(problems, 1, targets)
    ).squeeze(1)
    return selected.masked_fill(~valid, -1)
