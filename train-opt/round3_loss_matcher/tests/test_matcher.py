from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round3_loss_matcher.impl.matcher import assign_topk_exact


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ten_thousand_assignments_match_scipy() -> None:
    torch.cuda.set_per_process_memory_fraction(0.125)
    rng = np.random.default_rng(51)
    problems, queries, max_targets = 10_000, 100, 4
    cost = rng.normal(size=(problems, queries, max_targets)).astype(np.float32)
    lengths = rng.integers(0, max_targets + 1, size=problems)
    # Include many exact ties, but keep most cases continuous for index checks.
    cost[::7] = np.round(cost[::7], 1)
    valid = np.arange(max_targets)[None, :] < lengths[:, None]
    chosen = assign_topk_exact(
        torch.from_numpy(cost).cuda(), torch.from_numpy(valid).cuda()
    ).cpu().numpy()

    for n in range(problems):
        t = lengths[n]
        if t == 0:
            assert np.all(chosen[n] == -1)
            continue
        rows, cols = linear_sum_assignment(cost[n, :, :t])
        scipy_by_target = rows[np.argsort(cols)]
        got = chosen[n, :t]
        assert len(set(got)) == t
        got_cost = cost[n, got, np.arange(t)].sum(dtype=np.float64)
        scipy_cost = cost[n, scipy_by_target, np.arange(t)].sum(dtype=np.float64)
        assert abs(got_cost - scipy_cost) < 1e-6, n
        if n % 7:
            assert np.array_equal(got, scipy_by_target), n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_more_than_four_targets_uses_exact_fallback() -> None:
    torch.cuda.set_per_process_memory_fraction(0.125)
    rng = np.random.default_rng(53)
    cost = rng.normal(size=(32, 100, 6)).astype(np.float32)
    lengths = rng.integers(0, 7, size=32)
    valid = np.arange(6)[None, :] < lengths[:, None]
    chosen = assign_topk_exact(
        torch.from_numpy(cost).cuda(), torch.from_numpy(valid).cuda()
    ).cpu().numpy()
    for n, length in enumerate(lengths):
        if not length:
            assert np.all(chosen[n] == -1)
            continue
        rows, cols = linear_sum_assignment(cost[n, :, :length])
        expected = rows[np.argsort(cols)]
        assert np.array_equal(chosen[n, :length], expected)
