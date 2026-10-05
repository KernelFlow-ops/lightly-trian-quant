from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.rng import RngTape


def test_replay_preserves_later_global_rng() -> None:
    torch.manual_seed(17)
    recorded = RngTape(mode="record")
    original = recorded.rand("mask/j1", (3, 5), device="cpu")
    after_record = torch.rand(4)

    torch.manual_seed(17)
    replayed = RngTape(mode="replay", values=recorded.values)
    restored = replayed.rand("mask/j1", (3, 5), device="cpu")
    after_replay = torch.rand(4)

    assert torch.equal(original, restored)
    assert torch.equal(after_record, after_replay)
    replayed.assert_complete()
