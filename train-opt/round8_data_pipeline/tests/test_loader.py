from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from lightning_fabric import Fabric
from torch.utils.data import TensorDataset

from lightly_train._commands import train_task_helpers as helpers
from round8_data_pipeline.impl.loader import LoaderOptions, loader_args
from round8_data_pipeline.impl.patch import apply


def test_loader_options_validate_worker_specific_flags():
    assert LoaderOptions(workers=0).args() == {"pin_memory": True}
    assert LoaderOptions(workers=16, prefetch_factor=4, pin_memory=False, persistent_workers=True).args() == {"pin_memory": False, "prefetch_factor": 4, "persistent_workers": True}
    with pytest.raises(ValueError, match="requires workers"):
        LoaderOptions(workers=0, persistent_workers=True).args()
    with pytest.raises(ValueError, match="positive"):
        LoaderOptions(workers=4, prefetch_factor=0).args()


def test_lightly_disables_persistent_workers_when_transform_is_step_aware():
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    dataset = TensorDataset(torch.arange(8))
    loader = helpers.get_train_dataloader(
        fabric=fabric, dataset=dataset, batch_size=4, num_workers=2,
        loader_args=LoaderOptions(workers=2, prefetch_factor=4, persistent_workers=True).args(),
    )
    assert loader.prefetch_factor == 4 and loader.persistent_workers
    aware = SimpleNamespace(uses_step_dependent_worker_state=lambda: True)
    unaware = SimpleNamespace(uses_step_dependent_worker_state=lambda: False)
    adjusted = helpers.get_train_dataloader(
        fabric=fabric, dataset=dataset, batch_size=4, num_workers=2,
        loader_args=loader_args(LoaderOptions(2, persistent_workers=True), aware, unaware),
    )
    assert adjusted.persistent_workers is False


def test_patch_registers_loader_factory_without_model_changes():
    cfg = SimpleNamespace(num_workers=4)
    ctx = SimpleNamespace(cfg=cfg, meta={})
    result = apply(ctx, workers=16, prefetch_factor=4)
    assert result["ops"] == []
    assert cfg.num_workers == 16
    assert ctx.meta["loader_options"] == LoaderOptions(16, 4, True, False)
    factory = ctx.meta["loader_factory"]
    apply(ctx, workers=16, prefetch_factor=4)
    assert ctx.meta["loader_factory"] is factory
