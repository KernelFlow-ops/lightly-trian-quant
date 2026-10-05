from __future__ import annotations

from types import SimpleNamespace
import json
import tempfile
from pathlib import Path

import pytest
import torch
from lightning_fabric import Fabric
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lightly_train._task_models.dinov3_eomt_semantic_segmentation.train_model import DINOv3EoMTSemanticSegmentationTrain as Train
from common.check import read_lightly_metrics
from round10_integration.train import _patches
from round10_integration.impl.fast_loop import _advance_transform, _checkpoint_state, _export_dict, _save
from round10_integration.impl import fast_loop
from round10_integration.impl.lightly_patch import patch_lightly_loop
from round10_integration.verify_export import _state_shapes
from round10_integration.compare import _plot_curves, _write_average_curves, compare


def test_export_dict_matches_lightly_contract():
    task = SimpleNamespace(class_path="package.Task", init_args={"model_name": "test"})
    train = SimpleNamespace(get_task_model=lambda: task, get_export_state_dict=lambda: {"model.weight": 1})
    ctx = SimpleNamespace(train_model=train, optimizer=object(), scheduler=object())
    loader = object()
    state = _checkpoint_state(ctx, loader)
    assert set(state) == {"train_model", "optimizer", "scheduler", "train_dataloader", "step", "model_class_path", "model_init_args", "license_info"}
    assert state["train_dataloader"] is loader
    exported = _export_dict(ctx, state)
    assert set(exported) == {"model_class_path", "model_init_args", "train_model", "license_info"}
    assert exported["train_model"] == {"model.weight": 1}


def test_lightly_helpers_write_same_checkpoint_and_export_keys_on_cpu():
    class TinyTrain(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(2, 1)

        def forward(self, x):
            return self.model(x)

        def get_task_model(self):
            return SimpleNamespace(class_path="pkg.Tiny", init_args={"model_name": "tiny"})

        def get_export_state_dict(self):
            return self.state_dict()

    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    train = TinyTrain()
    optimizer = torch.optim.AdamW(train.parameters())
    train, optimizer = fabric.setup(train, optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loader = fabric.setup_dataloaders(DataLoader(TensorDataset(torch.randn(2, 2)), batch_size=1))
    ctx = SimpleNamespace(fabric=fabric, train_model=train, optimizer=optimizer, scheduler=scheduler)
    state = _checkpoint_state(ctx, loader)
    state["step"] = 0
    with tempfile.TemporaryDirectory(dir="/root/autodl-tmp/train-opt-tmp/A") as directory:
        root = Path(directory)
        _save(ctx, root, state, "last")
        checkpoint = torch.load(root / "checkpoints" / "last.ckpt", weights_only=False)
        exported = torch.load(root / "exported_models" / "exported_last.pt", weights_only=False)
        assert set(checkpoint) == set(state)
        assert set(exported) == {"model_class_path", "model_init_args", "train_model", "license_info"}


def test_lightly_runtime_patch_is_scoped_and_rejects_unsupported_components():
    original_step = Train.training_step
    original_optimizer = Train.get_optimizer
    with patch_lightly_loop(r1_variant="nosync", precision="bf16-mixed"):
        assert Train.training_step is not original_step
        assert Train.get_optimizer is not original_optimizer
    assert Train.training_step is original_step
    assert Train.get_optimizer is original_optimizer
    with pytest.raises(ValueError, match="cannot apply"):
        with patch_lightly_loop(r1_variant="ckpt_async", precision="bf16-mixed"):
            pass


def test_explicit_opt_stack_keeps_variant_options():
    args = SimpleNamespace(
        opt=True,
        patches="round1_torch_cleanup,round4_attention",
        patch_kwargs='{"round1_torch_cleanup":{"variant":"nosync+mask_skip"}}',
    )
    patches, kwargs = _patches(args)
    assert patches == ("round1_torch_cleanup", "round4_attention")
    assert kwargs["round1_torch_cleanup"]["variant"] == "nosync+mask_skip"
    args.patches = "round4_attention,round1_torch_cleanup"
    with pytest.raises(ValueError, match="standard combination order"):
        _patches(args)


def test_p0_candidate_has_deterministic_default_variants():
    args = SimpleNamespace(opt=True, patches="", patch_kwargs="{}", preset="p0")
    patches, kwargs = _patches(args)
    assert patches == (
        "round1_torch_cleanup", "round3_loss_matcher", "round4_attention",
        "round5_block_fusion", "round7_optimizer",
    )
    assert kwargs["round3_loss_matcher"]["variant"] == "batched_torch"
    assert kwargs["round4_attention"]["variant"] == "sdpa_rope"
    assert kwargs["round7_optimizer"]["variant"] == "capturable_torch"


def test_fast_validation_log_is_readable_by_d_gate():
    with tempfile.TemporaryDirectory(dir="/root/autodl-tmp/train-opt-tmp/A") as directory:
        Path(directory, "val_metrics.log").write_text(
            "epoch: 1\nval_metric/miou: 0.6\nval_metric_classwise/iou_head: 0.4\n"
        )
        assert read_lightly_metrics(directory) == {"miou": 0.6, "head_iou": 0.4}


def test_export_metadata_reader_uses_state_shapes():
    with tempfile.TemporaryDirectory(dir="/root/autodl-tmp/train-opt-tmp/A") as directory:
        path = Path(directory, "export.pt")
        torch.save({"model_class_path": "pkg.Model", "model_init_args": {"model_name": "vitb16"}, "train_model": {"weight": torch.ones(2, 3)}, "license_info": "notice"}, path)
        shapes, meta = _state_shapes(path)
        assert shapes == {"weight": (2, 3)}
        assert meta == {"model_class_path": "pkg.Model", "model_name": "vitb16"}


def test_fast_epoch_updates_steps_validates_and_saves_best_last(monkeypatch):
    class Transform:
        def __init__(self):
            self.steps = []

        def set_step(self, step):
            self.steps.append(step)

        def requires_dataloader_reinitialization(self):
            return False

    class Loader:
        def __init__(self):
            self.dataset = SimpleNamespace(transform=Transform())
            self.collate_fn = Transform()

        def __len__(self):
            return 2

        def __iter__(self):
            for _ in range(2):
                yield {"image": torch.zeros(1)}

    class Metrics:
        def __init__(self, split):
            self.split = split

        def compute_aggregated_values(self):
            values = {"train_loss": 1.0} if self.split == "train" else {"val_metric/miou": 0.6, "val_metric_classwise/iou_head": 0.4}
            return SimpleNamespace(metric_values=values, watch_metric="val_metric/miou" if self.split == "val" else None, watch_metric_value=0.6 if self.split == "val" else None, watch_metric_mode="max" if self.split == "val" else None)

        def reset(self):
            pass

    class TrainModel:
        def __init__(self):
            self.train_metrics = Metrics("train")
            self.val_metrics = Metrics("val")
            self.metric_args = SimpleNamespace(watch_metric="val_metric/miou")

        def get_task_model(self):
            return SimpleNamespace(class_path="pkg.Model", init_args={"model_name": "test"})

        def get_export_state_dict(self):
            return {"weight": torch.ones(1)}

        def eval(self):
            pass

        def set_train_mode(self):
            pass

        def validation_step(self, fabric, batch, step):
            return SimpleNamespace(metrics=self.val_metrics)

    train_loader, val_loader = Loader(), Loader()
    monkeypatch.setattr(fast_loop, "make_loader", lambda ctx, split: train_loader if split == "train" else val_loader)
    steps = []
    monkeypatch.setattr(fast_loop, "train_step", lambda ctx, batch, step: steps.append(step))
    saves = []
    monkeypatch.setattr(fast_loop.helpers, "save_checkpoint", lambda **kw: saves.append((kw["best_or_last"], kw["state"]["step"])))
    monkeypatch.setattr(fast_loop.helpers, "export_model", lambda **kw: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    ctx = SimpleNamespace(
        train_model=TrainModel(), optimizer=object(), scheduler=object(), fabric=object(),
        meta={"total_steps": 2}, cfg=SimpleNamespace(num_workers=0),
    )
    with tempfile.TemporaryDirectory(dir="/root/autodl-tmp/train-opt-tmp/A") as directory:
        history = fast_loop.run_fast(ctx, Path(directory), epochs=1)
        assert steps == [0, 1]
        assert train_loader.dataset.transform.steps == [0, 1]
        assert saves == [("last", 1), ("best", 1)]
        assert history[0]["val_head_iou"] == 0.4
        assert read_lightly_metrics(directory) == {"miou": 0.6, "head_iou": 0.4}


def test_fast_step_aware_loader_resets_workers_at_transition():
    class StepAware:
        def __init__(self):
            self.step = None
            self.marked = False

        def set_step(self, step):
            self.step = step

        def requires_dataloader_reinitialization(self):
            return not self.marked

        def mark_dataloader_as_reinitialized(self):
            self.marked = True

    transform, collate = StepAware(), StepAware()
    loader = SimpleNamespace(dataset=SimpleNamespace(transform=transform), collate_fn=collate)
    iterator = SimpleNamespace(resets=0)
    iterator.reset = lambda: setattr(iterator, "resets", iterator.resets + 1)
    _advance_transform(loader, iterator, 10, workers=2)
    assert iterator.resets == 1
    assert transform.step == collate.step == 10
    assert transform.marked and collate.marked


def test_two_seed_final_gate_uses_matched_budget_and_real_metrics():
    with tempfile.TemporaryDirectory(dir="/root/autodl-tmp/train-opt-tmp/A") as directory:
        root = Path(directory)
        baseline, final = [], []
        for seed, (base_miou, base_head, final_miou, final_head) in enumerate(((0.60, 0.40, 0.58, 0.39), (0.62, 0.42, 0.60, 0.41))):
            base_dir, final_dir = root / f"base{seed}", root / f"final{seed}"
            base_dir.mkdir()
            final_dir.mkdir()
            (base_dir / "train_opt_config.json").write_text(json.dumps({"model": "vitb16", "batch_size": 14, "epochs": 20, "seed": seed}))
            (final_dir / "train_opt_run.json").write_text(json.dumps({"model": "vitb16", "batch": 14, "epochs": 20, "seed": seed, "patches": ["round1_torch_cleanup"], "patch_kwargs": {"round1_torch_cleanup": {"variant": "nosync"}}}))
            (base_dir / "val_metrics.log").write_text(f"val_metric/miou: {base_miou}\nval_metric_classwise/iou_head: {base_head}\n")
            (final_dir / "val_metrics.log").write_text(f"val_metric/miou: {final_miou}\nval_metric_classwise/iou_head: {final_head}\n")
            for directory_path, miou, head in ((base_dir, base_miou, base_head), (final_dir, final_miou, final_head)):
                (directory_path / "metrics.jsonl").write_text(json.dumps({"val_miou": miou, "val_head_iou": head, "train_metrics": {"train_loss": 0.5}}) + "\n")
            baseline.append(base_dir)
            final.append(final_dir)
        result = compare("vitb16", baseline, final, epochs=20)
        assert result["gate"]["pass"] is True
        assert result["baseline_mean"]["miou"] == pytest.approx(0.61)
        assert result["baseline_mean"]["head_iou"] == pytest.approx(0.41)
        assert _plot_curves(result, root / "curves.png") == root / "curves.png"
        assert (root / "curves.png").is_file()
        paths = _write_average_curves(result, root / "compare.json")
        assert len(paths) == 4 and all(path.is_file() for path in paths)
        with pytest.raises(ValueError, match="Budget mismatch"):
            compare("vitb16", baseline, final, epochs=6)
