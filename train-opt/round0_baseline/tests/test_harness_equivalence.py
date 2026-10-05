from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.bench import gpu_lock
from common.harness import RunCfg, build, preload_batches, train_step


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_default_harness_matches_lightly_step() -> None:
    torch.cuda.set_per_process_memory_fraction(0.125)
    cfg = RunCfg(model="vitb16", batch_size=1, precision="bf16-mixed", phase="early", total_steps=2, num_workers=0)
    with gpu_lock():
        original = build(cfg)
        batch = preload_batches(original, k=1)[0]
        initial = {key: value.detach().cpu().clone() for key, value in original.model.state_dict().items()}
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()

        reference = original.train_model.training_step(original.fabric, batch, step=0)
        original.fabric.backward(reference.loss)
        ref_norm = original.train_model.clip_gradients(original.fabric, original.optimizer)
        original.optimizer.step()
        original.optimizer.zero_grad()
        original.scheduler.step()
        original.train_model.on_train_batch_end()
        ref_loss = reference.loss.detach().cpu()
        ref_norm = ref_norm.detach().cpu()
        updated = {key: value.detach().cpu().clone() for key, value in original.model.state_dict().items()}
        del original, reference
        gc.collect()
        torch.cuda.empty_cache()

        candidate = build(cfg)
        candidate.model.load_state_dict(initial, strict=True)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        measured = train_step(candidate, batch, step=0)

        assert torch.equal(ref_loss, measured.loss.cpu())
        assert torch.allclose(ref_norm, measured.grad_norm.cpu(), rtol=1e-5, atol=1e-5)
        assert all(torch.equal(value, candidate.model.state_dict()[key].cpu()) for key, value in updated.items())
