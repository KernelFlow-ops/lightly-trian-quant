"""Runtime injection for the original Lightly loop's supported round1 ops."""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Iterator

from common.fast_forward import forward_train_v2
from common.ops import default_ops
from common.rng import RngTape
from round1_torch_cleanup.impl.patch import _components, _install_adamw_fused, apply as apply_round1


SUPPORTED = {"nosync", "mask_skip", "rope_batch", "sdpa_flash", "sdpa_cudnn", "sdpa_efficient", "sdpa_math", "adamw_fused", "cudnn_bench", "head_cl", "kmask_bias", "val_fast", "loader"}


@contextmanager
def patch_lightly_loop(*, r1_variant: str, precision: str) -> Iterator[None]:
    """Install actual round1 behavior into Lightly's own training loop."""
    from lightly_train._task_models.dinov3_eomt_semantic_segmentation.train_model import DINOv3EoMTSemanticSegmentationTrain as Train
    import torch

    components = _components(r1_variant)
    unsupported = set(components) - SUPPORTED
    if unsupported:
        raise ValueError(f"The Lightly loop cannot apply these round1 components: {sorted(unsupported)}")
    if not components:
        yield
        return
    forward_components = [component for component in components if component not in {"adamw_fused", "loader", "val_fast"}]
    old_step = Train.training_step
    old_optimizer = Train.get_optimizer
    old_validation = Train.validation_step

    def get_optimizer(self, *args, **kwargs):
        optimizer, scheduler = old_optimizer(self, *args, **kwargs)
        if "adamw_fused" in components:
            _install_adamw_fused(SimpleNamespace(optimizer=optimizer))
        return optimizer, scheduler

    def training_step(self, fabric, batch, step):
        if forward_components and not hasattr(self, "_train_opt_round1_ctx"):
            ctx = SimpleNamespace(
                model=self.model, train_model=self, ops=default_ops(), rng=RngTape(),
                host_probs=[float(value) for value in self.model.attn_mask_probs.detach().cpu()],
                meta={}, cfg=SimpleNamespace(precision=precision), nvtx=False,
            )
            apply_round1(ctx, variant="+".join(forward_components))

            def forward(model_self, images, return_logits_per_layer):
                return forward_train_v2(ctx, images, return_logits_per_layer)

            self.model.forward_train = MethodType(forward, self.model)
            self._train_opt_round1_ctx = ctx
        return old_step(self, fabric=fabric, batch=batch, step=step)

    def validation_step(self, fabric, batch, step):
        if "val_fast" in components:
            with torch.inference_mode():
                return old_validation(self, fabric=fabric, batch=batch, step=step)
        return old_validation(self, fabric=fabric, batch=batch, step=step)

    Train.get_optimizer = get_optimizer
    Train.training_step = training_step
    Train.validation_step = validation_step
    try:
        yield
    finally:
        Train.get_optimizer = old_optimizer
        Train.training_step = old_step
        Train.validation_step = old_validation
