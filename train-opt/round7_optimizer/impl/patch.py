"""Capturable AdamW fallback with checkpoint-compatible serialization."""

from __future__ import annotations

import re
from typing import Any

import torch

from .fused_optimizer import fused_clip_grad_norm_

NAME = "round7_optimizer"
REQUIRES: tuple[str, ...] = ()
CONFLICTS: tuple[str, ...] = ()
DEFAULT_SHADOW_PATTERNS = (
    r"model\.backbone\.blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)\.weight",
)


def _underlying_optimizer(optimizer: Any) -> Any:
    return getattr(optimizer, "optimizer", optimizer)


def _install_checkpoint_compatibility(ctx, optimizer, device_lrs, old_settings) -> None:
    original_state_dict = optimizer.state_dict
    original_load_state_dict = optimizer.load_state_dict

    def state_dict():
        raw = original_state_dict()
        exported = {**raw, "param_groups": [dict(g) for g in raw["param_groups"]],
                    "state": {k: dict(v) for k, v in raw["state"].items()}}
        for group, old in zip(exported["param_groups"], old_settings):
            group["lr"] = float(group["lr"].detach().cpu())
            for key in ("fused", "capturable", "foreach"):
                group[key] = old[key]
        for item in exported["state"].values():
            if "step" in item:
                item["step"] = item["step"].detach().cpu()
        return exported

    def load_state_dict(saved):
        adjusted = dict(saved)
        adjusted["param_groups"] = [
            {**group, "fused": True, "capturable": True, "foreach": None}
            for group in saved["param_groups"]
        ]
        original_load_state_dict(adjusted)
        for group, lr_tensor in zip(optimizer.param_groups, device_lrs):
            loaded_lr = group["lr"]
            with torch.no_grad():
                lr_tensor.fill_(
                    float(loaded_lr.detach().cpu())
                    if isinstance(loaded_lr, torch.Tensor) else float(loaded_lr)
                )
            group["lr"] = lr_tensor

    optimizer.state_dict = state_dict
    optimizer.load_state_dict = load_state_dict

    original_scheduler_state_dict = ctx.scheduler.state_dict

    def scheduler_state_dict():
        exported = original_scheduler_state_dict()
        exported["_last_lr"] = [
            float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value
            for value in exported.get("_last_lr", [])
        ]
        return exported

    ctx.scheduler.state_dict = scheduler_state_dict


def _shadow_tensors(ctx) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    patterns = ctx.meta.get("shadow_params", DEFAULT_SHADOW_PATTERNS)
    masters, shadows = [], []
    for name, param in ctx.train_model.named_parameters():
        if not any(re.fullmatch(pattern, name) for pattern in patterns):
            continue
        if name not in ctx.shadow:
            ctx.shadow[name] = param.detach().to(torch.bfloat16).contiguous()
        shadow = ctx.shadow[name]
        if shadow.requires_grad or shadow.grad_fn is not None:
            raise ValueError(f"Shadow tensor must not track autograd: {name}")
        masters.append(param.detach())
        shadows.append(shadow)
    if not shadows:
        raise ValueError("No parameters matched the shadow-weight protocol")
    return masters, shadows


def _ensure_state(optimizer: Any, params: list[torch.Tensor]) -> None:
    """Materialize Adam state with graph-safe device step tensors."""
    for param in params:
        item = optimizer.state[param]
        if "step" not in item:
            item["step"] = torch.zeros((), device=param.device, dtype=torch.float32)
        elif not isinstance(item["step"], torch.Tensor) or item["step"].device != param.device:
            item["step"] = torch.as_tensor(item["step"], device=param.device, dtype=torch.float32)
        if "exp_avg" not in item:
            item["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format, dtype=torch.float32)
        if "exp_avg_sq" not in item:
            item["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format, dtype=torch.float32)



def apply(ctx, variant: str = "capturable_torch", **kw) -> dict:
    if kw:
        raise ValueError(f"Unknown round7 options: {sorted(kw)}")
    if variant == "cuda_fused":
        return _apply_cuda_fused(ctx)
    if variant not in ("capturable_torch", "capturable_shadow", "cuda_multi_tensor", "multi_tensor"):
        raise ValueError(f"Unknown round7 variant: {variant}")
    installed = ctx.meta.get(NAME)
    if installed is not None:
        if installed != variant:
            raise RuntimeError("round7 already installed with a different variant")
        return {"ops": ["opt_step"], "desc": f"{NAME}:{variant} (already installed)"}

    optimizer = _underlying_optimizer(ctx.optimizer)
    if any(group.get("amsgrad") for group in optimizer.param_groups):
        raise ValueError("round7 capturable fallback does not support AMSGrad")
    old_settings = [
        {key: group.get(key) for key in ("fused", "capturable", "foreach")}
        for group in optimizer.param_groups
    ]
    device_lrs = []
    for group in optimizer.param_groups:
        device = group["params"][0].device
        lr_tensor = torch.as_tensor(group["lr"], dtype=torch.float32, device=device).clone()
        group["lr"] = lr_tensor
        group["fused"] = True
        group["capturable"] = True
        group["foreach"] = None
        device_lrs.append(lr_tensor)
    for param, item in optimizer.state.items():
        if "step" in item:
            item["step"] = item["step"].to(device=param.device, dtype=torch.float32)
    _install_checkpoint_compatibility(ctx, optimizer, device_lrs, old_settings)

    masters: list[torch.Tensor] = []
    shadows: list[torch.Tensor] = []
    if variant == "capturable_shadow":
        masters, shadows = _shadow_tensors(ctx)
        ctx.meta["shadow_writer"] = NAME

    if variant in ("multi_tensor", "cuda_multi_tensor"):
        from .multi_tensor import MultiTensorAdamW
        backend = MultiTensorAdamW(optimizer)
        @torch.no_grad()
        def cuda_opt_step(c):
            norm = backend.step(float(c.train_model.model_args.gradient_clip_val))
            if not c.meta.get("round7_graph_capture", False):
                c.scheduler.step()
            return norm
        ctx.ops.opt_step = cuda_opt_step
        ctx.meta["round7_multi_tensor_backend"] = backend
        ctx.meta[NAME] = variant
        ctx.meta["round7_device_lrs"] = device_lrs
        ctx.meta["round7_scheduler_after_graph"] = ctx.scheduler.step
        ctx.meta["round7_graph_capture"] = False
        return {"ops": ["opt_step"], "desc": f"{NAME}:three-launch CUDA AdamW + clip"}

    def opt_step(ctx):
        clip_value = float(ctx.train_model.model_args.gradient_clip_val)
        if clip_value > 0:
            grad_norm = fused_clip_grad_norm_(ctx.train_model.parameters(), clip_value)
        else:
            grad_norm = ctx.train_model.clip_gradients(ctx.fabric, ctx.optimizer)
        ctx.optimizer.step()
        ctx.optimizer.zero_grad(set_to_none=False)
        if shadows:
            with torch.no_grad():
                torch._foreach_copy_(shadows, masters)
        if not ctx.meta.get("round7_graph_capture", False):
            ctx.scheduler.step()
        return grad_norm

    ctx.ops.opt_step = opt_step
    ctx.meta[NAME] = variant
    ctx.meta["round7_device_lrs"] = device_lrs
    ctx.meta["round7_scheduler_after_graph"] = ctx.scheduler.step
    ctx.meta["round7_graph_capture"] = False
    return {"ops": ["opt_step"], "desc": f"{NAME}:{variant}"}


def _apply_cuda_fused(ctx) -> dict:
    """Install the correctness-first CUDA C++ AdamW kernel.

    This variant deliberately keeps scheduler updates outside the kernel and
    uses one CUDA launch per parameter for the first implementation. It is a
    real CUDA path; launch fusion and flattened multi-tensor storage are later
    tuning steps.
    """
    from .cuda_adamw import fused_adamw_step

    optimizer = _underlying_optimizer(ctx.optimizer)
    if any(group.get("amsgrad") for group in optimizer.param_groups):
        raise ValueError("cuda_fused does not support AMSGrad")
    state = ctx.meta.setdefault("cuda_adamw_state", {})
    device_lrs: list[torch.Tensor] = []
    param_groups: list[tuple[Any, float, float]] = []
    for group in optimizer.param_groups:
        lr = torch.as_tensor(float(group["lr"]), device=ctx.model.cls_token.device, dtype=torch.float32)
        wd = torch.as_tensor(float(group.get("weight_decay", optimizer.defaults.get("weight_decay", 0.0))), device=lr.device, dtype=torch.float32)
        device_lrs.append(lr)
        for param in group["params"]:
            param_groups.append((param, float(group["lr"]), float(group.get("weight_decay", optimizer.defaults.get("weight_decay", 0.0)))))
    ctx.meta["round7_device_lrs"] = device_lrs
    ctx.meta["round7_cuda_extension"] = "trainopt_cuda_adamw"

    def opt_step(c):
        params: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        exp_avg: list[torch.Tensor] = []
        exp_avg_sq: list[torch.Tensor] = []
        steps: list[torch.Tensor] = []
        lrs: list[torch.Tensor] = []
        wds: list[torch.Tensor] = []
        for group_index, group in enumerate(optimizer.param_groups):
            for param in group["params"]:
                if param.grad is None:
                    continue
                item = optimizer.state[param]
                if "exp_avg" not in item:
                    item["step"] = torch.zeros((), device=param.device, dtype=torch.float32)
                    item["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format, dtype=torch.float32)
                    item["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format, dtype=torch.float32)
                item["step"] = item["step"].to(device=param.device, dtype=torch.float32)
                params.append(param)
                grads.append(param.grad)
                exp_avg.append(item["exp_avg"])
                exp_avg_sq.append(item["exp_avg_sq"])
                steps.append(item["step"])
                lrs.append(device_lrs[group_index])
                wds.append(torch.as_tensor(float(group.get("weight_decay", optimizer.defaults.get("weight_decay", 0.0))), device=param.device, dtype=torch.float32))
        grad_norm = fused_adamw_step(
            params, grads, exp_avg, exp_avg_sq, steps, lrs, wds, [],
            float(c.train_model.model_args.gradient_clip_val),
            float(optimizer.defaults.get("betas", (0.9, 0.999))[0]),
            float(optimizer.defaults.get("betas", (0.9, 0.999))[1]),
            float(optimizer.defaults.get("eps", 1e-8)),
        )
        for param in optimizer.param_groups[0]["params"] if False else []:
            pass
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()
        if not c.meta.get("round7_graph_capture", False):
            c.scheduler.step()
        return grad_norm

    ctx.ops.opt_step = opt_step
    ctx.meta[NAME] = "cuda_fused"
    ctx.meta["round7_graph_capture"] = False
    ctx.meta["round7_scheduler_after_graph"] = ctx.scheduler.step
    return {"ops": ["opt_step"], "desc": "round7 CUDA C++ fused AdamW + clip baseline"}
