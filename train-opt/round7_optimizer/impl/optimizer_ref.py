"""Small fp32 AdamW + global L2 clip reference for optimizer kernel checks."""

from __future__ import annotations

import torch
from torch import Tensor


def clipped_adamw_reference_step(
    groups: list[dict],
    state: dict[Tensor, dict[str, Tensor]],
    max_norm: float,
) -> Tensor:
    """Update parameters in place, using the original param-group hyperparameters."""
    gradients = [
        param.grad for group in groups for param in group["params"]
        if param.grad is not None
    ]
    if gradients:
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(grad.detach(), ord=2) for grad in gradients
        ]), ord=2)
    else:
        norm = torch.tensor(0.0)
    clip = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    with torch.no_grad():
        for group in groups:
            beta1, beta2 = group.get("betas", (0.9, 0.999))
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)
            eps = group.get("eps", 1e-8)
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad * clip
                item = state.setdefault(param, {})
                if not item:
                    item["step"] = torch.tensor(0.0)
                    item["exp_avg"] = torch.zeros_like(param)
                    item["exp_avg_sq"] = torch.zeros_like(param)
                item["step"].add_(1)
                step = int(item["step"].item())
                param.mul_(1 - lr * wd)
                item["exp_avg"].lerp_(grad, 1 - beta1)
                item["exp_avg_sq"].mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                )
                correction1 = 1 - beta1 ** step
                correction2 = 1 - beta2 ** step
                denominator = item["exp_avg_sq"].sqrt().div_(
                    correction2 ** 0.5
                ).add_(eps)
                param.addcdiv_(
                    item["exp_avg"], denominator,
                    value=-lr / correction1,
                )
    return norm
