"""Register training-capable head variants without replacing parameters."""

from __future__ import annotations

from .fused_head import predict

NAME = "round6_head_fusion"
REQUIRES: tuple[str, ...] = ()
CONFLICTS: tuple[str, ...] = ()


def apply(ctx, variant: str = "channels_last_bmm", **kw) -> dict:
    if kw:
        raise ValueError(f"Unknown round6 options: {sorted(kw)}")
    if variant not in ("bmm", "channels_last_bmm", "convt_gemm", "dwln", "all", "fused_dwln"):
        raise ValueError(f"Unknown head variant: {variant}")
    installed = ctx.meta.get(NAME)
    if installed is not None:
        if installed != variant:
            raise RuntimeError("round6 already installed with a different variant")
        return {"ops": ["predict"], "desc": f"{NAME}:{variant} (already installed)"}

    def predict_op(ctx, x, grid_size):
        return predict(ctx.model, x, grid_size, variant)

    ctx.ops.predict = predict_op
    ctx.meta[NAME] = variant
    return {"ops": ["predict"], "desc": f"{NAME}:{variant}"}
