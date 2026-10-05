"""Round2 patch entrypoint: segmented fwd+bwd graph or compile comparison."""
from __future__ import annotations

from .segmented import install

NAME = "round2_cuda_graph"
REQUIRES = ("round1_torch_cleanup",)
CONFLICTS: tuple[str, ...] = ()


def apply(ctx, variant: str = "segmented", **kwargs):
    max_signatures = int(kwargs.pop("max_joint_signatures", 7))
    if kwargs:
        raise ValueError(f"Unsupported round2 options: {sorted(kwargs)}")
    if variant == "segmented" and "nosync" not in ctx.meta.get("round1_components", ()):
        raise RuntimeError("round2 segmented graph requires round1 nosync to remove tensor-to-host mask branches")
    state = install(ctx, variant=variant, max_joint_signatures=max_signatures)
    ctx.meta["round2_variant"] = variant
    return {"ops": ["forward_train"], "desc": f"{variant}; graph capture deferred to first real batch", "captured": False}
