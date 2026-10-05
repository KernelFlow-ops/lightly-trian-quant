"""Round8 loader-only patch; cache/GPU augmentation require later evidence."""
from __future__ import annotations

from .loader import LoaderOptions, make_loader

NAME = "round8_data_pipeline"
REQUIRES: tuple[str, ...] = ()
CONFLICTS: tuple[str, ...] = ()


def apply(ctx, variant: str = "loader", *, workers: int | None = None, prefetch_factor: int = 2, pin_memory: bool = True, persistent_workers: bool = False, **kwargs) -> dict:
    if kwargs:
        raise ValueError(f"Unsupported round8 options: {sorted(kwargs)}")
    if variant != "loader":
        raise ValueError("Only the measured loader variant is available; decoder cache/GPU augmentation are not implemented")
    options = LoaderOptions(
        workers=ctx.cfg.num_workers if workers is None else workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    options.args()
    if ctx.meta.get(NAME) == options:
        return {"ops": [], "desc": f"loader {options}"}
    ctx.cfg.num_workers = options.workers
    ctx.meta["loader_factory"] = lambda c, split: make_loader(c, split, options)[0]
    ctx.meta["loader_options"] = options
    ctx.meta[NAME] = options
    return {"ops": [], "desc": f"loader {options}"}
