"""Numerical gates for training-step, curve, and convergence checks."""
from __future__ import annotations

import gc
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .bench import gpu_lock
from .harness import RunCfg, apply_patches, build, train_step
from .rng import RngTape


def _params(ctx: Any) -> dict[str, torch.Tensor]:
    return {name: parameter for name, parameter in ctx.model.named_parameters() if parameter.requires_grad}


def _cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return 1.0 if a is None and b is None else 0.0
    aa, bb = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    if aa.numel() == 0 and bb.numel() == 0:
        return 1.0
    denom = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    if float(denom) == 0.0:
        return 1.0 if torch.equal(aa, bb) else 0.0
    return float(torch.dot(aa, bb) / denom)


def _rel(a: torch.Tensor | float, b: torch.Tensor | float) -> float:
    av = float(a.detach().float().item()) if isinstance(a, torch.Tensor) and a.ndim == 0 else float(a)
    bv = float(b.detach().float().item()) if isinstance(b, torch.Tensor) and b.ndim == 0 else float(b)
    return abs(av - bv) / max(abs(av), abs(bv), 1e-8)


def _clone_tape(tape: RngTape | None) -> tuple[RngTape, RngTape]:
    if tape is None:
        return RngTape(mode="off"), RngTape(mode="off")
    # A record run becomes the source of a replay run. A supplied replay tape is
    # shared by value to keep both contexts on precisely the same random tensors.
    if tape.mode == "record":
        return tape, RngTape(mode="replay", values={})
    return RngTape(mode=tape.mode, values={k: v.clone() for k, v in tape.values.items()}), RngTape(mode=tape.mode, values={k: v.clone() for k, v in tape.values.items()})


def _capture_step(ctx: Any, batch: Any, tape: RngTape, step: int) -> dict[str, Any]:
    ctx.rng = tape
    parameters = _params(ctx)
    before = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    grads: dict[str, torch.Tensor | None] = {}
    original = ctx.ops.opt_step

    def capture(*args: Any, **kwargs: Any) -> Any:
        grads.update({name: None if parameter.grad is None else parameter.grad.detach().cpu().clone() for name, parameter in parameters.items()})
        return original(*args, **kwargs)

    ctx.ops.opt_step = capture
    try:
        out = train_step(ctx, batch, step)
    finally:
        ctx.ops.opt_step = original
    after = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    updates = {name: after[name] - before[name] for name in before}
    return {"out": out, "grads": grads, "updates": updates}


def step_equivalence(
    cfg_ref: RunCfg,
    cfg_opt: RunCfg,
    batches: Sequence[Any],
    tape: RngTape | None = None,
    *,
    step: int = 0,
) -> dict[str, Any]:
    """Run one update per context and compare loss, every gradient, norm, and update."""
    with gpu_lock():
        return _step_equivalence_locked(cfg_ref, cfg_opt, batches, tape=tape, step=step)


def _step_equivalence_locked(
    cfg_ref: RunCfg,
    cfg_opt: RunCfg,
    batches: Sequence[Any],
    tape: RngTape | None = None,
    *,
    step: int = 0,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("step_equivalence requires at least one batch")
    ref = build(cfg_ref)
    apply_patches(ref)
    # Build sequentially so the large ViT-L reference and candidate never coexist
    # on the GPU. Snapshot before the reference update, including optimizer state.
    initial_state = {key: value.detach().cpu().clone() for key, value in ref.model.state_dict().items()}
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    ref_tape, opt_tape = _clone_tape(tape)
    first = _capture_step(ref, batches[0], ref_tape, step)
    del ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    opt = build(cfg_opt)
    apply_patches(opt)
    opt.model.load_state_dict(initial_state, strict=True)
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    if tape is not None and tape.mode == "record":
        opt_tape.values = {key: value.clone() for key, value in ref_tape.values.items()}
        opt_tape.mode = "replay"
    second = _capture_step(opt, batches[0], opt_tape, step)
    if tape is not None and tape.mode == "record":
        opt_tape.assert_complete()
    graph_state = opt.meta.get("round2_graph_state")
    capture_report = graph_state.report() if graph_state is not None and callable(getattr(graph_state, "report", None)) else None
    del opt

    loss_rel = _rel(first["out"].loss, second["out"].loss)
    grad_cos: dict[str, float] = {}
    update_cos: dict[str, float] = {}
    grad_norm_ref, grad_norm_opt = [], []
    for name in first["grads"]:
        grad_cos[name] = _cosine(first["grads"][name], second["grads"].get(name))
        update_cos[name] = _cosine(first["updates"][name], second["updates"].get(name))
        if first["grads"][name] is not None:
            grad_norm_ref.append(torch.linalg.vector_norm(first["grads"][name].float().reshape(-1)))
        if second["grads"].get(name) is not None:
            grad_norm_opt.append(torch.linalg.vector_norm(second["grads"][name].float().reshape(-1)))
    norm_ref = torch.linalg.vector_norm(torch.stack(grad_norm_ref)) if grad_norm_ref else torch.tensor(0.0)
    norm_opt = torch.linalg.vector_norm(torch.stack(grad_norm_opt)) if grad_norm_opt else torch.tensor(0.0)
    grad_norm_rel = _rel(norm_ref, norm_opt)
    grad_cos_min = min(grad_cos.values(), default=1.0)
    update_cos_min = min(update_cos.values(), default=1.0)
    passed = loss_rel <= 1e-2 and grad_cos_min >= 0.99 and grad_norm_rel <= 0.02 and update_cos_min >= 0.99
    return {
        "loss_rel": loss_rel,
        "grad_cos_min": grad_cos_min,
        "grad_cos": grad_cos,
        "grad_norm_ref": float(norm_ref),
        "grad_norm_opt": float(norm_opt),
        "grad_norm_rel": grad_norm_rel,
        "update_cos_min": update_cos_min,
        "update_cos": update_cos,
        "capture_report": capture_report,
        "pass": passed,
    }


def curve(cfg: RunCfg, batches: Sequence[Any] | None = None, *, steps: int = 300, tape: RngTape | None = None) -> dict[str, Any]:
    """Collect a 300-step loss/gradient curve using the harness training path."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    from .harness import make_loader

    with gpu_lock():
        ctx = build(cfg)
        apply_patches(ctx)
        ctx.rng = tape or RngTape(mode="off")
        loader = None if batches is not None else make_loader(ctx, "train")
        iterator = iter(loader) if loader is not None else None
        losses: list[float] = []
        grad_norms: list[float | None] = []
        ema: list[float] = []
        value = None
        for step in range(steps):
            if batches is None:
                assert iterator is not None and loader is not None
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
            else:
                if not batches:
                    raise ValueError("batches cannot be empty")
                batch = batches[step % len(batches)]
            out = train_step(ctx, batch, step)
            loss = float(out.loss.float().item())
            losses.append(loss)
            grad_norms.append(None if out.grad_norm is None else float(out.grad_norm.float().item()))
            value = loss if value is None else 0.9 * value + 0.1 * loss
            ema.append(value)
    finite = all(math.isfinite(v) for v in losses) and all(v is None or math.isfinite(v) for v in grad_norms)
    return {"steps": steps, "loss": losses, "loss_ema": ema, "grad_norm": grad_norms, "finite": finite}


def read_lightly_metrics(out_dir: str | Path) -> dict[str, float]:
    """Read final mIoU and actual classwise head IoU from Lightly logs."""
    root = Path(out_dir)
    found: dict[str, float] = {}
    if not root.exists():
        raise FileNotFoundError(root)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        for event_path in sorted(root.rglob("events.out.tfevents*")):
            accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0, "tensors": 0})
            accumulator.Reload()
            for tag, key in (("val_metric/miou", "miou"), ("val_metric_classwise/iou_head", "head_iou")):
                if tag in accumulator.Tags().get("scalars", []):
                    events = accumulator.Scalars(tag)
                    if events:
                        found[key] = float(events[-1].value)
                elif tag in accumulator.Tags().get("tensors", []):
                    from tensorboard.util import tensor_util

                    events = accumulator.Tensors(tag)
                    if events:
                        found[key] = float(tensor_util.make_ndarray(events[-1].tensor_proto).item())
    except ImportError:
        pass
    patterns = {
        "miou": re.compile(r"(?:val_metric/miou|Val mIoU)\s*[:=| ]\s*([0-9]+(?:\.[0-9]+)?)", re.I),
        "head_iou": re.compile(r"(?:val_metric_classwise/iou_head)\s*[:=| ]\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    }
    for path in sorted(root.rglob("*")) if root.exists() else []:
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".txt", ".log"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for key, pattern in patterns.items():
            if key in found:
                continue
            matches = pattern.findall(text)
            if matches:
                try:
                    found[key] = float(matches[-1])
                except ValueError:
                    pass
    if "miou" not in found or "head_iou" not in found:
        raise ValueError(f"Missing required Lightly validation metrics {sorted({'miou', 'head_iou'} - found.keys())} in {root}; enable metric_args.classwise")
    return found


def gate_d(metrics: dict[str, float], baseline: dict[str, float], *, threshold: float = 0.95) -> dict[str, Any]:
    for source, values in (("candidate", metrics), ("baseline", baseline)):
        missing = {"miou", "head_iou"} - values.keys()
        if missing:
            raise ValueError(f"{source} is missing D-gate metrics: {sorted(missing)}")
    if baseline["miou"] <= 0 or baseline["head_iou"] <= 0:
        raise ValueError("Baseline D-gate metrics must be positive")
    if not all(math.isfinite(values[key]) for values in (metrics, baseline) for key in ("miou", "head_iou")):
        raise ValueError("D-gate metrics must be finite")
    miou_ratio = metrics.get("miou", float("nan")) / baseline.get("miou", float("nan"))
    head_ratio = metrics.get("head_iou", float("nan")) / baseline.get("head_iou", float("nan"))
    passed = math.isfinite(miou_ratio) and math.isfinite(head_ratio) and miou_ratio >= threshold and head_ratio >= threshold
    return {"val_miou": metrics.get("miou"), "val_head_iou": metrics.get("head_iou"), "ratio_vs_baseline": {"miou": miou_ratio, "head_iou": head_ratio}, "threshold": threshold, "pass": passed}


def gate_c(reference: dict[str, Any], candidate: dict[str, Any], *, interval: int = 50, max_rel: float = 0.05) -> dict[str, Any]:
    """Compare EMA(0.9) at 50-step checkpoints and reject nonfinite curves."""
    ref, opt = reference.get("loss_ema", []), candidate.get("loss_ema", [])
    if len(ref) != len(opt) or not ref or interval <= 0:
        raise ValueError("C-gate curves must have the same nonzero length")
    checkpoints = list(range(interval - 1, len(ref), interval))
    deviations = {str(i + 1): _rel(ref[i], opt[i]) for i in checkpoints}
    finite = all(math.isfinite(float(value)) for curve_values in (reference.get("loss", []), candidate.get("loss", []), reference.get("grad_norm", []), candidate.get("grad_norm", [])) for value in curve_values if value is not None)
    norms_ref, norms_opt = reference.get("grad_norm", []), candidate.get("grad_norm", [])
    norm_ratios: dict[str, float | None] = {}
    for i in checkpoints:
        left = norms_ref[i] if i < len(norms_ref) else None
        right = norms_opt[i] if i < len(norms_opt) else None
        norm_ratios[str(i + 1)] = (float(right) / float(left)) if left is not None and right is not None and float(left) > 0 else None
    grad_norm_ok = all(value is not None and math.isfinite(value) and 0.1 <= value <= 10.0 for value in norm_ratios.values())
    worst = max(deviations.values(), default=float("inf"))
    return {"curve300_max_rel": worst, "checkpoints": deviations, "grad_norm_ratios": norm_ratios, "grad_norm_same_order": grad_norm_ok, "finite": finite, "pass": finite and worst <= max_rel and grad_norm_ok}
