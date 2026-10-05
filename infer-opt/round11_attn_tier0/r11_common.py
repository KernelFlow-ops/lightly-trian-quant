"""Round 11 shared helpers: sys.path setup, variant registry (r5_v1 baseline + r11 variants), pipeline construction.

Import this first in every round-11 script.  Old rounds are only imported (read-only).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # /root/lightly-train/infer-opt
TMP = Path("/root/autodl-tmp/infer-opt-tmp/round11_attn_tier0")
TMP.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("LIGHTLY_TRAIN_EVENTS_DISABLED", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
for _p in ("/root/miniconda3/bin", "/usr/local/cuda/bin"):
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + ":" + os.environ.get("PATH", "")

# order matters: round9 video_infer first (bench9 does the same), round8 helpers appended
for _p in (ROOT, ROOT / "round9_integration", ROOT / "common", ROOT / "round7_head_postprocess" / "impl",
           ROOT / "round5_gemm_sm120", HERE, HERE / "impl"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(ROOT / "round8_pipeline") not in sys.path:
    sys.path.append(str(ROOT / "round8_pipeline"))

import torch  # noqa: E402
import model_factory as mf  # noqa: E402
import video_infer as vi  # noqa: E402  (round9 version)
import postfuse as pf  # noqa: E402

R11_VARIANTS = ("r5_v1", "r11")


def make_variant(name: str, model, variant: str):
    """variant: 'r5_v1' | 'r11[+flag...]'  -> (model_fn, pipeline_class).  Flags are parsed by impl/r11_model.py."""
    if variant == "r5_v1":
        from r5_model_impls import make_model_fn_r5
        return make_model_fn_r5(name, model, impl="r5_v1"), vi.VideoPipeline
    if variant.split("+")[0] == "r11":
        import r11_model
        return r11_model.make_variant(name, model, variant)
    raise ValueError(variant)


def make_pipeline(model, fn, pcls, slots=4, inflight=2):
    return pcls(model, fn, 1440, 1900, slots=slots, inflight=inflight)
