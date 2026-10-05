"""Stack presets for round19 on top of round16's run.py parser (round16's run.py is the shared bench / gate / graph harness).

    R0      original Lightly step (patches = ())
    P1/TB1  torch BF16 reference (common.config.P1_STACK)
    P2      TB1 + r13 fused block + r17 MXFP8
    P2R15   P2 + r15 flat optimizer
    P2R14   P2 + r14 fused head
    FINAL   P2 + r14 + r15 (common.config.P2R14R15_STACK) = the candidate final stack, run under StepGraphRunner
Bisect presets (used only if FINAL fails a gate): FINAL_noR14 (= P2R15), FINAL_noR15 (= P2R14), FINAL_fc (r17 policy "fc"), FINAL_noMX (P1R13 + r14 + r15).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import (  # noqa: E402
    P1_PATCH_KWARGS, P1_STACK, P1R13_PATCH_KWARGS, P1R13_STACK, P1R13R14_PATCH_KWARGS, P1R13R14_STACK, P2_PATCH_KWARGS, P2_STACK,
    P2R14_PATCH_KWARGS, P2R14_STACK, P2R14R15_PATCH_KWARGS, P2R14R15_STACK, P2R15_PATCH_KWARGS, P2R15_STACK,
)
from round16_full_graph import run as r16  # noqa: E402

FINAL_STACK = P2R14R15_STACK
FINAL_KWARGS = P2R14R15_PATCH_KWARGS


def _fc(stack, kw):
    kw = {**kw, "round17_mxfp8_sm120": {"variant": "fc"}}
    return stack, kw


def register(round_dir: Path) -> None:
    """Add the presets / short names to round16's parser and redirect its outputs (profile/, results.json, tmp) to ``round_dir``."""
    r16.SHORT["r14"] = "round14_head_sm120"
    r16.DEFAULT_KW["round14_head_sm120"] = {"variant": "fused"}
    r16.PRESET_CFG.update({
        "TB1": (P1_STACK, P1_PATCH_KWARGS),
        "P2R15": (P2R15_STACK, P2R15_PATCH_KWARGS),
        "P2R14": (P2R14_STACK, P2R14_PATCH_KWARGS),
        "P1R13R14": (P1R13R14_STACK, P1R13R14_PATCH_KWARGS),
        "FINAL": (FINAL_STACK, FINAL_KWARGS),
        "FINAL_noR14": (P2R15_STACK, P2R15_PATCH_KWARGS),
        "FINAL_noR15": (P2R14_STACK, P2R14_PATCH_KWARGS),
        "FINAL_fc": _fc(FINAL_STACK, FINAL_KWARGS),
        "FINAL_noMX": (P1R13R14_STACK + ("round15_optimizer_flat",), {**P1R13R14_PATCH_KWARGS, "round15_optimizer_flat": {"variant": "flat"}}),
    })
    r16.ROUND = round_dir
    r16.TMP = Path("/root/autodl-tmp/train-opt-tmp") / round_dir.name
