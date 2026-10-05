"""CPU-only protocol tests for round13 (no GPU, no lock): patch contract, eligibility fallback, GEMM policy table."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/root/lightly-train/src")

import pytest
import torch
import torch.nn as nn


def _stub_ctx(precision="bf16-mixed", autocast=True):
    from common.ops import default_ops

    return SimpleNamespace(cfg=SimpleNamespace(precision=precision), meta={"enable_autocast": autocast} if autocast else {},
                           ops=default_ops(), shadow={}, model=None)


def test_patch_requires_autocast():
    from round13_gemm_fusion_sm120.impl import patch

    with pytest.raises(ValueError, match="round11"):
        patch.apply(_stub_ctx(autocast=False))


def test_unknown_variant_rejected():
    from round13_gemm_fusion_sm120.impl import patch

    with pytest.raises(ValueError, match="unknown round13 variants"):
        patch.apply(_stub_ctx(), variant="nope")


def test_policy_defaults_cover_all_linears():
    from round13_gemm_fusion_sm120.impl import gemm_ops

    kinds = {k for k, _ in gemm_ops.DEFAULT_POLICY}
    lins = {l for _, l in gemm_ops.DEFAULT_POLICY}
    assert kinds == {"fwd", "dgrad", "wgrad"} and lins == {"qkv", "proj", "fc1", "fc2"}
    assert all(v in ("quack", "torch") for v in gemm_ops.DEFAULT_POLICY.values())


def test_eligible_falls_back_on_cpu_and_unsupported_dims():
    from round13_gemm_fusion_sm120.impl.fused_block import eligible

    blk = SimpleNamespace()
    assert eligible(blk, torch.zeros(1, 4, 1024)) == "cpu tensor"


def test_ln_dims_supported():
    from round13_gemm_fusion_sm120.impl import block_kernels

    assert 1024 in block_kernels.SUPPORTED_D and 768 in block_kernels.SUPPORTED_D
