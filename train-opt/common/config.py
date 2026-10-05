"""Paths and fixed benchmark conventions for train-opt."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_OPT_ROOT = PROJECT_ROOT / "train-opt"
TMP_ROOT = Path("/root/autodl-tmp/train-opt-tmp")
DATASET_ROOT = Path("/root/autodl-tmp/sam3_head_add_nolabel_2026_7_20_19h57")
DATA_ROOT = TMP_ROOT / "data" / "sam3_head"
WEIGHTS_ROOT = TMP_ROOT / "weights"
RUNS_ROOT = TMP_ROOT / "runs"

MODEL_NAMES = {
    "vitl16": "dinov3/vitl16-eomt",
    "vitb16": "dinov3/vitb16-eomt",
}
EXPORT_WEIGHTS = {
    "vitl16": Path("/root/autodl-tmp/vitl-best.pth"),
    "vitb16": Path("/root/autodl-tmp/vitb-best.pth"),
}
BACKBONE_WEIGHTS = {
    model: WEIGHTS_ROOT / f"{model}_backbone.pth" for model in MODEL_NAMES
}
MODEL_SHAPES = {
    "vitl16": {"layers": 24, "dim": 1024, "heads": 16, "head_dim": 64},
    "vitb16": {"layers": 12, "dim": 768, "heads": 12, "head_dim": 64},
}
B_AUTO: dict[str, int | None] = {"vitl16": 7, "vitb16": 14}
PHASES = {"early": (1.0, 1.0, 1.0), "mid": (0.0, 0.5, 1.0), "late": (0.0, 0.0, 0.0)}
PHASE_WEIGHTS = {"early": 0.2, "mid": 0.6, "late": 0.2}
# The stack has deliberately been kept as data rather than hidden in the
# training entrypoint.  This makes it possible for the per-round runners and
# the integration benchmark to use exactly the same composition.  Variants
# are supplied by ``DEFAULT_PATCH_KWARGS`` below; callers may override them
# explicitly for experiments.
P0_STACK: tuple[str, ...] = (
    "round1_torch_cleanup",
    "round3_loss_matcher",
    "round4_attention",
    "round5_block_fusion",
    "round7_optimizer",
)
OPT_EAGER_PATCHES: tuple[str, ...] = ("round1_torch_cleanup",)
# Published only after both models pass the measured performance/C/D gates.
BEST_STACK: tuple[str, ...] = ()
# Phase 2 (PLAN_v2.md / CONTRACTS.md).  P1_STACK is the torch-BF16 reference
# ("T-BF16") published by round11; BEST_STACK_V2 is filled by round19 only.
PHASE2_ROUNDS: tuple[str, ...] = (
    "round11_bf16_mixed",
    "round12_attention_sm120",
    "round13_gemm_fusion_sm120",
    "round14_head_sm120",
    "round15_optimizer_flat",
    "round16_full_graph",
    "round17_mxfp8_sm120",
    "round18_data_val",
    "round19_final",
)
# T-BF16 reference ("TB1", round11): torch BF16 autocast + SDPA/RoPE/head choices on top of r1/r3/r5/r7.
# Published by the coordinator on 2026-10-03 from round11 measurements (see round11_bf16_mixed/NOTES.md).
P1_STACK: tuple[str, ...] = (
    "round1_torch_cleanup",
    "round3_loss_matcher",
    "round5_block_fusion",
    "round7_optimizer",
    "round11_bf16_mixed",
)
P1_PATCH_KWARGS: dict[str, dict[str, str]] = {
    "round1_torch_cleanup": {"variant": "nosync+mask_skip+adamw_fused"},
    "round3_loss_matcher": {"variant": "batched_torch"},
    "round5_block_fusion": {"variant": "ln_fuse"},
    "round7_optimizer": {"variant": "capturable_torch"},
    "round11_bf16_mixed": {"variant": "autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16"},
}
# P1 + round13 fused block (+ round17 MXFP8). Published 2026-10-03.
P1R13_STACK: tuple[str, ...] = P1_STACK + ("round13_gemm_fusion_sm120",)
P1R13_PATCH_KWARGS: dict[str, dict[str, str]] = {**P1_PATCH_KWARGS, "round13_gemm_fusion_sm120": {"variant": "full"}}
# Recommended MXFP8 policy (fc1/fc2/proj all GEMMs, qkv dgrad+wgrad; qkv forward stays bf16). Safe fallback: variant "fc".
P2_STACK: tuple[str, ...] = P1R13_STACK + ("round17_mxfp8_sm120",)
P2_PATCH_KWARGS: dict[str, dict[str, str]] = {**P1R13_PATCH_KWARGS, "round17_mxfp8_sm120": {"variant": "fc+proj+qkv_dw"}}
# P2 + round15 flat optimizer; run with round16 StepGraphRunner (RngTape off). Published 2026-10-03.
P2R15_STACK: tuple[str, ...] = P2_STACK + ("round15_optimizer_flat",)
P2R15_PATCH_KWARGS: dict[str, dict[str, str]] = {**P2_PATCH_KWARGS, "round15_optimizer_flat": {"variant": "flat"}}
# + round14 fused head (ops.predict only). Published 2026-10-03.
P1R13R14_STACK: tuple[str, ...] = P1R13_STACK + ("round14_head_sm120",)
P1R13R14_PATCH_KWARGS: dict[str, dict[str, str]] = {**P1R13_PATCH_KWARGS, "round14_head_sm120": {"variant": "fused"}}
P2R14_STACK: tuple[str, ...] = P2_STACK + ("round14_head_sm120",)
P2R14_PATCH_KWARGS: dict[str, dict[str, str]] = {**P2_PATCH_KWARGS, "round14_head_sm120": {"variant": "fused"}}
# Candidate final stack (to be verified by round19): P2 + r14 + r15, run under round16 StepGraphRunner.
P2R14R15_STACK: tuple[str, ...] = P2R14_STACK + ("round15_optimizer_flat",)
P2R14R15_PATCH_KWARGS: dict[str, dict[str, str]] = {**P2R14_PATCH_KWARGS, "round15_optimizer_flat": {"variant": "flat"}}
# Final stack (round19). Run under round16 StepGraphRunner via round19_final/run_final.py. Safer ViT-B tier: r17 variant "fc".
BEST_STACK_V2: tuple[str, ...] = P2R14R15_STACK
BEST_STACK_V2_KWARGS: dict[str, dict[str, str]] = P2R14R15_PATCH_KWARGS
DEFAULT_PATCH_KWARGS: dict[str, dict[str, str]] = {
    "round1_torch_cleanup": {"variant": "nosync+mask_skip+adamw_fused"},
    "round3_loss_matcher": {"variant": "batched_torch"},
    "round4_attention": {"variant": "sdpa_rope"},
    "round5_block_fusion": {"variant": "ln_fuse"},
    "round7_optimizer": {"variant": "capturable_torch"},
}

# ---- Phase 3 (PLAN_v3.md / CONTRACTS Amendment 6, 6a, 6b): NVFP4 on top of FINAL = BEST_STACK_V2.  Appended 2026-10-04; nothing above is changed.
# Every stack runs under round16's StepGraphRunner (RngTape off).  Policies are the 6b 13.2 ones; the variant strings use '+' (round20 grammar).
# P3H  = FINAL + round21 real-kernel FP4 head (H3: head.f=mx8, head.dw=nv4+sr).  Passed A/B/S/graph_equiv/export on both models.
P3H_STACK: tuple[str, ...] = BEST_STACK_V2 + ("round21_fp4_quant_head",)
P3H_PATCH_KWARGS: dict[str, dict[str, str]] = {**BEST_STACK_V2_KWARGS, "round21_fp4_quant_head": {"variant": "head.f=mx8,head.dw=nv4+sr"}}
# P3B  = FINAL + round22 real-kernel FP4 backbone dgrad/wgrad.  ViT-L policy B4 (nofirst), ViT-B policy B3.  Passed A/B/S/graph_equiv/export (one ViT-L batch of 24 misses the relaxed min).
P3B_STACK: tuple[str, ...] = BEST_STACK_V2 + ("round22_fp4_backbone",)
P3B_PATCH_KWARGS_VITL: dict[str, dict[str, str]] = {**BEST_STACK_V2_KWARGS, "round22_fp4_backbone": {"variant": "all.d=nv4+sr+all.w=nv4+sr+nofirst"}}
P3B_PATCH_KWARGS_VITB: dict[str, dict[str, str]] = {**BEST_STACK_V2_KWARGS, "round22_fp4_backbone": {"variant": "all.d=nv4+sr+all.w=nv4+sr"}}
# P3   = FINAL + head + backbone (combined stack; the two patches own separate FP4States and each wraps ops.opt_step).  Verified by round23.
P3_STACK: tuple[str, ...] = BEST_STACK_V2 + ("round21_fp4_quant_head", "round22_fp4_backbone")
P3_PATCH_KWARGS_VITL: dict[str, dict[str, str]] = {**P3H_PATCH_KWARGS, **P3B_PATCH_KWARGS_VITL}
P3_PATCH_KWARGS_VITB: dict[str, dict[str, str]] = {**P3H_PATCH_KWARGS, **P3B_PATCH_KWARGS_VITB}

# ---- Final stack of phase 3 (CONTRACTS Amendment 6c, round23).  Appended 2026-10-04; nothing above is changed.
# BEST_STACK_V3 = P3_STACK, run under round16's StepGraphRunner via round23_fp4_final/run_final.py.  The per-model variant strings differ only in ``nofirst`` (ViT-L).
# Passes the relaxed FP4 gate (median >= 0.98) on both models; the strict gate (Amendment 5) is NOT met; see Amendment 6c 14.2 for the exact numbers.
# BEST_STACK_V2 (= FINAL) is untouched and remains the strict-gate fallback.
BEST_STACK_V3: tuple[str, ...] = P3_STACK
BEST_STACK_V3_KWARGS_VITL: dict[str, dict[str, str]] = P3_PATCH_KWARGS_VITL
BEST_STACK_V3_KWARGS_VITB: dict[str, dict[str, str]] = P3_PATCH_KWARGS_VITB
