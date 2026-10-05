"""Assemble results.json of round20_fp4_probe part R20a from the measured profile/*.json."""
import json
from pathlib import Path

ROUND = Path(__file__).resolve().parents[1]
P = ROUND / "profile"
est = json.load(open(P / "per_step_estimate.json"))
sel = json.load(open(P / "fp4_ref_selftest.json"))["results"]
err = json.load(open(P / "fp4_error_table.json"))
qc = json.load(open(P / "fp4_quant_cost.json"))
qut = json.load(open(P / "fp4_gemm_qutlass.json"))
pref = json.load(open(P / "debug_prefix_tokens.json"))
bs = json.load(open(P / "backend_sums.json"))
STEP = {"vitl16": 101.8, "vitb16": 86.6}

res = {"round": "round20_fp4_probe", "part": "R20a", "measured_on": "infer-5090 (RTX 5090), CUDA-graph replay, dual lock, median of 3 interleaved rounds"}
res["fp4_ref_selftest"] = {
    "a_nv4_rn_vs_spark_cuda_quantizer": {k: sel["a_rn_vs_quack_spark"][k]["nv4_vs_spark"] for k in ("gauss", "spiky8", "tiny_blocks")},
    "a_nv4_rn_vs_quack_to_nvfp4": {k: sel["a_rn_vs_quack_spark"][k]["nv4_vs_quack"] for k in ("gauss", "spiky8", "tiny_blocks")},
    "a_mx4_floor_mode_vs_quack_and_spark": {k: [sel["a_rn_vs_quack_spark"][k]["mx4_floor_vs_quack"], sel["a_rn_vs_quack_spark"][k]["mx4_floor_vs_spark"]] for k in ("gauss", "spiky8", "tiny_blocks")},
    "b_gemm_vs_fp64_dequant": sel["b_gemm_vs_dequant_fp64"],
    "c_sr": sel["c_sr_unbiased"], "d_rht": sel["d_rht"], "f_weight_2d": sel["f_weight_2d"], "g_padding": sel["g_padding"], "e_default_rng_untouched": sel["e_default_rng_untouched"],
}
summ = {}
for m, v in est["models"].items():
    summ[m] = {"per_step_gemm_ms": {k: (None if x is None else round(x / 1e3, 3)) for k, x in v["per_step_gemm_us"].items()},
               "items": {k: {kk: (None if vv is None else round(vv, 3)) for kk, vv in it.items()} for k, it in v["items"].items()},
               "speed_conditions_x": {k: (None if x is None else round(x, 3)) for k, x in v["speed_conditions"].items()},
               "check_vs_r19_kernel_table": v["measured_gemm_vs_r19_kernel_table"], "prefix_side_gemm_ms": v.get("prefix_side_gemm_ms"), "plain_nv4_backend_sum_ms": bs[m]["plain_nv4_backend_sum_ms"]}
res["gemm_matrix_summary"] = {"models": summ, "files": ["profile/fp4_gemm_matrix.json", "profile/fp4_gemm_matrix_<model>_<bbM|head>.json"],
                              "backend_notes": {"quack": "best overall; FP4 epilogues (bias, gamma+fp32 residual, gelu, gelu') compile and run; fp32 wgrad out; 1.07-1.26 PFLOPS plain",
                                                "cublaslt_scaled_mm": "NVFP4 only (MXFP4: NotImplementedError 'only supported for B200/B300'); within 2-10% of QuACK, wins on small-N wgrad outputs without split-K; no epilogue, no out=",
                                                "spark_fp4gemm": "bf16 out only, K%256 (wgrad token axis padded to 256), TMA variant needs M,N%128: 1.1-1.5x slower than QuACK on the training shapes",
                                                "qutlass": {"built": True, "correctness_relL2_nv4": qut.get("correctness_nv4_qutlass_relL2"), "correctness_relL2_mx4": qut.get("correctness_mx4_qutlass_relL2"),
                                                            "note": "bf16 out only; faster than QuACK only on some large-K wgrad/head shapes (up to ~12%), slower (10-80%) elsewhere; sum over 48 shapes nv4 4.14 ms vs QuACK 3.67 ms"},
                                                "cutlass_ex79": "not touched (QuACK/cuBLASLt reach >=80% of Spark: they are faster than Spark)"}}
res["quant_cost"] = {"hbm_copy_GBps": qc["hbm_copy_GBps"], "file": "profile/fp4_quant_cost.json",
                     "measured_us_examples": {"vitl16_M7203x1024": {k: round(x, 1) for k, x in qc["models"]["vitl16"]["M7203x1024"]["us"].items()}, "vitl16_M7203x4096": {k: round(x, 1) for k, x in qc["models"]["vitl16"]["M7203x4096"]["us"].items()},
                                              "vitl16_head_dZ28672x4096": {k: round(x, 1) for k, x in qc["models"]["vitl16"]["head_dZ28672x4096"]["us"].items()}},
                     "floor_us_examples": {"vitl16_M7203x4096": {k: round(x, 1) for k, x in qc["models"]["vitl16"]["M7203x4096"]["floor_us"].items()}},
                     "conclusions": ["Row-only FP4 quantizers (Spark, QuTLASS) run at the HBM-read floor (~1.0-1.2x of read time)", "Every existing transposed-copy path is 6-10x off the floor (x.t().contiguous() alone 72 us for [7203,1024]; QuTLASS backward_t_bf16 59 us; Spark col = transpose+quant 98 us vs r17 mx8 dual 8 us)",
                                     "=> a custom one-read dual (row + transposed) FP4 quantizer is mandatory; unfused existing kernels turn the step estimate negative (-23..-31 ms)", "NVFP4 needs the tensor amax before quantizing: from the producer (free) or a separate pass (+2 B/elem read, +0.5-1.5 ms .. +3 ms/step)"]}
res["error_table_summary"] = {"files": ["profile/fp4_error_table.json", "profile/fp4_error_table_<model>.json"], "capture": "R0 modules, bf16-mixed autocast, one real batch (seed 0); hooks on nn.Linear / ConvTranspose2d (see NOTES 5.1)",
                              "median_and_max_relL2_by_group": {m: err["models"][m]["summary"] for m in err["models"]},
                              "massive_activation_finding": {"file": "profile/debug_wgrad_block0.json, profile/debug_prefix_tokens.json (ViT-L)", "per_layer_prefix_exclusion": pref,
                                                             "text": "cls/register(/query) prefix tokens carry 29-93% of the dW norm; FP4 wgrad over all tokens is 0.26-0.60 relL2 in block0 fc1/fc2 (RHT: 0.72 / 9.1-25), with the prefix tokens (5 per image, 105 in joint blocks) kept out of the FP4 reduction and added exactly the error is 0.04-0.06 (RHT 0.045-0.06) and 0.005-0.02 in block 22"}}
res["per_step_estimate"] = {"step_ms_final": STEP, "model": {m: {"step_estimate": v["step_estimate"], "items": v["items"], "quant_scenarios": v["quant_us"].get("scenarios")} for m, v in est["models"].items()},
                            "assumptions": ["GEMM times are measured (matrix); quantizer costs for the fused FP4 dual quantizer are ESTIMATES = measured r17 mx8 dual time scaled by bytes moved x1.25 (not measured: no such kernel exists yet)", "existing-kernels scenario is measured (Spark row + transpose-copy + Spark quant)",
                                            "weight quantization (W and W^T per step) assumed equal to the r17 weight quantization already in FINAL (not separately estimated)", "S1 = head dgrad+wgrad nv4 + backbone dgrad+wgrad nv4 (forward unchanged); S2 adds fc fwd nv4, qkv fwd mx8, head fwd mx8", "no accuracy gate applied (R20b)"]}
fx = {}
for m, v in est["models"].items():
    fx[m] = {}
    for scen in ("fused_nv4_amax_from_producer", "fused_nv4_separate_amax_pass", "fused_mx4"):
        e = v["step_estimate"][scen]
        fx[m][scen] = {k: {"saving_ms": round(x["saving_ms"], 2), "saving_pct_of_step": round(100 * x["saving_ms"] / STEP[m], 2), "meets_>=3ms_or_>=3pct": bool(x["saving_ms"] >= 3.0 or 100 * x["saving_ms"] / STEP[m] >= 3.0)} for k, x in e["per_item"].items()}
res["go_no_go_speed_and_saving_only"] = {"note": "Amendment 6 11.6 conditions 1 (speed) and 3 (saving) only; condition 2 (simulated B/S gates) is R20b", "per_item": fx,
                                         "speed_conditions_x": {m: v["speed_conditions"] for m, v in est["models"].items()}}
res["recommendation"] = {
    "format": "nv4 default (mx4 GEMM is only 3-7% faster; its operand error is 1.5-1.7x nv4: fwd 0.18 vs 0.11, dgrad 0.20 vs 0.14, wgrad 0.14 vs 0.076 median relL2); mx4 stays a comparison arm: it only wins on the step if nv4 needs a separate amax pass (+4.3 ms ViT-L / +4.9 ms ViT-B)",
    "backend": "QuACK 0.6.5 (FP4 operands via BlockScaledOperand, nv4 per-tensor scale as a device-scalar epilogue arg, fp32 wgrad out, split_k for small outputs); cuBLASLt F.scaled_mm nv4 as plain-GEMM alternative; Spark / QuTLASS not recommended",
    "r20b_candidates": ["head.dw=nv4+sr (+rht on w) [main]; head.dw=mx8 [fallback]", "all.w=nv4+rht+sr, nofirst (block 0 massive activations) and the prefix-token side GEMM variant", "all.d=nv4+sr (w2d vs w1d)",
                        "fc.f=nv4 (+rht) [stretch; speed condition fails on ViT-B (1.47x) and the saving is < 3 ms]", "qkv.f=mx8 and head.f=mx8 (r17 kernels; saving 2.7 / 3.3 ms on ViT-L)", "head.f=nv4 [stretch: operand error 0.144 vs mx8 0.036]",
                        "controls: mx4 vs nv4 at the best nv4 policy, sr vs rn, rht on/off, nolast/nojoint"],
    "design_requirements_for_R21_R22": ["one-read dual quantizer (row + transposed copy, zero padded token axis, blocked scales), amax from the producer for nv4", "prefix-token (cls+reg) exclusion from the FP4 wgrad reduction + small bf16 side GEMM, or nofirst",
                                        "wgrad/dgrad small-output shapes need split-K (QuACK split_k=None) or cuBLASLt", "NVFP4 e4m3 scale range: 24% of block0 dY blocks flush to scale 0 under a per-tensor ts set by outlier tokens"]}
res["failures_and_limits"] = ["F.scaled_mm MXFP4 on sm_120: NotImplementedError 'MXFP4 scaling only supported in CUDA for B200/B300'", "Spark fp4gemm: bf16 out only; K%256 (wgrad Mpad 7296/14464 needs padding to 7424/14592); first build missed fp8quant.cu (absmax_bits undefined symbol), fixed in the build script; my binding needs an explicit variant (-1 not resolved)",
                               "Spark device-amax path: ts not bit-equal to amax/(6*448) fp32 division on 2 of 3 test inputs (not quantified)", "QuACK to_nvfp4 differs from fp4_ref: 0.09% of data bytes on Gaussian (reciprocal multiply), 6-8% scale/data bytes on tiny blocks (scale clamped to 2^-6)",
                               "error table captured from the R0/bf16-mixed reference modules, not from FINAL's fused blocks (no per-Linear boundary there)", "no nsys capture (operator-level graph replay only)", "mx4 epilogue-fused numbers measured only for fc1 fwd and fc2 dgrad, other mx4 epilogue cases are derived (marked)"]
res["not_measured"] = ["a real fused FP4 quantizer", "end-to-end step time of any FP4 policy", "accuracy gates B/S (R20b)", "QuACK FP4 + epilogue for mx4 on all cases", "weight quantization cost per step"]
(ROUND / "results.json").write_text(json.dumps(res, indent=1, default=str))
print("results.json written", len(json.dumps(res)))
print(json.dumps(res["go_no_go_speed_and_saving_only"]["per_item"], indent=None)[:3500])
