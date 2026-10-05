"""merge finished measurements (peaks, nsys baseline, P0 emulation check, overflow scan summary) into results.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, "impl")
import bench10 as b
H = Path(__file__).resolve().parent
peaks = {
    "unit": "TFLOPS (dense), SM clock ~2.84 GHz during the loops, GPU idle otherwise (infer+train dual lock held)",
    "peak_extra.cu (own, same method as Spark bench_peak)": {
        "bf16_f32acc_m16n8k16": 246.8, "fp16_f32acc_m16n8k16": 246.9, "fp16_f16acc_m16n8k16": 492.6, "int8_s32acc_m16n8k32": 983.2,
        "e4m3_plain_f32acc_m16n8k32": 492.8, "bf16_f32acc_repeat": 247.1},
    "spark bench_peak.cu (built unmodified)": {
        "peak_bf16_mma": 247.09, "peak_fp8_mma(mxf8f6f4 block-scale, unit scales)": 981.32, "peak_fp8_mma_plain(f32 acc)": 492.95, "peak_fp8_mma_f16acc": 983.49,
        "peak_fp4_mma(nvf4)": 1964.35, "peak_fp4_mma_mx(mxf4)": 1964.63, "peak_tf32_mma": 123.32, "peak_fp32_fma": 118.18},
    "ratios_vs_bf16_f32acc": {"fp16_f16acc": 1.995, "int8": 3.98, "mxf8f6f4": 3.97, "plain_fp8_f32acc": 2.0, "plain_fp8_f16acc": 3.98, "nvf4_mxf4": 7.95},
    "log": "profile/peaks.log"}
nsys = {m: json.loads((H / "profile" / f"{m}_r5v1_categories.json").read_text()) for m in ("vitl16", "vitb16")}
b.merge_results(H / "results.json", {"peaks": peaks, "nsys_r5v1_per_frame": nsys})
print("ok")
