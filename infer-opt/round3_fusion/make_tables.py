"""Print markdown tables for NOTES.md from results.json (so numbers are copied, never typed)."""
import json
from pathlib import Path

d = json.loads((Path(__file__).parent / "results.json").read_text())
print("### microbench K1-K4 (gpu = CUDA-event p50 per call incl. launch gaps of multi-op torch sequences; wall = host view)")
print("| kernel | ours gpu us | torch gpu us | speedup | ours kernel-time us (n) | torch kernel-time us (n) | speedup (kernel) | ours TFLOPS by kernel-time (% of 232) | torch linear-only kernel us | tiles / waves |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, v in d.items():
    if not (isinstance(v, dict) and "ours" in v and k[:2] in ("k1", "k2", "k3", "k4")):
        continue
    tf = f"{v['ours_kernel_tflops']:.0f} ({v['ours_kernel_pct_of_max232']:.0f}%)" if "ours_kernel_tflops" in v else "-"
    lo = f"{v['torch_linear_only_kernel_us']:.1f}" if "torch_linear_only_kernel_us" in v else "-"
    wv = f"{v['waves']['tiles']} / {v['waves']['waves']:.2f}" if "waves" in v else "-"
    print(f"| {k} | {v['ours']['gpu_us']:.1f} | {v['torch']['gpu_us']:.1f} | x{v['speedup_gpu']:.2f} | "
          f"{v['ours']['kernel_us']:.1f} ({v['ours']['kernels']:.0f}) | {v['torch']['kernel_us']:.1f} ({v['torch']['kernels']:.0f}) | x{v['speedup_kernel_time']:.2f} | {tf} | {lo} | {wv} |")
print()
print("### whole layer")
print("| model | layer / variant | kernels | kernel-time us (GPU busy) | gpu us (events) | wall us |")
print("|---|---|---|---|---|---|")
for m, r in d.get("block_bench", {}).items():
    for k, v in r.items():
        print(f"| {m} | {k} | {v['kernels']} | {v.get('kernel_us', float('nan')):.1f} | {v['gpu_us']:.1f} | {v['wall_us']:.1f} |")
print()
print("### model stage (batch=2 crops)")
for m, r in d.get("model_stage_latency", {}).items():
    for k, v in r.items():
        print(f"| {m} | {k} | gpu {v['gpu_ms']:.2f} ms | wall {v['wall_ms']:.2f} ms | kernels {v['kernels']} |")
