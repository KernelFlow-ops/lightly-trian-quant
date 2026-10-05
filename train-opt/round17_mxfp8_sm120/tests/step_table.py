"""Markdown step table from round17 results.json (bench.time_steps results); ratios vs the same-session TB1_R13 and vs the Amendment-1 TB1 / R0 baselines."""
import json
from pathlib import Path

R = Path("/root/lightly-train/train-opt/round17_mxfp8_sm120/results.json")
BASE = {"vitl16": {"R0": 16.2, "TB1": 41.4}, "vitb16": {"R0": 37.2, "TB1": 96.0}}  # CONTRACTS Amendment 1 9.3 (weighted img/s, same machine)
ORDER = ["TB1_R13", "MX8_fc", "MX8_fc+proj", "MX8_fc+proj+qkv_dw", "MX8_all"]
d = json.loads(R.read_text())
for model, title in (("vitl16", "ViT-L B=7"), ("vitb16", "ViT-B B=14")):
    ref = d[model]["TB1_R13"]["gpu_step"]["weighted_img_s"]
    print(f"\n**{model}** ({title}; phases early/mid/late, weights 0.2/0.6/0.2)\n")
    print("| stack | early ms | mid ms | late ms | weighted img/s | vs TB1_R13 | vs TB1 (41.4/96.0) | vs R0 (16.2/37.2) | fwd / bwd / opt / loss (mid, ms) | peak mem GB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        e = d[model].get(k)
        if e is None:
            continue
        g = e["gpu_step"]
        w = g["weighted_img_s"]
        b = e["breakdown_ms"]
        print(f"| {k} | {g['early']['step_ms']['mean']:.1f} | {g['mid']['step_ms']['mean']:.1f} | {g['late']['step_ms']['mean']:.1f} | {w:.2f} | {w / ref:.3f}x | {w / BASE[model]['TB1']:.2f}x | {w / BASE[model]['R0']:.2f}x | "
              f"{b['fwd']:.1f} / {b['bwd']:.1f} / {b['opt']:.1f} / {b['loss']:.1f} | {e['peak_mem_gb']:.2f} |")
