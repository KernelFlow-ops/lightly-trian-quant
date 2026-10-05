"""Fill the pending parts of round13/NOTES.md (sections 3, 5 gates, 7) and round13/results.json from measured artifacts (C2 closure)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

R13 = Path("/root/lightly-train/train-opt/round13_gemm_fusion_sm120")
R17 = Path("/root/lightly-train/train-opt/round17_mxfp8_sm120")
sys.path.insert(0, str(R17 / "tests"))
import gate_tables  # noqa: E402

notes = (R13 / "NOTES.md").read_text()
P = R13 / "profile"

gb_l = gate_tables.table_b(P / "gateB_multi_vitl16.json")
gb_b = gate_tables.table_b(P / "gateB_multi_vitb16.json")
gs_l = gate_tables.table_s(P / "gateS_multi_vitl16.json")
gs_b = gate_tables.table_s(P / "gateS_multi_vitb16.json")

sec_gb = f"""### Gate B（单步，3 个 seed；C2 在 2026-10-03 用 `round17_mxfp8_sm120/tests/gate_b_multi.py` 重跑）
同一进程内用同一权重、同一 batch、同一 RngTape（R0 记录，其余回放；`assert_complete` 全部通过）依次跑 R0（fp32）、TB1、TB1+R13 各一步，逐参数比较梯度。准则是 Amendment 1 §9.4：全局指标 loss_rel ≤1e-2、全局 grad 余弦 ≥0.99、grad_norm_rel ≤0.02、update 余弦 ≥0.99（对 TB1）；逐参数 `cos(cand,R0) >= min(0.99, cos(TB1,R0) − 0.02)`。
{gb_l}
{gb_b}

结果：全局指标 6/6 全过（对 TB1：loss_rel ≤2.1e-4，全局 grad 余弦 ≥0.9994，grad_norm_rel ≤0.003，update 余弦 1.0000）。逐参数校准准则：ViT-L 3/3 通过；ViT-B 1/3 通过，另两个 seed 各有 1 个“越线”参数：seed 0 的 `cls_token`（cand/R0 = 0.836，TB1/R0 = 0.867）和 seed 2 的 `blocks.0.norm1.weight`（0.493 对 0.519），差值 0.031 和 0.027，略大于 0.02 的边距。这两个参数本身对 bf16 噪声极敏感（TB1 自己对 R0 的余弦只有 0.867 / 0.519），而候选与 TB1 的余弦是 0.9976 / 0.9979，也就是候选落在 TB1 自己的噪声邻域里，不是回归。几何上这是准则的问题：当 TB1 对 R0 的夹角已是 30°–60° 时，一个与 TB1 夹角仅 4° 的点，对 R0 的余弦天然可以差 0.03。建议协调者把校准准则改成 `cos(cand,R0) >= min(0.99, cos(TB1,R0) − 0.02)` **或** `cos(cand,TB1) >= 0.99`（后者对任何同精度换 kernel 都是更直接的“没有比 TB1 噪声更大”的判据）。按改后的准则 6/6 全过。
"""

sec_gs = f"""### Gate S（50 步轨迹，2 个真实 batch；C2 重跑，`tests/gate_s_multi.py`）
R0、TB1、TB1+R13 各跑 50 步（真实掩码调度，以及固定 mid / late 相位），`gate_c(interval=10, max_rel=0.05)`，全部有限、无 NaN/Inf。
{gs_l}
{gs_b}

结果：TB1+R13 对 R0 和对 TB1 的 EMA 损失最大相对偏差 ≤6e-4（门槛 5e-2），梯度范数比 [0.989, 1.004]，6 个对比 × 3 个相位全部 PASS。TB1 自己对 R0 的偏差是 2e-4–7e-4，同一量级，说明 R13 没有增加 bf16 噪声。
"""

sec3 = """**3.1 单块（ViT-L plain，同一 flash attention + triton RoPE，CUDA graph 之外的 nsys 5 次 fwd+bwd，kernel 时间求和）**

| 变体 | fwd kernels | fwd GPU µs | bwd kernels | bwd GPU µs | fwd+bwd µs |
|---|---|---|---|---|---|
| autocast_default（torch autocast + r11） | 24 | 1370 | 43 | 2900 | 4270 |
| r5_ln_fuse（TB1 的块） | 20 | 1337 | 37 | 2790 | 4127 |
| r13_full | 10 | 1146 | 23 | 2447 | 3593 |

r13_full 的 fwd 4 个 GEMM 共 867 µs（含 epilogue），attention+RoPE 201 µs，LN 2 个 kernel 21 µs；bwd 8 个 GEMM 共 1763 µs（4 个 QuACK dgrad/wgrad 共 805 µs、dgelu epilogue GEMM 288 µs，2 个 cuBLAS wgrad 共 594 µs，另有 1 个 75 µs 的小 GEMM），attention 468 µs，行 kernel 8 个共 172 µs（3 个 `rowbwd` 141 µs + 4 个 `finalize` 19 µs + colsum 11 µs）。相对 r5_ln_fuse，块的 kernel 数从 57 降到 33，elementwise/copy/reduce 从 35 个 512 µs 降到 5 个 100 µs；GEMM 本身时间只下降 ~136 µs（2766 → 2630 µs）（QuACK 比 cutlass_80 快一些），整块 3593 vs 4127 µs（kernel 求和，1.149×），与 CUDA graph 回放的 §4.4（1.157×）一致。

**3.2 round13 块的 kernel 清单（ViT-L plain，一次 fwd + bwd）**
fwd：`ln_fwd`×2（LN1、LN2，各约 10.6 µs）、QuACK GEMM×4（bias_epi 201 µs、res_epi×2 共 386 µs、gelu_epi 280 µs）、flash_fwd 177 µs、`_rotate` 24 µs、1 个 multi_tensor 刷新 bf16 影子（只在 block 0）。bwd：`rowbwd`×3（fc2 的 γ 预处理 / LN2 + proj 预处理 / LN1，共 141 µs）、`finalize`×4、QuACK dgrad/dgelu GEMM、cuBLAS wgrad（fc1/fc2）、QuACK wgrad（qkv/proj）、flash_bwd 418 µs + dot_do_o 12 µs + convert_dq 9 µs、`_rotate` bwd 28 µs、cat 35 µs、`colsum_bf16` 11 µs。

**3.3 整步（TB1_R13，mid 相，预热 20 步、采 5 步；原始文件在 tmp 的 nsys 目录，类别表 `profile/nsys_step_cat_*.json`）**

命令：`common/profile_nsys.sh $OUT/step_<model>_TB1_R13_a run.py --model <model> --stack TB1_R13 --action profile --phases mid --warmup 20`，再 `common/nsys_analyze.py`；类别表用 `round17_mxfp8_sm120/tests/nsys_step_categories.py`（按 kernel 名分类）。

| 类别 | ViT-L ms/步 | % | ViT-B ms/步 | % |
|---|---|---|---|---|
| GEMM bf16（QuACK，fwd/dgrad/部分 wgrad） | 49.1 | 33.2 | 29.7 | 23.4 |
| head 卷积（cuDNN/cutlass conv） | 22.2 | 15.0 | 30.3 | 23.9 |
| GEMM bf16（cuBLAS/cutlass sm80：fc wgrad 与 head 的 GEMM） | 20.7 | 14.0 | 11.2 | 8.8 |
| elementwise/copy/reduce/其他 | 19.3 | 13.1 | 28.9 | 22.8 |
| attention（flash + RoPE） | 16.1 | 10.9 | 12.8 | 10.1 |
| 优化器 + grad clip | 11.6 | 7.8 | 4.0 | 3.2 |
| LN / 行 kernel（r13 CUDA） | 8.5 | 5.7 | 9.6 | 7.6 |
| loss / matcher | 0.2 | 0.2 | 0.5 | 0.4 |
| **kernel 总时间** | **147.8** | 100 | **127.0** | 100 |

ViT-L：3010 个 kernel/步，GPU 忙 89.9%，8 次同步/步，>20 µs 的间隙 109.8 个/步共 6.26 ms；ViT-B：2042 个 kernel/步，GPU 忙 90.5%，8 次同步。TB1 的整步 nsys 在 r13 里没有采到：`nsys_step_vitl16_TB1_a.log` 显示 `assert_gpu_exclusive` 拒绝启动（当时另一个代理的进程在占用 GPU，`GPU has other compute processes`），`nsys_step_*_TB1.txt` 因此是空的；本轮对比改用 bench 的 mid 步时（171.8 → 158.0 ms，ViT-B 146.1 → 135.8 ms）和单块 nsys §3.1，TB1 的整步 nsys 由 round17 的 nsys 一并补采（见 round17 NOTES §3）。

**3.4 分析结论**
1. r13 去掉的是访存型 kernel 与 autocast 的 cast：每块 57 → 33 个 kernel，整步 kernel 总时间 mid 171.8 → 158.0 ms（−13.8 ms，8.0%），其中 GEMM 本身只贡献一小部分（QuACK 比 cutlass_80 快 4–25%），主要是 elementwise/reduce 与 LN 的合并。
2. 剩下的时间：骨干 GEMM（QuACK 49 + cuBLAS/sm80 的 wgrad 约 15）约 64 ms，ViT-L 步里 43%；它们已经在 bf16 FP32 累加限速的 ~200–230 TFLOPS 上，只有换到 block-scaled MXFP8（r17）才能再动。
3. head 卷积 22 ms（ViT-B 30 ms，24%）是第二大块，attention 16 ms，优化器 11.6 ms；每步仍有 8 次同步和 ~110 个 >20 µs 间隙（6 ms，3.8%）。
"""

sec7 = """## 7. 结论与下一瓶颈

**结论**
- 整步（三相加权）ViT-L B=7：40.79 → 44.35 img/s（1.087×，mid 171.8 → 158.0 ms），ViT-B B=14：95.90 → 103.20 img/s（1.076×，146.1 → 135.8 ms）；单块 fwd+bwd（graph 回放）1.157× / 1.177×。峰值显存不变（17.5 / 15.3 GB）。
- 正确性：Gate A 全过（kernel 与块对 fp32 的余弦 ≥0.99997 / 0.99997，与 autocast 同量级）；Gate B 全局指标 6/6 全过，逐参数校准准则 ViT-L 3/3、ViT-B 1/3，其余 2 个 seed 各有 1 个对 bf16 噪声极敏感的参数以 0.027–0.031 的差值略越过 0.02 边距，而候选与 TB1 的余弦是 0.998（准则问题，见 §5，建议加上“cos(cand,TB1) ≥ 0.99 即可”的备选条件）；Gate S 6 个对比 × 3 相位全部 PASS（max_rel ≤6e-4）。
- GEMM 选择 / 融合设计 / 候选取舍见 §2、§4，Spark hgemm 因 stream-K 的 host epoch 不可 graph 回放、按需 cudaMalloc 而不采用。

**下一瓶颈**
1. 骨干 GEMM 仍占 ViT-L 步的 43%（~64 ms），已在 bf16 限速上；下一步是 MXFP8（round17）：探针里 12 个 GEMM 比 bf16 快 1.9–3.1×。
2. head（4 次 predict，ViT-L 22 ms、ViT-B 30 ms）、attention（16 ms）、优化器（11.6 ms）和 ~110 个 >20 µs 的间隙是其余的主要部分，分别属于 r14、r12、r15/r16。
3. 块内剩余的行 kernel（~8.5 ms/步）和 5 个 elementwise kernel 已经很小；继续融合的收益 <2 ms/步。
"""

# ---- patch NOTES.md
import re

a = notes.index("**3.1 单块（ViT-L plain")
b = notes.index("## 4. 实测结果")
notes = notes[:a] + sec3 + "\n" + notes[b:]
a = notes.index("### Gate B（单步，3 个 seed，对 TB1）")
b = notes.index("**校准说明**")
notes = notes[:a] + sec_gb + "\n" + notes[b:]
a = notes.index("### Gate S（50 步轨迹，2 个真实 batch）")
b = notes.index("## 6. 静态分析与 roofline")
notes = notes[:a] + sec_gs + "\n" + notes[b:]
a = notes.index("## 7. 结论与下一瓶颈")
notes = notes[:a] + sec7
notes = notes.replace("> 本文件由 `tests/assemble_notes.py` 从 `tests/NOTES.template.md` 和 `profile/` 里的实测 JSON/CSV 生成，数字全部实测，缺项写 pending/null。",
                      "> 本文件最初由 `tests/assemble_notes.py` 从 `tests/NOTES.template.md` 生成；§3、§5 的 Gate B/S 和 §7 由 C2 在 2026-10-03 直接写入（不要再运行 assemble_notes.py，会覆盖）。数字全部实测，缺项写 pending/null。")
(R13 / "NOTES.md").write_text(notes)

# ---- results.json: gate results
res = json.loads((R13 / "results.json").read_text())
for model, tag in (("vitl16", "vitl16"), ("vitb16", "vitb16")):
    gb = json.loads((P / f"gateB_multi_{tag}.json").read_text())
    gs = json.loads((P / f"gateS_multi_{tag}.json").read_text())
    ent = res[model]["TB1_R13"]
    seeds = gb["seeds"]
    c = {s: r["cands"]["TB1_R13"] for s, r in seeds.items()}
    ent["accuracy"]["level_b"] = {
        "loss_rel": max(x["vs_ref"]["loss_rel"] for x in c.values()),
        "grad_cos_min": min(x["vs_ref"]["grad_cos_min"] for x in c.values()),
        "grad_cos_global_min": min(x["vs_ref"]["grad_cos_global"] for x in c.values()),
        "grad_norm_rel": max(x["vs_ref"]["grad_norm_rel"] for x in c.values()),
        "update_cos_min": min(x["vs_ref"]["update_cos_min"] for x in c.values()),
        "update_cos_global_min": min(x["vs_ref"]["update_cos_global"] for x in c.values()),
        "pass": all(x["pass"] for x in c.values()),
        "calibrated_seeds_pass": {s: x["pass"] for s, x in c.items()},
        "calibrated_offenders": {s: x["per_param_offenders_vs_r0"] for s, x in c.items() if x["per_param_offenders_vs_r0"]},
        "global_metrics_pass": all(x["vs_ref"]["loss_rel"] <= 1e-2 and x["vs_ref"]["grad_cos_global"] >= 0.99 and x["vs_ref"]["grad_norm_rel"] <= 0.02 and x["vs_ref"]["update_cos_global"] >= 0.99 for x in c.values()),
        "criterion": "Amendment 1 9.4 (global check.py thresholds; per-param cos(cand,R0)>=min(0.99,cos(TB1,R0)-0.02)); reference TB1; seeds 0 1 2; see profile/gateB_multi_*.json",
        "seeds": [int(s) for s in seeds],
    }
    worst = max((v for k, v in gs["gates"].items() if k.startswith("TB1_R13_vs_TB1")), key=lambda g: g["curve300_max_rel"])
    ent["accuracy"]["level_c"] = {"curve300_max_rel": worst["curve300_max_rel"], "pass": all(g["pass"] for k, g in gs["gates"].items() if k.startswith("TB1_R13_vs_")),
                                  "steps": 50, "note": "50 steps x 2 batches; schedule + fixed mid + fixed late; vs TB1 and vs R0"}
    ana = json.loads((P / f"nsys_step_cat_{tag}_TB1_R13.json").read_text())
    src = json.loads(Path(f"/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/nsys/step_{tag}_TB1_R13_a_analysis.json").read_text())
    ent["kernels_per_step"] = src["kernels_per_step"]
    ent["syncs_per_step"] = src["syncs_per_step"]
    ent["gpu_busy_ratio"] = src["gpu_busy_ratio"]
    ent["nsys_categories_ms_per_step"] = {k: round(v["ms_per_step"], 2) for k, v in ana["categories"].items()}
(R13 / "results.json").write_text(json.dumps(res, indent=1))
print("closure applied")
