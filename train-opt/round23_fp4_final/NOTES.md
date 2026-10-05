# Round23（R23）：FINAL + FP4 head（R21b）+ FP4 骨干（R22）组合栈 P3 的整步实测、门和训练入口（第三阶段）

> 数字全部在 `infer-5090`（RTX 5090）上实测，所有 GPU 进程持双锁；没测的写 not measured / null。机器可读结果：`results.json`（由 `tests/make_results.py` 从 `profile/*.json` 生成，随时可重跑）。
> 没改 round0–22、common、PLAN、CONTRACTS、README；只 import 了 r13–r22 的代码。会话额度中断过两次（不是任务出错），已从队列续跑，已完成的项没有重做。

## 0. 摘要

> 状态：**本轮全部完成**（冒烟、整步 bench、门 A / B / S / graph_equiv、导出 / resume、`run_final.py`、e2e、nsys、roofline、可选 5a）；没测的见 §7。

| 项 | 结果 |
|---|---|
| 交付 | `run.py`（预设 `P3` `P3H` `P3B` `P3_RN` `P3_strict` `P3_q8`，按 `--model` 自动选 ViT-L / ViT-B 的策略；bench / profile / gate_s / graph_equiv / e2e；`--q2` 开关，默认关）、`run_final.py`（P3 训练入口）、`impl/presets23.py`、`tests/`（smoke、gate B、队列、nsys、roofline、`ro_run23.py`、表格 / 结果汇总） |
| 0. 冒烟（组合栈能否一起工作） | **通过**（两个模型）：两个 patch 同时应用，`opt_step` 包装链 = r22 `RollingOptStep` → r21 闭包 → FINAL 的 r15 步；两个 `FP4State` 每步各 roll 一次（graph 和 eager）；loss 有限（§4.1） |
| 1. 整步速度（graph，三相加权，同场次交替；R0 同机重测） | **ViT-L：R0 16.25 / FINAL 68.43 / P3H 76.49 / P3B 72.46 / P3 81.87 img/s；ViT-B：37.82 / 159.24 / 182.76 / 165.67 / 191.40 img/s。P3 对 FINAL：×1.196 / ×1.202；对 R0：×5.04 / ×5.06**（§4.2）；≥2% 可发布 |
| 2a. 门 A | 通过（R21b、R22 的 Gate A 脚本重跑，0 失败；叠加后的证据见 §4.3） |
| 2b. 门 B（同进程对 R0 和 P1） | P3 共 40 批 / 模型（5 次独立抽批 × 8）：**ViT-L 对 R0 中位 / 最小 0.9921 / 0.9731，对 P1 0.9921 / 0.9693；ViT-B 0.9895 / 0.9749，对 P1 0.9898 / 0.9735。** ViT-B 放宽门过、严格门不过，**最差批余量只有 0.0035–0.0049**；ViT-L 放宽门过，只有一批 MXFP8 离群批（FINAL 自己 0.9643）使对 P1 的最小值 0.9693 低于 0.97（§4.4） |
| 2c. 门 S（StepGraphRunner） | 8 个场景全过：EMA 偏差对 R0 最大 0.41%（ViT-L）/ 0.67%（ViT-B），对 FINAL graph 最大 0.26% / 0.74%；grad-norm 比值 0.948–1.037；head 0 次饱和；ViT-L 骨干 dY 在 schedule / compressed 有 1.3–1.4% 的调用超限（最大 49×）（§4.5） |
| 2d. graph_equiv | 8 个场景全过，两个 state 各前进 8 步（§4.6） |
| 2e. 导出 / resume | 两个模型都 **pass**（§4.7） |
| 3. 训练入口 `run_final.py` | 20 步 + val + ckpt + export 跑通；`check_export` pass；`load_model` 可用；原版（R0）路径 resume 一步 loss 有限；顺带修了 r19 `run_final.py --persistent` 存 checkpoint 会失败的问题（只在 r23 入口里绕过） |
| 4. e2e / nsys / roofline | e2e 稳态 / GPU-only：ViT-L **99.1%**，ViT-B **98.2%**（§4.8）；nsys P3 对 FINAL kernel 总时间 L −16.0 / B −13.5 ms（§4.9）；剩余瓶颈：attention、FP4 / MXFP8 量化、ViT-B 的 r14 head 融合 kernel、ViT-L 的优化器 |
| 5. 可选 ⑤a qkv 前向 MXFP8（`P3_q8`，只记录） | 做了 bench / S / B（各 n=16）：整步再 +3.7%（ViT-L）/ +2.3%（ViT-B），梯度余弦再降中位约 0.002–0.0035，ViT-B 最差批对 P1 只剩 0.0028 余量；**不发布**，由用户决定（§4.10） |

## 1. 目标与假设

| 假设 | 结果 |
|---|---|
| 两个 patch 覆盖不同的 op（`predict` / `block`），各持一个 `FP4State`，各包一层 `ops.opt_step`，可以叠加 | 成立（冒烟：两个 state 每步各 roll 一次，graph 与 eager 都是；包装链顺序正确），见 §4.1 |
| 整步收益近似相乘：R21b ×1.119 / ×1.147，R22 ×1.062 / ×1.034（各自对 FINAL） | 成立：P3 ×1.196 / ×1.202（乘积 ×1.184 / ×1.194），见 §4.2 |
| 组合后 B 门最差批余量（ViT-B 骨干 FP4 最差批只比放宽线高 ~0.005，ViT-L seed 13 最小 0.9675） | ViT-B 40 批最差 P3 0.9749（对 R0）/ 0.9735（对 P1），余量 0.0049 / 0.0035，放宽门过、严格门不过；ViT-L seed 13 在本轮无法复现，用同类离群批代替：P3 0.9731 / 0.9693（FINAL 自己 0.9643 / 0.9690），见 §4.4 |

## 2. 改动（全部在 `train-opt/round23_fp4_final/`）

| 文件 | 内容 |
|---|---|
| `impl/presets23.py` | 预设：`P3` = `common.config.P3_STACK` + `P3_PATCH_KWARGS_VITL/_VITB`（按模型选）；`P3H`、`P3B`、`P3_RN`（head 与骨干的 SR 都换 RN，只作对照）、`P3_strict`（ViT-B = P3H；ViT-L 没有更严格的档，就是 P3）；`maybe_q2()`：`--q2` 时在 round21 / round22 被 import 之前用 `sys.modules` 把 `round21_fp4_quant_head.impl.fp4q` 换成 `round24_fp4_quant_opt.impl.fp4q2`（默认关，本轮所有数字都是 R21a 量化器）；`opt_step_chain()`：检查包装链 |
| `run.py` | r19 `run.py` 改写：bench（同时记 reserved 显存）、profile、gate_s（`--stack` 在 graph 下对 `--ref` R0 eager，同时对 FINAL graph；两个 `FP4State` 的饱和统计）、graph_equiv（两个 FP4 state 在预热后快照、每次运行前原地恢复，并检查两个 state 各前进 K 步）、e2e |
| `run_final.py` | P3 训练入口（模板 r19 `run_final.py`；loader / graph runner / train loop / val+ckpt+export 以及 r15 的 `fix_scheduler_state` 绕过都是 import r19 的，没有改）。默认 `--stack P3 --workers 16 --persistent` |
| `tests/smoke_p3.py` | 冒烟 |
| `tests/gate_b_p3.py`、`gate_b_p3_summary.py` | 门 B：同进程 R0 / P1 / FINAL / P3H / P3B / P3（一个 P3_STACK 上下文里用 `set_policy` 切候选），同权重、同 batch、同 RngTape |
| `tests/ro_run23.py` | 重跑 R21b / R22 的 Gate A 脚本，输出重定向到 `profile/gateA_<round>_*.json`（写、读、exists 都重定向，不碰它们的 profile） |
| `tests/queue_bench.sh` `queue_gates.sh` `queue_b3.sh` `queue_q8.sh` `queue_rn.sh` `retry.sh` | 排队（持双锁，`assert_gpu_exclusive` 走重试）：bench、门、第 3 次抽批、5a、RN 对照 |
| `tests/nsys_report23.py` `roofline23.py` `make_results.py` `make_tables.py` `fill_notes.py` | nsys 类别表（r19 分类器 + NVFP4 GEMM、FP4 量化两个新类别）、roofline、结果汇总 |

## 3. nsys 分析过程

方法：`tests/queue_gates.sh nsys`（r16 `nsys_inner.sh`：graph 模式，mid 相，5 个 graph 步，`--cuda-graph-trace=node`；FINAL 与 P3 同一天、独占 GPU、依次抓取，原始 `.nsys-rep` 在 `/root/autodl-tmp/train-opt-tmp/round23_fp4_final/nsys/`）。分类器 = r19 `nsys_final_report.py`（原样 import）+ 两个新类别：**NVFP4 GEMM**（MMA atom m16n8k64 或 `*_ts` epilogue 的 QuACK kernel）、**FP4 量化**（R21 的 `act_kernel` / `w_kernel` / amax roll）。看三件事：
1. P3 相对 FINAL 的类别增减是否等于 R21b + R22 各自类别增减之和（叠加有没有产生额外开销）：L −16.01 对 −15.40，B −13.45 对 −12.98，吻合（§4.9）；
2. 窗口与忙碌率：是否引入 host 间隙（没有，剩下的是 5 步窗口边界的一个间隙）；
3. 剩余的 kernel 时间花在哪、离 roofline 下界多远（§4.9 末尾）。

## 4. 实测结果

### 4.1 冒烟（`tests/smoke_p3.py`，`profile/smoke_vitl16_P3.json` / `smoke_vitb16_P3.json`）

| 检查 | ViT-L | ViT-B |
|---|---|---|
| 两个 patch 同时应用（`P3_STACK` = FINAL + r21 + r22），`ops.predict` 属于 round21、`ops.block` 属于 round22 | 是 | 是 |
| `ops.opt_step` 包装链（外 → 内）：r22 `RollingOptStep`（handle = r22 的）→ r21 闭包 `opt_step`（handle = r21 的）→ FINAL 的 r15 flat 优化器步 | 是 | 是 |
| 两个 `FP4State` 是不同对象；site 数 | head 18 / 骨干 288 | head 18 / 骨干 144 |
| StepGraphRunner 预热：head 18 个 site 全部 exact 初始化；骨干所有活动 site 初始化（ViT-L 的 nofirst 块没有 site） | 是 | 是 |
| graph 回放 6 步：两个 state 的 `step` 计数器每步各 +1 | 是 | 是 |
| eager 2 步（同一个 ctx，`train_step` 经过两层 `opt_step`）：各 +1 | 是 | 是 |
| 全部 loss 有限 | 是（graph 30.86 → 30.77，eager 30.65 / 30.61） | 是（graph 31.00 → 30.89，eager 30.78 / 30.72） |
| 饱和统计（8 步） | head 0 / 208，骨干 0 / 2944 | head 0 / 208，骨干 0 / 1536 |
| 峰值 reserved | 15.6 GB | 11.8 GB |

结论：**组合成功**，包装链顺序正确（r22 在外、r21 在中、FINAL 在内：先原函数，再 r21 的 `roll()`，再 r22 的 `roll()`）。

### 4.2 整步速度（`run.py --action bench --runner graph --phases all`，r16 `StepGraphRunner`，三相 0.2 / 0.6 / 0.2，warmup 20，50 × 3；每次一个进程；同一场次 FINAL / P3 / P3H / P3B 交替；R0 用 eager 同一 harness，同机重测）

<!-- TABLE:bench -->
| 模型 | 栈 | 次数 | 每次 weighted img/s | 均值 img/s | 均值 mid 步时 | early / late 步时 | 对 FINAL | 对 R0 | 峰值 reserved |
|---|---|---:|---|---:|---:|---|---:|---:|---:|
| ViT-L | R0（eager） | 1 | 16.25 | 16.25 | 430.47 ms | 431.17 / 431.24 | ×0.237 | 1.00 | 30.02 GB |
| ViT-L | FINAL | 6 | 68.99 / 68.18 / 68.13 / 68.40 / 68.46 / 68.40 | 68.43 | 102.42 ms | 102.28 / 101.97 | ×1.000 | 4.21 | 15.95 GB |
| ViT-L | P3H（head） | 1 | 76.49 | 76.49 | 91.61 ms | 91.89 / 90.85 | ×1.118 | 4.71 | 15.85 GB |
| ViT-L | P3B（骨干） | 1 | 72.46 | 72.46 | 96.67 ms | 96.79 / 96.21 | ×1.059 | 4.46 | 15.12 GB |
| ViT-L | **P3（head + 骨干）** | 4 | 82.03 / 81.76 / 81.70 / 82.00 | 81.87 | 85.64 ms | 85.72 / 84.85 | ×1.196 | 5.04 | 14.97 GB |
| ViT-L | P3_RN（RN 对照） | 2 | 83.21 / 83.52 | 83.37 | 84.14 ms | 84.07 / 83.34 | ×1.218 | 5.13 | 14.97 GB |
| ViT-L | P3_q8（5a：P3 + qkv 前向 MXFP8，只记录） | 2 | 85.20 / 84.56 | 84.88 | 82.52 ms | 82.71 / 82.07 | ×1.240 | 5.22 | 15.07 GB |
| ViT-B | R0（eager） | 1 | 37.82 | 37.82 | 370.59 ms | 369.16 / 369.86 | ×0.238 | 1.00 | 30.36 GB |
| ViT-B | FINAL | 6 | 159.26 / 159.05 / 159.25 / 159.04 / 159.82 / 159.02 | 159.24 | 88.09 ms | 88.76 / 86.57 | ×1.000 | 4.21 | 12.35 GB |
| ViT-B | P3H（head） | 1 | 182.76 | 182.76 | 76.78 ms | 77.65 / 75.04 | ×1.148 | 4.83 | 12.14 GB |
| ViT-B | P3B（骨干） | 1 | 165.67 | 165.67 | 84.71 ms | 85.52 / 82.88 | ×1.040 | 4.38 | 11.86 GB |
| ViT-B | **P3（head + 骨干）** | 4 | 191.47 / 191.27 / 191.34 / 191.51 | 191.40 | 73.25 ms | 74.43 / 71.53 | ×1.202 | 5.06 | 11.59 GB |
| ViT-B | P3_RN（RN 对照） | 2 | 194.67 / 194.57 | 194.62 | 72.07 ms | 73.06 / 70.41 | ×1.222 | 5.15 | 11.59 GB |
| ViT-B | P3_q8（5a：P3 + qkv 前向 MXFP8，只记录） | 2 | 195.77 / 195.69 | 195.73 | 71.66 ms | 72.70 / 69.95 | ×1.229 | 5.18 | 11.62 GB |
<!-- /TABLE:bench -->

- 漂移（同一机器、同一天、各进程之间）：FINAL 6 次 ViT-L 68.13–68.99（1.3%）、ViT-B 159.02–159.82（0.5%）；P3 4 次 ViT-L 81.70–82.03（0.4%）、ViT-B 191.27–191.51（0.13%）。倍数用本轮 FINAL 的均值。
- **P3 对 FINAL：ViT-L ×1.196（mid −16.8 ms），ViT-B ×1.202（mid −14.8 ms），都远超 2% 的发布线。**
- 叠加几乎是相乘：ViT-L（对 FINAL）head ×1.118、骨干 ×1.059，乘积 ×1.184（实测 ×1.196，比乘积还高 1.0%）；ViT-B head ×1.148、骨干 ×1.040，乘积 ×1.194（实测 ×1.202）。R21b / R22 各自 NOTES 的数字：ViT-L ×1.119 / ×1.062，ViT-B ×1.147 / ×1.034，同量级。
- 对 R0（本轮同机重测，eager，同一 harness）：FINAL ×4.21 / ×4.21（r19：4.22 / 4.29，ViT-B 的差别来自 R0 重测的 37.82 img/s），P3 ×5.04 / ×5.06。R0 的 reserved 是 eager fp32 的 30 GB，各 graph 栈 reserved 都在 11–16 GB；P3 比 FINAL 少 1.0 / 0.8 GB。
- P3_RN（head 和骨干都用 RN 代替 SR，只作对照）：比 P3 快 1.8%（ViT-L 83.37 对 81.87）/ 1.7%（ViT-B 194.62 对 191.40），与 R22 单独测的 SR 代价（ViT-L 0.9%、ViT-B 1.4%）加上 head 的 SR 代价同量级；本轮没有为 RN 单独跑 B 门，契约选 SR，不改。
- P3H / P3B / R0 各只跑了 1 次（对照臂）；`P3_q8` 见 §4.10。注：`peak_reserved_gb` 是整个进程（含建模、预热、graph 池）的 `max_memory_reserved`，不是稳态的显存占用。

### 4.3 门 A（复用 R21b 的 `tests/test_head_fp4.py` 和 R22 的 `tests/test_block22.py`，通过 `tests/ro_run23.py` 重跑，输出重定向到 `profile/gateA_round21_gateA_head_fp4.json`、`profile/gateA_round22_test_block22_{vitb16,vitl16}.json`；它们自己的 profile 没动）

| 脚本 | 内容 | 结果 |
|---|---|---|
| R21b `test_head_fp4.py --part all`（head） | gemm（16 例）+ head RN 对 R20b 仿真 head（12 组，logits / 梯度逐位）+ bf16 恒等（4 组）+ SR 统计 + graph（2 例：预热 → 捕获 → 重放对 eager 逐位） | **0 失败**（`all_pass = true`） |
| R22 `test_block22.py --model vitb16` / `vitl16`（骨干） | a GEMM 层（34 + 2 例）、b RN 块对 R20b 仿真块（20 例）、c SR 统计一致性、d nofirst / 空策略 = FINAL、捕获内首次调用报错、`opt_step` 包装每步 roll 恰好 1 次 | **0 失败**（两个模型的 `a_failures` / `b_failures` / `c_failures` 都是 0；47 条 ok 记录全为 true） |

说明：这两个脚本测的是各自的组件，结果与 R21b / R22 当时一致，**单独不能证明叠加后不被破坏**。叠加后的证据是：(1) 冒烟：两个 state 各 roll 一次、包装链正确（§4.1）；(2) 门 B 里 P3 的 head ConvT 梯度对 P3H（只有 head 的栈）的余弦逐批最小 1.0000，即骨干 FP4 不改变 head 的梯度；(3) graph_equiv（§4.6）与门 S（§4.5）都在叠加栈上通过。


### 4.4 门 B（`tests/gate_b_p3.py`；同一进程内 R0 / P1 / FINAL / P3H / P3B / P3，同权重、同 batch、同 RngTape；候选在一个 `P3_STACK` 上下文里用 `set_policy` 切换，每个候选新建 `FP4State`，第一次调用 exact 初始化、之后 headroom × 上一步 amax，等于 graph 里一步看到的量化条件）
<!-- TABLE:gate_b -->
**ViT-L**（draw 数 3，批数 draw1: 8 + draw2: 8 + draw3: 8）

| 候选 | 抽批 | n | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 / 最小 | 最小值所在批 |
|---|---|---:|---|---|---:|---|---|---|---|
| FINAL | draw1 | 8 | 0.9967 / 0.9954 | 0.9964 / 0.9953 | 2.5e-03 | 过 | 过 | 1.0000 / 1.0000 | draw1:7 |
| FINAL | draw2 | 8 | 0.9962 / 0.9885 | 0.9968 / 0.9911 | 2.7e-03 | 过 | 过 | 1.0000 / 1.0000 | draw2:13 |
| FINAL | draw3 | 8 | 0.9929 / 0.9854 | 0.9938 / 0.9855 | 3.7e-03 | 过 | 过 | 1.0000 / 1.0000 | draw3:17 |
| FINAL | 合并 | 24 | 0.9960 / 0.9854 | 0.9963 / 0.9855 | 3.7e-03 | 过 | 过 | 1.0000 / 1.0000 | draw3:17 |
| P3H | draw1 | 8 | 0.9938 / 0.9850 | 0.9938 / 0.9850 | 2.3e-03 | 过 | 过 | 0.9952 / 0.9876 | draw1:6 |
| P3H | draw2 | 8 | 0.9936 / 0.9887 | 0.9950 / 0.9902 | 2.9e-03 | 过 | 过 | 0.9971 / 0.9898 | draw2:13 |
| P3H | draw3 | 8 | 0.9895 / 0.9868 | 0.9901 / 0.9869 | 3.8e-03 | 过 | 不过 | 0.9966 / 0.9921 | draw3:17 |
| P3H | 合并 | 24 | 0.9931 / 0.9850 | 0.9936 / 0.9850 | 3.8e-03 | 过 | 过 | 0.9965 / 0.9876 | draw1:6 |
| P3B | draw1 | 8 | 0.9951 / 0.9935 | 0.9952 / 0.9935 | 2.5e-03 | 过 | 过 | 0.9989 / 0.9965 | draw1:6 |
| P3B | draw2 | 8 | 0.9952 / 0.9855 | 0.9958 / 0.9888 | 2.7e-03 | 过 | 过 | 0.9990 / 0.9969 | draw2:13 |
| P3B | draw3 | 8 | 0.9920 / 0.9846 | 0.9922 / 0.9846 | 3.7e-03 | 过 | 过 | 0.9990 / 0.9970 | draw3:17 |
| P3B | 合并 | 24 | 0.9945 / 0.9846 | 0.9950 / 0.9846 | 3.7e-03 | 过 | 过 | 0.9990 / 0.9965 | draw3:17 |
| P3 | draw1 | 8 | 0.9924 / 0.9823 | 0.9924 / 0.9823 | 2.3e-03 | 过 | 过 | 0.9941 / 0.9849 | draw1:6 |
| P3 | draw2 | 8 | 0.9926 / 0.9850 | 0.9941 / 0.9891 | 2.9e-03 | 过 | 过 | 0.9963 / 0.9891 | draw2:13 |
| P3 | draw3 | 8 | 0.9887 / 0.9858 | 0.9892 / 0.9860 | 3.8e-03 | 过 | 不过 | 0.9952 / 0.9910 | draw3:18 |
| P3 | 合并 | 24 | 0.9914 / 0.9823 | 0.9921 / 0.9823 | 3.8e-03 | 过 | 过 | 0.9952 / 0.9849 | draw1:6 |

ViT-L 逐批对 R0 的全局梯度余弦（FINAL_again 与 FINAL 的差 = 同栈重复运行噪声）：

| 批 | P1 | FINAL | P3H | P3B | P3 | FINAL_again |
|---|---:|---:|---:|---:|---:|---:|
| draw1:0 | 0.9971 | 0.9990 | 0.9942 | 0.9985 | 0.9937 | 0.9990 |
| draw1:1 | 0.9997 | 0.9969 | 0.9937 | 0.9955 | 0.9922 | 0.9969 |
| draw1:2 | 0.9996 | 0.9955 | 0.9912 | 0.9946 | 0.9903 | 0.9955 |
| draw1:3 | 0.9984 | 0.9984 | 0.9955 | 0.9978 | 0.9948 | 0.9984 |
| draw1:4 | 0.9999 | 0.9982 | 0.9969 | 0.9975 | 0.9963 | 0.9982 |
| draw1:5 | 0.9997 | 0.9958 | 0.9939 | 0.9944 | 0.9925 | 0.9957 |
| draw1:6 | 0.9999 | 0.9965 | 0.9850 | 0.9935 | 0.9823 | 0.9965 |
| draw1:7 | 0.9989 | 0.9954 | 0.9922 | 0.9942 | 0.9910 | 0.9954 |
| draw2:8 | 0.9963 | 0.9932 | 0.9914 | 0.9923 | 0.9905 | 0.9932 |
| draw2:9 | 0.9999 | 0.9966 | 0.9949 | 0.9954 | 0.9938 | 0.9966 |
| draw2:10 | 0.9998 | 0.9980 | 0.9952 | 0.9974 | 0.9946 | 0.9980 |
| draw2:11 | 1.0000 | 0.9911 | 0.9902 | 0.9905 | 0.9896 | 0.9911 |
| draw2:12 | 1.0000 | 0.9978 | 0.9963 | 0.9973 | 0.9957 | 0.9978 |
| draw2:13 | 0.9972 | 0.9885 | 0.9887 | 0.9855 | 0.9850 | 0.9885 |
| draw2:14 | 0.9978 | 0.9962 | 0.9943 | 0.9953 | 0.9934 | 0.9962 |
| draw2:15 | 1.0000 | 0.9962 | 0.9929 | 0.9951 | 0.9918 | 0.9962 |
| draw3:16 | 1.0000 | 0.9921 | 0.9900 | 0.9914 | 0.9892 | 0.9921 |
| draw3:17 | 1.0000 | 0.9854 | 0.9868 | 0.9846 | 0.9860 | 0.9854 |
| draw3:18 | 0.9996 | 0.9907 | 0.9870 | 0.9894 | 0.9858 | 0.9907 |
| draw3:19 | 0.9928 | 0.9904 | 0.9890 | 0.9895 | 0.9881 | 0.9904 |
| draw3:20 | 0.9994 | 0.9952 | 0.9932 | 0.9929 | 0.9906 | 0.9953 |
| draw3:21 | 0.9999 | 0.9965 | 0.9889 | 0.9955 | 0.9879 | 0.9965 |
| draw3:22 | 0.9996 | 0.9964 | 0.9948 | 0.9956 | 0.9941 | 0.9964 |
| draw3:23 | 0.9983 | 0.9938 | 0.9936 | 0.9927 | 0.9924 | 0.9938 |

组合一致性（P3 对 P3H / P3B 的梯度，逐批取最小 / 中位）：vs P3H 全局 0.9963，骨干 blocks 0.9811 / 0.9855，head ConvT 1.0000 / 1.0000；vs P3B 全局 0.9858，blocks 0.9912 / 0.9943，head ConvT 0.9865 / 0.9949

- ViT-L seed 13（draw2）P3：对 R0 0.9850，对 P1 0.9891，对 FINAL 0.9931
- ViT-L seed 13（draw2）P3B：对 R0 0.9855，对 P1 0.9888，对 FINAL 0.9969
- ViT-L seed 13（draw2）FINAL：对 R0 0.9885，对 P1 0.9916，对 FINAL 1.0000
- ViT-L seed 13（draw2）P3H：对 R0 0.9887，对 P1 0.9925，对 FINAL 0.9965

**ViT-B**（draw 数 3，批数 draw1: 8 + draw2: 8 + draw3: 8）

| 候选 | 抽批 | n | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 / 最小 | 最小值所在批 |
|---|---|---:|---|---|---:|---|---|---|---|
| FINAL | draw1 | 8 | 0.9948 / 0.9804 | 0.9950 / 0.9832 | 2.5e-03 | 过 | 过 | 1.0000 / 1.0000 | draw1:2 |
| FINAL | draw2 | 8 | 0.9944 / 0.9876 | 0.9947 / 0.9906 | 3.0e-03 | 过 | 过 | 1.0000 / 1.0000 | draw2:8 |
| FINAL | draw3 | 8 | 0.9949 / 0.9884 | 0.9952 / 0.9886 | 3.0e-03 | 过 | 过 | 1.0000 / 1.0000 | draw3:23 |
| FINAL | 合并 | 24 | 0.9947 / 0.9804 | 0.9950 / 0.9832 | 3.0e-03 | 过 | 过 | 1.0000 / 1.0000 | draw1:2 |
| P3H | draw1 | 8 | 0.9925 / 0.9874 | 0.9925 / 0.9868 | 2.8e-03 | 过 | 过 | 0.9970 / 0.9782 | draw1:5 |
| P3H | draw2 | 8 | 0.9907 / 0.9770 | 0.9914 / 0.9755 | 3.8e-03 | 过 | 不过 | 0.9948 / 0.9825 | draw2:8 |
| P3H | draw3 | 8 | 0.9925 / 0.9870 | 0.9929 / 0.9873 | 2.5e-03 | 过 | 过 | 0.9964 / 0.9943 | draw3:23 |
| P3H | 合并 | 24 | 0.9921 / 0.9770 | 0.9925 / 0.9755 | 3.8e-03 | 过 | 不过 | 0.9959 / 0.9782 | draw2:8 |
| P3B | draw1 | 8 | 0.9922 / 0.9769 | 0.9922 / 0.9796 | 2.5e-03 | 过 | 不过 | 0.9968 / 0.9960 | draw1:2 |
| P3B | draw2 | 8 | 0.9919 / 0.9856 | 0.9925 / 0.9885 | 3.0e-03 | 过 | 过 | 0.9972 / 0.9964 | draw2:8 |
| P3B | draw3 | 8 | 0.9921 / 0.9843 | 0.9923 / 0.9844 | 3.0e-03 | 过 | 过 | 0.9973 / 0.9955 | draw3:23 |
| P3B | 合并 | 24 | 0.9922 / 0.9769 | 0.9922 / 0.9796 | 3.0e-03 | 过 | 不过 | 0.9972 / 0.9955 | draw1:2 |
| P3 | draw1 | 8 | 0.9895 / 0.9848 | 0.9895 / 0.9832 | 2.8e-03 | 过 | 不过 | 0.9932 / 0.9756 | draw1:2 |
| P3 | draw2 | 8 | 0.9880 / 0.9749 | 0.9883 / 0.9735 | 3.8e-03 | 过 | 不过 | 0.9921 / 0.9801 | draw2:8 |
| P3 | draw3 | 8 | 0.9901 / 0.9829 | 0.9904 / 0.9831 | 2.5e-03 | 过 | 过 | 0.9931 / 0.9905 | draw3:23 |
| P3 | 合并 | 24 | 0.9895 / 0.9749 | 0.9899 / 0.9735 | 3.8e-03 | 过 | 不过 | 0.9927 / 0.9756 | draw2:8 |

ViT-B 逐批对 R0 的全局梯度余弦（FINAL_again 与 FINAL 的差 = 同栈重复运行噪声）：

| 批 | P1 | FINAL | P3H | P3B | P3 | FINAL_again |
|---|---:|---:|---:|---:|---:|---:|
| draw1:0 | 0.9997 | 0.9977 | 0.9962 | 0.9962 | 0.9947 | 0.9977 |
| draw1:1 | 0.9996 | 0.9955 | 0.9926 | 0.9925 | 0.9894 | 0.9955 |
| draw1:2 | 0.9954 | 0.9804 | 0.9883 | 0.9769 | 0.9848 | 0.9805 |
| draw1:3 | 0.9988 | 0.9946 | 0.9923 | 0.9921 | 0.9896 | 0.9946 |
| draw1:4 | 0.9982 | 0.9943 | 0.9910 | 0.9913 | 0.9880 | 0.9943 |
| draw1:5 | 0.9996 | 0.9920 | 0.9874 | 0.9897 | 0.9851 | 0.9920 |
| draw1:6 | 0.9994 | 0.9964 | 0.9954 | 0.9927 | 0.9917 | 0.9964 |
| draw1:7 | 0.9996 | 0.9949 | 0.9938 | 0.9922 | 0.9912 | 0.9949 |
| draw2:8 | 0.9967 | 0.9876 | 0.9770 | 0.9856 | 0.9749 | 0.9876 |
| draw2:9 | 0.9956 | 0.9940 | 0.9880 | 0.9914 | 0.9848 | 0.9938 |
| draw2:10 | 0.9985 | 0.9960 | 0.9930 | 0.9935 | 0.9905 | 0.9960 |
| draw2:11 | 0.9995 | 0.9952 | 0.9913 | 0.9926 | 0.9886 | 0.9952 |
| draw2:12 | 0.9996 | 0.9936 | 0.9789 | 0.9905 | 0.9755 | 0.9936 |
| draw2:13 | 0.9996 | 0.9964 | 0.9952 | 0.9947 | 0.9935 | 0.9964 |
| draw2:14 | 0.9999 | 0.9916 | 0.9915 | 0.9883 | 0.9882 | 0.9916 |
| draw2:15 | 0.9990 | 0.9947 | 0.9901 | 0.9925 | 0.9878 | 0.9947 |
| draw3:16 | 0.9991 | 0.9939 | 0.9911 | 0.9923 | 0.9894 | 0.9940 |
| draw3:17 | 0.9999 | 0.9976 | 0.9923 | 0.9952 | 0.9899 | 0.9976 |
| draw3:18 | 0.9985 | 0.9943 | 0.9927 | 0.9920 | 0.9903 | 0.9943 |
| draw3:19 | 0.9997 | 0.9953 | 0.9938 | 0.9917 | 0.9903 | 0.9953 |
| draw3:20 | 0.9997 | 0.9975 | 0.9958 | 0.9958 | 0.9939 | 0.9975 |
| draw3:21 | 0.9992 | 0.9951 | 0.9920 | 0.9925 | 0.9895 | 0.9951 |
| draw3:22 | 0.9992 | 0.9946 | 0.9934 | 0.9917 | 0.9903 | 0.9946 |
| draw3:23 | 0.9990 | 0.9884 | 0.9870 | 0.9843 | 0.9829 | 0.9884 |

组合一致性（P3 对 P3H / P3B 的梯度，逐批取最小 / 中位）：vs P3H 全局 0.9956，骨干 blocks 0.9697 / 0.9762，head ConvT 1.0000 / 1.0000；vs P3B 全局 0.9777，blocks 0.9838 / 0.9929，head ConvT 0.9730 / 0.9941

- ViT-B seed 13（draw2）P3：对 R0 0.9935，对 P1 0.9936，对 FINAL 0.9958
- ViT-B seed 13（draw2）P3B：对 R0 0.9947，对 P1 0.9947，对 FINAL 0.9982
- ViT-B seed 13（draw2）FINAL：对 R0 0.9964，对 P1 0.9964，对 FINAL 1.0000
- ViT-B seed 13（draw2）P3H：对 R0 0.9952，对 P1 0.9952，对 FINAL 0.9976
<!-- /TABLE:gate_b -->

要点（三次独立抽批各 8 批，n = 24 / 模型；批次按（进程, seed）计，**加载器跨进程不确定，所以 R22 / R21b 的 seed N 与本轮的 seed N 不是同一批**，`batch_sum` 已核对不同）：
- **ViT-L P3：24 批对 R0 中位 / 最小 0.9914 / 0.9823，对 P1 0.9921 / 0.9823，loss_rel ≤3.8e-3：放宽门通过；合并后严格门也通过**（中位 ≥0.99、最小 ≥0.98；但 draw3 单独的中位是 0.9887 / 0.9892，严格门单独不过，所以 ViT-L 对严格门的余量也不大）。FINAL 同 24 批 0.9960 / 0.9854。16 批（draw1 + draw2）时 0.9924 / 0.9823。最小值批 draw1:6：这批 P3H 单独已是 0.9850（对 P1）、P3B 0.9935，P3 0.9823。
- **ViT-B P3：24 批对 R0 中位 / 最小 0.9895 / 0.9749，对 P1 0.9899 / 0.9735：放宽门通过，严格门不过（最小 <0.98、中位 <0.99）。最小值离放宽线 0.97 只剩 0.0049（对 R0）/ 0.0035（对 P1），出现在 draw2 的 seed 8 和 seed 12（P3 0.9749 / 0.9755）。**三次抽批里只有 draw1（0.9848）和 draw3（0.9829）的最小值 >0.98，draw2 的两批是 0.975。这两批上 FINAL 自己是 0.9876 / 0.9936，P3H 0.9770 / 0.9789，P3B 0.9856 / 0.9905，所以这两批的掉分主要来自 head 的 FP4，骨干 FP4 再叠一点（P3 比 P3H 再低 0.002–0.003）。draw1 seed 2 的 FINAL 自己只有 0.9804。
- P3H（只做 head，ViT-B 的“严格档”）n=24：对 R0 0.9921 / 0.9770，对 P1 0.9925 / 0.9755：**draw2 里它也没过严格门（最小 <0.98）**，只过放宽门；draw1 / draw3 过严格门。也就是说在 n=24 上，“ViT-B 只做 head = 严格档”这一说法不成立：严格门对 ViT-B 的批次抽样很敏感（FINAL 自己 draw1 的最小也只有 0.9804）。n=24 都过严格门的只有 FINAL。P3B 在 ViT-B 上 draw1 最小 0.9769（对 R0）。
- 组合一致性（P3 对 P3H / P3B 的梯度，逐批取最小 / 中位）：P3 的 **head ConvT 梯度对 P3H 的余弦逐批最小 1.0000**（骨干 FP4 不改变 head 的梯度，因为 head 梯度只依赖前向的 X（前向不变）和 head 自己的量化）；P3 的骨干 blocks 梯度对 P3B 的余弦最小 0.991（ViT-L）/ 0.984（ViT-B）——head 的 FP4 噪声通过 dY 回传到骨干；P3 的 blocks 对 P3H 的余弦最小 0.981 / 0.970，中位 0.986 / 0.976（骨干 FP4 引入的差）。
<!-- TABLE:gate_b_extra -->
| 模型 | 候选 | 抽批 | n | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 / 最小 | 最小值所在批 |
|---|---|---|---:|---|---|---:|---|---|---|---|
| ViT-L | FINAL | q8 队列 2 次抽批 | 16 | 0.9960 / 0.9643 | 0.9957 / 0.9690 | 3.1e-03 | **不过** | 不过 | 1.0000 / 1.0000 | qdraw2:11 |
| ViT-L | P3 | q8 队列 2 次抽批 | 16 | 0.9933 / 0.9731 | 0.9909 / 0.9693 | 3.2e-03 | **不过** | 不过 | 0.9964 / 0.9870 | qdraw2:11 |
| ViT-L | P3_q8 | q8 队列 2 次抽批 | 16 | 0.9885 / 0.9640 | 0.9884 / 0.9682 | 5.8e-03 | **不过** | 不过 | 0.9898 / 0.9817 | qdraw2:11 |
| ViT-L | FINAL | 5 次抽批合并 | 40 | 0.9960 / 0.9643 | 0.9959 / 0.9690 | 3.7e-03 | **不过** | 不过 | 1.0000 / 1.0000 | qdraw2:11 |
| ViT-L | P3 | 5 次抽批合并 | 40 | 0.9921 / 0.9731 | 0.9921 / 0.9693 | 3.8e-03 | **不过** | 不过 | 0.9958 / 0.9849 | qdraw2:11 |
| ViT-B | FINAL | q8 队列 2 次抽批 | 16 | 0.9943 / 0.9877 | 0.9950 / 0.9877 | 3.7e-03 | 过 | 过 | 1.0000 / 1.0000 | qdraw1:2 |
| ViT-B | P3 | q8 队列 2 次抽批 | 16 | 0.9893 / 0.9797 | 0.9898 / 0.9769 | 3.8e-03 | 过 | 不过 | 0.9922 / 0.9854 | qdraw1:2 |
| ViT-B | P3_q8 | q8 队列 2 次抽批 | 16 | 0.9876 / 0.9760 | 0.9869 / 0.9728 | 5.6e-03 | 过 | 不过 | 0.9888 / 0.9786 | qdraw2:8 |
| ViT-B | FINAL | 5 次抽批合并 | 40 | 0.9946 / 0.9804 | 0.9950 / 0.9832 | 3.7e-03 | 过 | 过 | 1.0000 / 1.0000 | draw1:2 |
| ViT-B | P3 | 5 次抽批合并 | 40 | 0.9895 / 0.9749 | 0.9898 / 0.9735 | 3.8e-03 | 过 | 不过 | 0.9925 / 0.9756 | draw2:8 |

MXFP8 离群批（FINAL 自己对 R0 的全局梯度余弦 < 0.985 的批，全部 5 次抽批；P1 = bf16 对 R0）：

| 模型 | 批 | P1 | FINAL | P3H | P3B | P3 | P3_q8 |
|---|---|---:|---:|---:|---:|---:|---:|
| ViT-L | q8_draw2:11 | 0.9940 | 0.9643 | - | - | 0.9731 | 0.9640 |
| ViT-B | p3_draw1:2 | 0.9954 | 0.9804 | 0.9883 | 0.9769 | 0.9848 | - |
<!-- /TABLE:gate_b_extra -->

- **两个风险点（单独标出）**
  1. **ViT-B 骨干 FP4 的最差批**：P3 在 5 次抽批共 40 批上，对 R0 最小 **0.9749**（draw2 seed 8；对 R0 离放宽线 0.97 的余量 0.0049），对 P1 最小 **0.9735**（余量 **0.0035**）；P3 不过严格门（40 批里对 R0 低于 0.98 的有 3 批：0.9749、0.9755、0.9797）。没有一批低于放宽线 0.97，但余量只有 0.003–0.005。这 3 批上 FINAL 自己是 0.9876 / 0.9936 / 0.9877，下降主要来自 head 的 FP4（P3H 单独在 draw2:8 / draw2:12 已是 0.9770 / 0.9789），骨干 FP4 再叠加 0.002–0.003。
  2. **ViT-L 的 seed 13（R22 里最小 0.9675 的那一批）**：该批在本轮无法复现——数据加载器跨进程不确定（6b 13.5），本轮 draw2 的 seed 13 的 `batch_sum` 是 −3301363.649，R22 那批是 −2025063.509，FINAL 对 R0 分别是 0.9885 和 0.9715；本轮这个 seed 上 P3 对 R0 / P1 / FINAL 是 **0.9850 / 0.9891 / 0.9931**（P3B 0.9855 / 0.9888，P3H 0.9887 / 0.9925，FINAL 0.9885 / 0.9916）。作为替代，**5 次抽批共 40 批里出现了一个同类的 MXFP8 离群批**（q8 队列 draw2 的 seed 11：bf16 的 P1 对 R0 = 0.9940，但 FINAL 自己只有 0.9643，R22 的 seed 13 是 0.9715）：这一批上 **P3 对 R0 / P1 / FINAL = 0.9731 / 0.9693 / 0.9870**（P3_q8 0.9640）。也就是说在这种离群批上 P3 没有比 FINAL 更差（0.9731 对 0.9643，同批不同随机噪声），放宽线（对 R0 ≥0.97）P3 刚好过、对 P1（0.9693）差 0.0007 没过；而 FINAL 在这批上自己对 R0（0.9643）和 P1（0.9690）都不过。对 ViT-L，放宽门的“最小 ≥0.97”在 MXFP8 离群批上连 FINAL 都不满足，不是 FP4 引入的。
- 5 次抽批合并后 ViT-L P3：40 批中位 / 最小对 R0 0.9921 / 0.9731，对 P1 0.9921 / 0.9693（只有那个离群批，FINAL 自己也不过）；ViT-B P3：对 R0 0.9895 / 0.9749，对 P1 0.9898 / 0.9735，放宽门过、严格门不过。


### 4.5 门 S（`run.py --action gate_s`，P3 在 StepGraphRunner 下对 R0 eager，同时对 FINAL graph；4 场景 × 50 步 × 2 批；每个场景在同一进程里依次跑 R0 eager、P3 graph、FINAL graph）
<!-- TABLE:gate_s -->
| 模型 | 场景 | EMA 偏差（检查点）对 R0 | EMA 偏差（所有 50 步）对 R0 | 对 FINAL graph（检查点） | 门 | 有限 | grad-norm 比值范围（对 R0） | head 饱和（exceed / calls，全部 site） | 骨干 dY 饱和（exceed / calls，最大 cur/prev） |
|---|---|---:|---:|---:|---|---|---|---|---|
| ViT-L | schedule | 0.38% | 0.45% | 0.03% | 过 | True | 0.977–1.014 | 0 / 1300 | 122 / 9200 (max ratio 49.2) |
| ViT-L | compressed | 0.22% | 0.31% | 0.12% | 过 | True | 1.002–1.010 | 0 / 1300 | 132 / 9200 (max ratio 22.6) |
| ViT-L | fixed_mid | 0.35% | 0.35% | 0.15% | 过 | True | 0.974–1.018 | 0 / 1300 | 6 / 9200 (max ratio 10.7) |
| ViT-L | fixed_late | 0.41% | 0.43% | 0.26% | 过 | True | 0.959–1.021 | 0 / 1300 | 20 / 9200 (max ratio 17.9) |
| ViT-B | schedule | 0.36% | 0.38% | 0.46% | 过 | True | 0.989–1.037 | 0 / 1300 | 0 / 4800 (max ratio 7.8) |
| ViT-B | compressed | 0.60% | 0.64% | 0.49% | 过 | True | 0.966–1.023 | 0 / 1300 | 10 / 4800 (max ratio 13.0) |
| ViT-B | fixed_mid | 0.67% | 0.68% | 0.74% | 过 | True | 0.948–1.014 | 0 / 1300 | 0 / 4800 (max ratio 6.0) |
| ViT-B | fixed_late | 0.67% | 0.72% | 0.72% | 过 | True | 0.950–1.023 | 0 / 1300 | 2 / 4800 (max ratio 9.9) |
<!-- /TABLE:gate_s -->

- 门（EMA 偏差 ≤5%、无 NaN / Inf）全部通过，两个模型 × 4 场景：对 R0 最大 0.41%（ViT-L）/ 0.67%（ViT-B）；对 FINAL graph 最大 0.26% / 0.74%。grad-norm 比值 0.948–1.037（R0 = 1）。
- **饱和统计**（`FP4State.stats()`，两个 state 分别读取，graph 外）：head 的 18 个 site（X 列副本、dZ 行 / 列副本、权重）50 步里 **0 次超限**（每个场景 1300 次比较）；骨干超限全部发生在 **dY**（X 与权重：ViT-L / ViT-B 都是 0 次，唯一例外见下），ViT-L 在 schedule / compressed 场景 122 / 132 次（9200 次 dY 比较的 1.3% / 1.4%，最大 cur/prev 比值 49.2 / 22.6，集中在块 1、2 的 fc2 / proj / fc1 / qkv 的 dY），fixed_mid / fixed_late 只有 6 / 20 次；ViT-B 0–10 次（4800 次里，最大比值 6.0–13.0）。**与 R22 单独时比：ViT-L 的 schedule / compressed 超限率高约 4–7 倍（R22 报告 0.18% / 0.09%，最大比值 16.5），原因没有查清**（候选：head 换成 FP4 后回传给骨干的 dY 噪声更大；或本轮是 5 步窗口里同样的早期动态；两者没有分开测）。50 步里 EMA 偏差只有 0.2–0.45%，看不出后果；饱和 = 当前步 amax 超过 8 × 上一步 amax，那一步该 site 的 e4m3 block scale 在 448 处裁剪（只裁剪个别大元素）。长训练风险项，写进 §7。
- ViT-L 的 fixed_mid / fixed_late 里 `b23.proj.x.col`（最后一块 proj 的输入 X 的转置副本，headroom 2）有 24 / 50 次 cur > 2 × prev，最大比值 2.4：两个交替 batch 的 amax 在这个 site 上相差 >2 倍（测试用 2 个 batch 交替），不是训练动态；超限幅度只有 20%，影响可忽略。

### 4.6 graph_equiv（`run.py --action graph_equiv`，P3，8 步，graph 回放对 eager；两个 FP4 state 在预热后快照，每次 eager / graph 运行前原地恢复；预热后断言 head 的 18 个 site 和骨干所有活动 site 都已 exact 初始化）
<!-- TABLE:graph_equiv -->
| 模型 | 场景 | 更新余弦 graph-vs-eager | eager-vs-eager（噪声底） | loss 相对差 graph-vs-eager | eager-vs-eager | 第一步 loss 逐位 | 两个 state 各前进 8 步 |
|---|---|---:|---:|---:|---:|---|---|
| ViT-L | schedule | 0.999694 | 0.999835 | 2.1e-04 | 2.2e-04 | True | True |
| ViT-L | compressed | 0.999909 | 0.999903 | 7.5e-05 | 5.6e-05 | True | True |
| ViT-L | fixed_mid | 0.999977 | 0.999977 | 8.0e-05 | 7.9e-05 | True | True |
| ViT-L | fixed_late | 0.999986 | 0.999983 | 2.4e-05 | 3.0e-05 | True | True |
| ViT-B | schedule | 0.999984 | 0.999978 | 2.0e-04 | 1.0e-04 | True | True |
| ViT-B | compressed | 0.999939 | 0.999938 | 2.0e-04 | 1.7e-04 | True | True |
| ViT-B | fixed_mid | 0.999974 | 0.999988 | 6.6e-05 | 9.9e-05 | True | True |
| ViT-B | fixed_late | 0.999959 | 0.999959 | 1.7e-04 | 1.3e-04 | True | True |
<!-- /TABLE:graph_equiv -->

- graph 与 eager 的差别和 eager 两次之间的差别同一量级（r13 的 dx 原子累加不确定性 + FP4 的放大）；第一步 loss 逐位相同；**两个 `FP4State` 每个场景、每种运行方式（eager ×2、graph）都恰好前进 8 步**。

### 4.7 训练入口 `run_final.py`、导出和 resume（`tests/queue_gates.sh exp`）

`run_final.py --model M --stack P3 --steps 20 --val --save-ckpt DIR --workers 16 --persistent`：真实 Lightly DataLoader（16 个 persistent worker）+ StepGraphRunner 跑 20 个 graph 步（1 个 graph 捕获，0 次 eager 回退）→ 一次完整验证 → Lightly 自己的 `save_checkpoint` + `export_model` → 新进程里 r19 的 `tests/check_export.py`（对 R0 原版路径）。

| 项 | ViT-L | ViT-B |
|---|---|---|
| 20 步 + val + ckpt + export 跑通 | 是 | 是 |
| 训练 loss（第 20 步） | 30.17 | 28.80 |
| 验证 | 12.5 s，val_loss 26.15 | 8.7 s，val_loss 26.11（20 步的 mIoU 无意义） |
| ckpt / export 大小 | 3.52 / 1.18 GB | 1.03 / 0.35 GB |
| 包装链（run_final 里打印） | r22 `RollingOptStep` → r21 闭包 → FINAL | 同 |
| 两个 `FP4State` 的 `step` = 训练步数（20） | 是 | 是 |
| `check_export.py` | **pass** | **pass** |
| ckpt / export / 原版训练导出的 state_dict 键、形状、dtype 与 R0 一致 | 全相等（388 键） | 全相等（208 键） |
| ckpt 元信息 / 优化器结构 / scheduler 键 | 与 R0 相等 | 与 R0 相等 |
| 导出权重全有限；参数已更新 | 是；19 个参数变化（最大 2.7e-5） | 是；19 个（最大 3.9e-5） |
| `lightly_train.load_model` 加载并 predict 一张图 | 通过（[1008,1008]，`DINOv3EoMTSemanticSegmentation`） | 通过 |
| 在原版（R0）路径上 resume 一步 | `load_state_dict` 全匹配，loss 26.98 有限 | 全匹配，loss 27.37 有限 |

要点：
- **发现并修了 r19 `run_final.py` 的一个潜在 bug（只在 `--persistent` 下出现）**：checkpoint 的 state 里带着训练 DataLoader 对象，persistent worker 打开时 torch 把活的 `_MultiProcessingDataLoaderIter` 放在 `DataLoader._iterator` 里，`torch.save` 报 `NotImplementedError: ('{} cannot be pickled', '_MultiProcessingDataLoaderIter')`（r19 的 persistent 选项从没和 checkpoint 一起跑过）。r23 的 `run_final.py` 里 `validate_and_save` 在保存期间把 `_iterator` 从 loader（及 Fabric 包装）上摘下来、保存完放回去；另外失败时 `os._exit(1)`，避免 persistent worker 在解释器退出时把进程（连同 GPU 锁）挂住（第一次失败时确实挂住了，手动杀掉了自己的进程）。r19 的文件没动。
- `FP4State` 不在 checkpoint 里。resume 后每个 site 在第一次调用时用当前张量的 exact amax 重新初始化（与全新运行的预热步一样），SR 计数器重新开始；这只让 resume 后的第一步用 exact 的 per-tensor scale，不影响参数 / 优化器 / scheduler 的兼容性。`check_export.py` 的 resume 在原版路径上进行，没有 FP4。这是预期行为。
- `run_final.py` 也可以直接 resume 之外的整训练：`--steps 0 --epochs 20 --val --save-ckpt DIR`（整个调度，每个 epoch 末验证 + checkpoint；本轮没有跑，只跑了 20 步）。

### 4.8 e2e（`run.py --action e2e`，真实 Lightly DataLoader + StepGraphRunner，200 步，`workers=16 prefetch=2 pin_memory persistent_workers`，`ulimit -n 65536`；同一进程里先测 GPU-only（预载 batch，early 相，50 × 3），再跑真实 loader）

| 模型 | GPU-only（early，预载 batch） | e2e 稳态（第 21–200 步） | 稳态 / GPU-only | 取 batch 等待（mean / p95 / max） | 200 步整段（含 worker 启动和第一个 batch） |
|---|---:|---:|---:|---|---:|
| ViT-L B=7 | 81.93 img/s（85.43 ms） | 81.17 img/s（86.24 ms） | **99.1%** | 9.7 / 15.4 / 534.8 ms | 75.42 img/s（第一个 batch 1.32 s） |
| ViT-B B=14 | 190.22 img/s（73.60 ms） | 186.77 img/s（74.96 ms） | **98.2%** | 10.7 / 20.6 / 31.8 ms | 164.24 img/s（第一个 batch 2.05 s） |

- 两个模型都 ≥97%（目标线）。ViT-B 稳态比 GPU-only 慢 1.4 ms / 步：16 个 worker 在 B=14、P3 的 74 ms 步时刚好够用，余量比 FINAL 小（步时短了 15 ms，loader 吞吐不变）；ViT-L 余量充足。`run_final.py` 的默认就是 `--workers 16 --persistent`。
- ViT-L 的最大等待 535 ms 是偶发（等待均值 9.7 ms，p95 15.4 ms，稳态 99.1% 说明被 prefetch 吸收）。


### 4.9 nsys 与 roofline（FINAL 与 P3，graph，mid 相，5 步；`profile/nsys_report23.txt` / `.json`）

kernel 类别表（ms / 步）：

| 类别 | ViT-L FINAL | ViT-L P3 | 差 | ViT-B FINAL | ViT-B P3 | 差 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM MXFP8（骨干前向 + 其余；P3 里还含 head ConvT 前向） | 24.68 | 9.65 | −15.03 | 14.61 | 6.59 | −8.02 |
| **GEMM NVFP4（骨干 dgrad / wgrad + head ConvT dgrad / wgrad）** | 0 | 10.56 | +10.56 | 0 | 7.21 | +7.21 |
| **FP4 量化（`act_kernel` / `w_kernel` / amax roll）** | 0 | 8.50 | +8.50 | 0 | 7.58 | +7.58 |
| MXFP8 量化（r17 `quant_dual`） | 5.07 | 1.73 | −3.34 | 4.09 | 1.35 | −2.74 |
| GEMM bf16 head（QuACK + cutlass sm80：ConvT 前向 / dgrad / wgrad、einsum、mask head） | 19.09 | 3.10 | −15.99 | 21.65 | 4.20 | −17.45 |
| GEMM bf16 骨干（qkv 前向） | 4.81 | 4.79 | −0.02 | 2.72 | 2.69 | −0.03 |
| attention（flash fwd / bwd + RoPE） | 16.13 | 16.06 | −0.07 | 12.76 | 12.69 | −0.07 |
| r14 head 融合 kernel（GELU + dwconv + LN2D） | 6.65 | 6.73 | +0.08 | 10.32 | 10.37 | +0.05 |
| 优化器（r15 flat） | 7.72 | 7.72 | 0 | 2.25 | 2.25 | 0 |
| LN / row kernels | 5.89 | 5.47 | −0.42 | 5.03 | 5.08 | +0.05 |
| 逐元素 / copy / 其他 | 6.92 | 6.64 | −0.28 | 7.74 | 7.73 | −0.01 |
| loss / matcher、cuDNN conv | 2.21 | 2.21 | 0 | 4.28 | 4.27 | 0 |
| **kernel 总时间 / 步** | **99.17** | **83.16** | **−16.01** | **85.45** | **72.00** | **−13.45** |
| kernel 数 / 步 | 2475 | 2860 | +385 | 1641 | 1861 | +220 |
| 窗口 / 步，GPU 忙碌率 | 100.87 ms，0.982 | 85.46 ms，0.972 | −15.41 ms | 86.66 ms，0.984 | 73.36 ms，0.980 | −13.30 ms |

- P3 的 kernel 总时间 L −16.0 / B −13.5 ms，与 bench 的 mid 步时差（L −16.8 / B −14.9 ms）一致（bench 多出的 0.8 / 1.4 ms 是 5 步窗口边界间隙与 150 步均值的统计差别）。
- 与分开的两轮对照：R21b（head）nsys kernel 总时间 L −10.41 / B −10.36，R22（骨干）L −4.99 / B −2.62，和 L −15.40 / B −12.98；本轮叠加 L −16.01 / B −13.45，比和多 0.6 / 0.5 ms（不同日期两次抓取的噪声内）——叠加没有产生新的开销或冲突。
- 忙碌率：P3 L 0.972（FINAL 0.982）。窗口里有 1 个 >20 µs 的间隙 3.75 ms（FINAL 2.07 ms），在 5 步窗口的起始边界（R22 同样观察到），其余 >20 µs 的间隙总计 L 0.79 ms / B 0.43 ms；kernel 数 +385 / +220 的 launch 开销在 graph 里可忽略。
- FP4 + MXFP8 量化合计 L 10.23 / B 8.93 ms（占 kernel 时间 12.3% / 12.4%）。`act_kernel`（dY 行 + 列副本带 SR）单次平均 L 58.6 µs（100 次）/ B 98.0 µs（56 次），是单个最大的 FP4 项。

剩余 kernel 时间的分解与 roofline（方法 = r19 `roofline_final.py` 扩展到 FP4：由 FLOP / 字节数与假设峰值算出的下界，**估计，不是更快 kernel 的测量**；假设 BF16 230、MXFP8 600、NVFP4 持续 1.2 PFLOPS（峰值 2.03）、DRAM 1.55 TB/s；量化下界按冷输入的首次访问字节算，偏松，因为输入多半在 L2 里）：

ViT-L（P3）
| part | measured ms | % of kernel time | analytic floor ms | basis |
|---|---:|---:|---:|---|
| attention (flash fwd/bwd + RoPE) | 16.06 | 19.3 | 11.37 | BF16 230 TFLOPS, RoPE excluded from the floor |
| GEMM NVFP4 (backbone dgrad / wgrad + head ConvT dgrad / wgrad) | 10.56 | 12.7 | 9.04 | NVFP4 1.2 PFLOPS sustained (peak 2.03 PFLOPS would be 5.3 ms) |
| FP4 + MXFP8 quantization (act_kernel / w_kernel / quant_dual) | 10.23 | 12.3 | 7.54 | DRAM 1.55 TB/s on cold first-touch bytes (loose: inputs mostly in L2) |
| GEMM MXFP8 (backbone proj / fc1 / fc2 forward + head ConvT forward) | 9.65 | 11.6 | 8.12 | MXFP8 600 TFLOPS |
| GEMM bf16 (qkv forward, einsum, mask head) | 7.89 | 9.5 | 6.24 | BF16 230 TFLOPS |
| optimizer (r15) | 7.72 | 9.3 | 7.32 | DRAM 1.55 TB/s x 36 B/float |
| r14 fused head kernels (GELU+dwconv+LN2D) | 6.73 | 8.1 | - | DRAM-bound |
| elementwise / copy / other | 6.64 | 8.0 | - | DRAM-bound |
| LN / row kernels | 5.47 | 6.6 | - | DRAM-bound |
| loss / matcher / mask build | 1.50 | 1.8 | - |  |
| cuDNN conv (patch embed) | 0.71 | 0.9 | - |  |
backbone Linear 13.2 TFLOP/step (bf16 1.1, MXFP8 3.7, NVFP4 8.4); attention 2.62 TFLOP; head ConvT 1.2 TFLOP MXFP8 + 2.4 TFLOP NVFP4, bf16 einsum / mask head 0.3 TFLOP; quantizer cold bytes 11.7 GB

ViT-B（P3）
| part | measured ms | % of kernel time | analytic floor ms | basis |
|---|---:|---:|---:|---|
| attention (flash fwd/bwd + RoPE) | 12.69 | 17.6 | 8.74 | BF16 230 TFLOPS, RoPE excluded from the floor |
| r14 fused head kernels (GELU+dwconv+LN2D) | 10.37 | 14.4 | - | DRAM-bound |
| FP4 + MXFP8 quantization (act_kernel / w_kernel / quant_dual) | 8.93 | 12.4 | 5.16 | DRAM 1.55 TB/s on cold first-touch bytes (loose: inputs mostly in L2) |
| elementwise / copy / other | 7.73 | 10.7 | - | DRAM-bound |
| GEMM NVFP4 (backbone dgrad / wgrad + head ConvT dgrad / wgrad) | 7.21 | 10.0 | 6.43 | NVFP4 1.2 PFLOPS sustained (peak 2.03 PFLOPS would be 3.8 ms) |
| GEMM bf16 (qkv forward, einsum, mask head) | 6.89 | 9.6 | 4.82 | BF16 230 TFLOPS |
| GEMM MXFP8 (backbone proj / fc1 / fc2 forward + head ConvT forward) | 6.59 | 9.1 | 5.39 | MXFP8 600 TFLOPS |
| LN / row kernels | 5.08 | 7.1 | - | DRAM-bound |
| loss / matcher / mask build | 3.17 | 4.4 | - |  |
| optimizer (r15) | 2.25 | 3.1 | 2.02 | DRAM 1.55 TB/s x 36 B/float |
| cuDNN conv (patch embed) | 1.10 | 1.5 | - |  |
backbone Linear 7.5 TFLOP/step (bf16 0.6, MXFP8 1.9, NVFP4 5.0); attention 2.01 TFLOP; head ConvT 1.4 TFLOP MXFP8 + 2.7 TFLOP NVFP4, bf16 einsum / mask head 0.5 TFLOP; quantizer cold bytes 8.0 GB

读法：
- **下一个最大的块是 attention**（L 16.1 / B 12.7 ms，占 19% / 18%；离 BF16 FLOP 下界还有 1.4× / 1.45×，flash-attn 2 的 sm_120 实现；没有成熟的 FP4 / FP8 attention 训练配方，不在本阶段范围）。
- **量化**（FP4 + MXFP8，L 10.2 / B 8.9 ms，12%）离冷输入下界 7.5 / 5.2 ms 还有 1.4× / 1.7×（暖输入时下界更低）：R24 在优化的就是这一项。
- NVFP4 GEMM 已接近 1.2 PFLOPS 的持续假设（L 10.6 对 9.0 ms，B 7.2 对 6.4 ms）；MXFP8 GEMM（含 head ConvT 前向）9.7 / 6.6 ms 对 8.1 / 5.4 ms；bf16 GEMM 7.9 / 6.9 ms（qkv 前向 4.8 / 2.7 ms + head einsum / mask head）对 6.2 / 4.8 ms。
- ViT-B 上 **r14 head 融合 kernel（10.4 ms，14%）比 NVFP4 GEMM 还大**，是内存受限的 elementwise（R21b 的建议：`bwd1` + `bwd2` 合并）；ViT-L 上是优化器（`update_kernel` 6.65 ms 单 kernel，接近 DRAM 下界）。


### 4.10 可选 ⑤a：`P3_q8`（qkv 前向改用 MXFP8；只记录，不发布）

**这是什么**：`P3_q8` 是我给任务书 §5「可选 ⑤a：qkv 前向用 mx8」起的候选名，不是新增的任务。任务书写的条件是「只有当 R22 的块已经支持这个策略、不需要新写 kernel 时才做；做法是 B 门 ≥16 批、S 门 4 个场景、整步 bench；**只记录，不发布**，由用户决定」。依据是 Amendment 6b 13.2：⑤a 在 R20b 的 n=3 放宽门能过，但按事先规则（11.6）速度和节省阈值不够，不单独发布，「R23 如果有时间，可以作为可选档补做 n≥8 的 B 门和 S 门」。

**定义**：`P3` 的骨干策略后面加 `+qkv.f=mx8`（预设 `P3_q8`，`impl/presets23.py`），head 策略不变：
- ViT-L：`all.d=nv4+sr+all.w=nv4+sr+nofirst+qkv.f=mx8`；ViT-B：`all.d=nv4+sr+all.w=nv4+sr+qkv.f=mx8`。
- R22 的块支持（`Handle.set_policy` 对前向格式 `f` 允许 `bf16` / `mx8`；策略解析核对过：ViT-L 的块 0 因 `nofirst` 仍是 FINAL 的 bf16 qkv 前向，其余块 `ff = mx8`），qkv 前向走 r17 的 MXFP8 路径（`mx8.quantize` + `gemm_mx8.fwd_bias`），**没有写任何新 kernel**。FINAL 的 qkv 前向是 bf16（`all.d` / `all.w` 的 dgrad / wgrad 已经是 FP4，前向 X 的 MXFP8 行副本由 r17 的量化器产出）。

**为什么做**：qkv 前向 bf16 GEMM 占 kernel 时间 L 4.8 / B 2.7 ms（§4.9），换成 MXFP8 预计省 2.5–3 / 1.5 ms（PLAN_v3 §2），代价是前向多一次量化噪声（r17 曾观察到前向量化让全局梯度余弦掉 0.002–0.007）。

**结果**（同一套 harness 和门；候选只跑了 bench / S / B，没跑 graph_equiv、导出 / resume、e2e、nsys）：

整步（graph，三相加权，与 FINAL / P3 同场次交替，见 §4.2 表）：

| 模型 | P3 img/s | P3_q8 img/s（2 次） | P3_q8 对 P3 | P3_q8 对 FINAL | P3_q8 mid 步时（P3） |
|---|---:|---:|---:|---:|---:|
| ViT-L | 81.83 | 84.88（85.20 / 84.56） | +3.7% | ×1.240 | 82.52 ms（85.68 ms） |
| ViT-B | 191.36 | 195.73（195.77 / 195.69） | +2.3% | ×1.230 | 71.66 ms（73.27 ms） |

门 S（P3_q8 在 StepGraphRunner 下对 R0 eager，4 场景 × 50 步 × 2 批；`profile/gateS_*_P3_q8_graph_vs_R0*.json`）：8 个场景全过，无 NaN；EMA 偏差（检查点）ViT-L 0.11–0.43%、ViT-B 0.46–0.62%；grad-norm 比值 ViT-L 0.977–1.036，**ViT-B 0.844–1.110**（P3 是 0.948–1.037；compressed 场景最低 0.844，fixed_mid 最高 1.110）；骨干 dY 饱和：ViT-L 10 / 16 / 26 / 58 次 / 9200（0.1–0.6%；P3 的 schedule / compressed 是 122 / 132 次，不同运行的随机性很大），ViT-B 0 次 / 4800。

门 B（P3_q8 与 P3 / FINAL 同进程，同权重同 batch 同 RngTape，另外 2 次独立抽批 × 8 批 = 16 批 / 模型；`profile/gateB_q8_*_draw{1,2}.json`；批与 §4.4 的三次抽批不是同一批）：

| 模型 | 候选 | n | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 / 最小 |
|---|---|---:|---|---|---:|---|---|---|
| ViT-L | FINAL | 16 | 0.9960 / 0.9643 | 0.9957 / 0.9690 | 3.1e-3 | **不过**（离群批） | 不过 | 1 |
| ViT-L | P3 | 16 | 0.9933 / 0.9731 | 0.9909 / 0.9693 | 3.2e-3 | **不过**（只因同一个离群批，对 P1 最小 0.9693） | 不过 | 0.9964 / 0.9870 |
| ViT-L | **P3_q8** | 16 | 0.9885 / 0.9640 | 0.9884 / 0.9682 | 5.8e-3 | **不过**（同一个离群批） | 不过 | 0.9898 / 0.9817 |
| ViT-B | FINAL | 16 | 0.9943 / 0.9877 | 0.9950 / 0.9877 | 3.7e-3 | 过 | 过 | 1 |
| ViT-B | P3 | 16 | 0.9893 / 0.9797 | 0.9898 / 0.9769 | 3.8e-3 | 过 | 不过 | 0.9922 / 0.9854 |
| ViT-B | **P3_q8** | 16 | 0.9876 / 0.9760 | 0.9869 / 0.9728 | 5.6e-3 | 过（最小对 P1 只比 0.97 高 **0.0028**） | 不过 | 0.9888 / 0.9786 |

- 逐批比较：P3_q8 比 P3 的对 R0 梯度余弦低（P3 − P3_q8）：ViT-L −0.011…+0.010（中位 0.0035），ViT-B −0.003…+0.007（中位 0.0023）；loss_rel 最大从 3.2e-3 / 3.8e-3 升到 5.8e-3 / 5.6e-3（仍远低于 2e-2）。
- ViT-L 这 16 批里有一批 MXFP8 离群批（`qdraw2:11`：bf16 的 P1 对 R0 = 0.9940，FINAL 自己 0.9643，P3 0.9731，P3_q8 0.9640）；FINAL 在这批上就已经不满足放宽门（最小 ≥0.97），所以表里 FINAL / P3 / P3_q8 的“放宽门”都是“不过”，不是 FP4 的责任。去掉这一批，P3_q8 的最小是 0.9800（对 R0）。
- **结论（只记录，由用户决定）**：qkv 前向 MXFP8 再给整步 +3.7%（ViT-L）/ +2.3%（ViT-B），都 ≥2%；代价：梯度余弦中位再降约 0.002–0.0035，ViT-B 的最差批对 P1 只比放宽线高 0.0028（同 16 批里 P3 是 0.0069；P3 在 5 次抽批共 40 批里是 0.0035，见 §4.4）。按 6b 13.2 的事先规则，它不单独发布；作为叠加在 P3 上的可选档，ViT-L 的精度代价小、速度收益 3.7%，ViT-B 的最差批余量最小，**我不建议在 ViT-B 上用**（`P3_q8` 的 ViT-B 最差批已经只剩 0.003 的余量，而且 S 门 grad-norm 比值 ViT-B 出现 0.844）。没有跑 graph_equiv / 导出 / resume / e2e / nsys，真要发布需要补。

## 5. 正确性

- **包装链与状态机**：`P3_STACK` 的应用顺序 = FINAL → r21 → r22。`ctx.ops.opt_step` 的链（外 → 内）= r22 `RollingOptStep`（handle = r22 的）→ r21 的闭包 `opt_step`（handle = r21 的）→ FINAL 的 r15 flat 步：每一步先跑优化器，再 r21 `state.roll()`，再 r22 `state.roll()`。两个 `FP4State` 是不同对象（head 18 个 site，骨干 144 / 288 个 site），互不读写，roll 的先后顺序无关。每步各 roll 恰好一次：eager 2 步、graph 重放 6 步（冒烟）、graph_equiv 里 eager ×2 + graph 各 8 步、`run_final.py` 20 步结束时两个 state 的 `step` 都等于 20、门 S 里 50 步结束时 `step = 50`。
- **delayed amax 的首次调用**：`StepGraphRunner.prepare` 的 eager 预热（每个 mask 签名各 2 次 fwd+bwd，不调 `opt_step`）在 graph 捕获前对 head 的 18 个 site 和骨干所有活动 site 做 exact 初始化；graph_equiv 里有断言（head 全部 18 个 site `inited`；骨干 `active_sites()` 全部 `inited`），冒烟里同样检查；`Handle.check_site` 在捕获内遇到未初始化 site 会报错（R22 Gate A d 部分验证过）。
- **随机数契约**：SR 用 Philox（种子是设备计数器，每次 roll +1，每个调用点固定 offset），不碰 torch 默认生成器：门 B 里 240 次候选运行（2 模型 × 3 次抽批 × 8 批 × 5 个候选）的 RngTape 回放全部 `tape_complete = True`。
- **参数身份与 state_dict**：`apply_patches` 在应用前后检查 state_dict 的键与形状（P3 栈通过）；导出 / resume 检查见 §4.7。`FP4State` 不在 checkpoint 里（预期行为，见 §4.7）。
- **r19 `run_final.py` 的 persistent + checkpoint 问题**：见 §4.7。已在 r23 入口里绕过，r19 没动；这是协调者应该知道的 r19 的潜在问题（r19 / r22 的 20 步导出是不带 `--persistent` 跑的）。
- **饱和（delayed amax 的超限）**：见 §4.5。ViT-L 骨干 dY 在 schedule / compressed 场景有 1.3–1.4% 的调用超过 8× 上一步 amax（最大 49×），比 R22 单独时高，原因没有查清；50 步 EMA 偏差 ≤0.45%，看不出后果。
- **显存**：见 §4.2：P3 的 reserved（bench 全进程峰值）比 FINAL 少约 1 GB。`gate_b` 里 P3 一个上下文里依次跑 5 个候选，每次 `free_gpu`，ViT-L 的峰值 `peak_gb` 约 20 GB（R0 的 28.8 GB 在它之前释放）。
- **没有跑**：整个 20 epoch 调度；dY headroom 16 / 32 与 exact 的对照；ViT-L 在 R22 的离群批上的 P3 数字；`--q2`（R24 quantizer）——开关已加，没有用 R24 的结果，本轮所有数字都是 R21a 量化器。

## 6. 静态分析与 roofline

本轮没有写新的 CUDA kernel，没有 ptxas / SASS 数据；roofline 见 §4.9。

## 7. 结论、推荐、失败项、契约变更请求、下一瓶颈

**结论**：R21b 的 FP4 head 与 R22 的 FP4 骨干**可以叠加，组合栈 P3 全部通过各自的门**：冒烟（包装链顺序正确，两个 `FP4State` 各 roll 一次）、门 A（0 失败）、门 B（5 次抽批各 8 批 = 40 批 / 模型，同进程对 R0 和 P1，放宽门：ViT-B 过；ViT-L 只在一个连 FINAL 都不满足的 MXFP8 离群批上不过）、门 S（8 个场景全过，EMA 偏差 ≤0.67%）、graph_equiv（8 个场景全过）、导出 / resume（两个模型都过）。整步（graph，同场次）：**ViT-L 68.43 → 81.87 img/s（×1.196，对 R0 ×5.04），ViT-B 159.15 → 191.40 img/s（×1.202，对 R0 ×5.06）**；e2e 稳态 99.1% / 98.2%。可发布（≥2%）。

**推荐**（给协调者）：
| 模型 | 推荐栈 | 理由 |
|---|---|---|
| ViT-L | **P3**（head `head.f=mx8,head.dw=nv4+sr` + 骨干 `all.d=nv4+sr+all.w=nv4+sr+nofirst`），整步 ×1.196 | 放宽门 40 批只有 1 批（FINAL 自己也不过的离群批）最小值低于 0.97（P3 对 P1 0.9693，差 0.0007）；24 批的三次主抽批合并也过严格门，但 draw3 单独不过严格门的中位（0.9887）。更保守的档：**P3H**（只做 head，×1.118，24 批过严格门）。 |
| ViT-B | **P3**，整步 ×1.202（放宽门） | 放宽门 40 批全过，但最差批余量只有 0.0035（对 P1 0.9735）至 0.0049（对 R0 0.9749），**不过严格门**；若要保持严格门，**只有 FINAL 在 n=24 上过严格门**（P3H 单独在 draw2 的最小是 0.9770，也不过；6b 13.2 说的“P3H 是 ViT-B 的严格档”在 n=24 上不成立）。若用户接受放宽门（2026-10-03 决定），用 P3；想留更大余量用 P3H（×1.148，最差批对 P1 0.9755，余量 0.0055，比 P3 多 0.002）。 |

长训练风险（只有 few-batch 证据）：ViT-L 骨干 dY 的 delayed amax 在 schedule / compressed 场景有 1.3–1.4% 的调用超限（最大 cur/prev 49×，R22 单独时 0.09–0.18%，原因没查清），50 步看不到后果；建议长训练时周期读 `FP4State.stats()`（`run_final.py` 末尾已打印）、对照 FINAL 的 val 曲线，必要时给 dY 加大 headroom（`dyhr=16`）或退回 `amax=exact`。

**失败项**：
1. 第一次导出运行失败：r19 的 `run_final.py` 在 `--persistent` 下 `torch.save` checkpoint 报 `_MultiProcessingDataLoaderIter cannot be pickled`，并且 persistent worker 让进程挂住（连带持有 GPU 锁），我杀掉了自己的进程；r23 的入口已绕过（§4.7），r19 没动。
2. ViT-L 的放宽门“最小 ≥0.97”在 5 次抽批里 1 批没过（`qdraw2:11`，FINAL 自己 0.9643 / P3 0.9731 对 R0、0.9693 对 P1），ViT-B 的严格门不过（§4.4）。
3. 没发生 stale lock / `assert_gpu_exclusive` 卡死（所有进程走 `tests/retry.sh`，没有触发重试）。会话额度中断两次，已续跑，没有重做已完成项。

**没测（not measured）**：整个 20 epoch 调度；`--q2`（R24 的量化器）在 P3 上（开关已加，本轮所有数字都是 R21a 量化器）；dY headroom 16 / 32 与 exact 的对照；P3 在 R22 的 seed-13 那一批上（该批无法复现，用了同类离群批 `qdraw2:11` 代替）；5a（`P3_q8`）的 graph_equiv / 导出 / resume / e2e / nsys。

**契约变更请求**：
1. 6b 13.2 的“ViT-B 严格档 = 只做 head”在 n=24 上不成立（P3H 在 draw2 最小 0.9770 <0.98）；严格门对 ViT-B 的批次抽样很敏感（FINAL 自己 draw1 最小 0.9804）。建议把放宽门写成 P3 的唯一判据，或把严格门的“最小”改成相对 FINAL 的差（同一批上 P3 ≥ FINAL − 0.02，也就是说本轮 40 批全满足，本轮 40 批里 P3 比 FINAL 的对 R0 梯度余弦低 −0.009…0.018，中位 0.003 / 0.0055（ViT-L / ViT-B））。
2. 放宽门“最小 ≥0.97”会被 MXFP8 离群批（FINAL 自己 0.9643 / R22 的 0.9715）卡住，不是 FP4 的责任；建议同上，改为相对 FINAL。
3. `common/config.py`：`P3_STACK` 与 `P3_PATCH_KWARGS_VITL/_VITB` 已够用，建议新增 `BEST_STACK_V3 = P3_STACK`（两个模型的 kwargs 不同）；r19 `run_final.py` 的 persistent + checkpoint 问题建议在 r19 之外的位置记录（r23 `run_final.py` 是更新的入口）。

**下一瓶颈**（P3，nsys kernel 时间，ViT-L / ViT-B，ms / 步，§4.9）：attention 16.1 / 12.7（19% / 18%，离 BF16 FLOP 下界 11.4 / 8.7 ms 还有 1.4×）；FP4 + MXFP8 量化 10.2 / 8.9（12%，离冷输入下界 7.5 / 5.2 ms 还有 1.4× / 1.7×，R24 在做）；NVFP4 GEMM 10.6 / 7.2（已接近 1.2 PFLOPS 假设）；ViT-B 上 r14 head 融合 kernel 10.4（14%，内存受限，`bwd1` + `bwd2` 可合并）；ViT-L 上优化器 7.7（单 kernel 6.65 ms，接近 DRAM 下界）；bf16 GEMM 7.9 / 6.9（qkv 前向 4.8 / 2.7 + head einsum / mask head，5a 把 qkv 前向换成 MXFP8 能再省 3.1 / 1.6 ms，见 §4.10）。
