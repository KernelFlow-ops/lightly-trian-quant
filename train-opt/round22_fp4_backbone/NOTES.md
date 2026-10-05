# Round22（R22）：真 kernel FP4（NVFP4）骨干 dgrad / wgrad（第三阶段，Sonnet 5.5 执行）

> 数字全部在 `infer-5090`（RTX 5090）上实测，持双锁；没测的写 not measured / null。结果文件：`results.json`（机器可读，由 `tests/summarize_r22.py` 从 `profile/*.json` 生成）、`profile/tables_r22.md`（全部表）。
> 没改 round0–21、common、PLAN、CONTRACTS；只 import 了 round13/14/15/16/17/19/20/21 的代码（R20 的策略解析 / GEMM 封装 / 仿真块，R21 的 `fp4q` 量化器）。
> 状态：门 A / B（三次抽批）/ S / graph_equiv / 导出 resume / 整步 bench / nsys 都已完成；附加对照臂（RN、不加 nofirst、只 FP4 wgrad、诊断臂）见 §4.8，没做的在 §0 与 §4.8 里列明。
> 会话中断过一次（额度用完），已从队列恢复，没有重做已完成的项。

## 0. 摘要

| 项 | 结果 |
|---|---|
| 交付 | `impl/fp4_block22.py`（真 kernel 块）、`impl/fp4_gemm22.py`（nv4 dgrad / dgrad+GELU' / wgrad 封装，proj wgrad 用 split-K）、`impl/patch.py`（`round22_fp4_backbone`：REQUIRES r13 + r17，覆盖 `ops.block`，旧块留作回退，`ctx.ops.opt_step` 外包 `RollingOptStep` = 先原函数再 `state.roll()`）、`run.py`（r19 run.py 改写 + 预设 `P3B` `P3B_B` `P3B_noFirst` `P3B_exact*` `P3B_rn*`）、`tests/`（`test_block22.py` = 门 A，`gate_b_real.py` = 门 B，`graph_equiv22.py`，队列 / retry / nsys / summarize 脚本） |
| 策略（6b 13.2） | ViT-L：`all.d=nv4+sr, all.w=nv4+sr` + `nofirst`（预设 `P3B`）；ViT-B：同上不加 nofirst（预设 `P3B_B`）。前向不变（qkv bf16，proj / fc1 / fc2 mx8，r17 kernel）；RHT 关；无 prefix；amax = delayed（X、权重 headroom 2，**dY headroom 8**）；head 保持 FINAL 的 r14 bf16 |
| 门 A | 通过：GEMM 对反量化乘积 dgrad ≤1.70e-3（bf16 输出舍入下限 1.66e-3）、wgrad ≤1e-6，34 例 + 小 M 尾块；RN 块对 R20b 仿真块余弦 1.0000000（10 例 × 2 模型，dW 相对差 ≤2.1e-4 = r13 dx 原子累加噪声）；SR 统计一致（差 ≤0.0024 < 0.003） |
| 门 B | 第一次抽批（种子 0–7，n=8）：ViT-L `P3B` 对 R0 中位 / 最小 0.9959 / 0.9915，对 P1 0.9957 / 0.9909，放宽门与严格门都过；ViT-B `P3B_B` 对 R0 0.9921 / 0.9881，对 P1 0.9922 / 0.9871，两个门都过；真 kernel 与 R20b 仿真几乎逐位相同。**第二次抽批（种子 8–15，n=8）**：ViT-B 仍过（REAL_B3 对 R0 0.9910 / 0.9840，对 P1 0.9908 / 0.9875，严格门也过）；**ViT-L 的放宽门最小值没过**：种子 13 那一批 REAL_B4 对 R0 / P1 为 0.9675 / 0.9685（门 0.97），而 FINAL 自己在这批上已经只有 0.9715 / 0.9724（P1 对 R0 却是 0.9995，即 MXFP8 本身就在这批上掉了），FP4 在其上再降约 0.004；这 8 批的中位 0.9923（对 P1 0.9935）。16 批合并：ViT-L REAL_B4 对 R0 中位 / 最小 0.9953 / 0.9675，ViT-B REAL_B3 0.9914 / 0.9840 |
| 门 S（StepGraphRunner） | 两个模型 × 4 场景全过：EMA 对 R0 最大偏差 ViT-L 0.48%、ViT-B 0.25%（门 5%），无 NaN；grad-norm 比值 0.975–1.047；dY 饱和：ViT-L 0.09–0.31% 的调用（最大比值 16.5），ViT-B 0–0.04% |
| graph_equiv | 两个模型 × 4 场景通过；FP4 状态每步恰好前进 1 次（eager / graph 都是） |
| 整步速度（graph，同场次 A-B-A-B，三相加权） | ViT-L：FINAL 69.00 → **73.30 img/s（×1.062）**，mid 101.57 → 95.65 ms；ViT-B：161.07 → **166.57 img/s（×1.034）**，mid 87.07 → 84.26 ms。两者都 ≥2%，可发布 |
| 对 R20a 估计 | R20a 估计骨干净省 L 6.5 / B 3.8 ms；实测 mid 步时省 L 5.9 / B 2.8 ms（nsys kernel 时间净省 5.0 / 2.6 ms） |
| 导出 / resume | 两个模型都过（r19 `run_final.py` 20 个 graph 步 + 验证 + ckpt + export，`check_export.py` 对 R0 原版路径：键 / 形状 / 优化器 / scheduler 相等，`load_model` 预测、原版 resume 一步 loss 有限） |
| 附加发现 | (1) 精度代价几乎全部来自 dgrad 的 FP4：只 FP4 wgrad 对 FINAL 的余弦 0.9998，但整步只有 ViT-L ×1.020 / ViT-B ×1.013；(2) SR 换 RN：整步 ViT-B +1.4%、ViT-L +0.9%，B 门与 ViT-B 的 S 门（2 场景）无差别；(3) ViT-L 去掉 nofirst 整步无可测差别（×1.0633 对 ×1.062），B 门也过 |
| 没测 / 缺失 | 叠加 R21b head patch 后的整步；ViT-L 不加 nofirst 的 S 门；dY headroom 16 / 32 与 exact 的臂；RN 臂的另外两个 S 场景；整步内存峰值对比（bench 的 `peak_allocated` 在 graph 池下无意义，`run_final_*_export.json` 的 `peak_reserved_gb`：ViT-L 17.4 GB、ViT-B 15.8 GB 是 20 步含验证的 reserved，不是 FINAL 对比值） |

## 1. 目标与假设

| 假设（6b 13.2 / R20a / R21a） | 结果 |
|---|---|
| 把 R20b 仿真块里的 torch 量化器换成 R21a 融合量化器、GEMM 用 QuACK nv4，门结果可逐位迁移 | 成立：RN 块对仿真块余弦 1.0000000；B 门里 REAL = SIM（ViT-L 0.9959 = 0.9959，ViT-B 0.9921 对 0.9919） |
| 骨干 dgrad + wgrad 的 GEMM 节省 ViT-L 8.1 / ViT-B 4.9 ms（R20a） | GEMM 净省 L 7.95 / B 4.78 ms（nsys，已扣 dgelu_ts epilogue 慢出来的部分）：成立 |
| 量化净增 L +1.66 / B +1.09 ms（R21a，冷输入） | 实测 L +3.33 / B +2.07 ms：**偏高约 1 倍**（量化输入大多在 L2，R21a 已知小形状暖输入下 act_kernel 比 mx8 慢约 2 倍） |
| delayed amax + headroom 在 graph 下可用，第一次调用在捕获前 eager 完成 | 成立：`StepGraphRunner.prepare` 的 eager 预热做了每个 site 的 exact 初始化；`Handle.check_site` 在捕获内遇到未初始化 site 会报错（测过） |

## 2. 改动（全部在 `train-opt/round22_fp4_backbone/`）

| 文件 | 内容 |
|---|---|
| `impl/fp4_block22.py` | r17 融合块 + R20b 策略语法（`fp4_policy.parse`，另加 `dyhr= xhr= whr=` 三个 headroom 记号）。每个 Linear 一个 `LayerQ`：<br>前向：X → `mx8.quantize(row=True,col=False)`（给前向 GEMM，qkv 前向是 bf16 无此项）+ `fp4q.quant_act(row=False,col=True)` 的 nv4 转置副本（给 wgrad，保留到反向）；权重在 block 0 每步一次：`mx8.quantize(row=True,col=False)` + `fp4q.quant_weight`（2D scale，只用 `qt` = Wᵀ，给 dgrad）。<br>反向：dY → `fp4q.quant_act(row=True,col=True,sr_row=True,sr_col=True,sr_offset=site)` 一次读入（`q` 给 dgrad，`qt` 给 wgrad）；dgrad = QuACK nv4 `dY.q × W.qt`（fc2 用 R20 的 `dgelu_ts` epilogue = GELU' + db1，其余 plain），wgrad = `dY.qt × X.qt` fp32 直接进 Parameter 梯度。alpha = 两个操作数 ts 之积（设备标量）。<br>LN / 残差 / LayerScale / GELU / attention / bias、γ 梯度全部是 r13 / r17 代码原样。site 编号 = `(block*4+linear)*3 + {x,dy,w}`；`nofirst`（及 nolast / nojoint）的块回到 FINAL 的 mx8。不支持（`set_policy` 时抛 NotImplementedError）：f=nv4/mx4、mx4、rht 用在 f / d、prefix |
| `impl/fp4_gemm22.py` | `dgrad` / `dgrad_dgelu` 复用 R20 `fp4_gemm`（鸭子类型接 `FP4Q`）；`wgrad` 自写：N·K ≤1.2e6（proj）用 QuACK 自动 split-K，其余 plain（R20a 矩阵：只有 proj 受益，46 → 21–25 µs） |
| `impl/patch.py` | `round22_fp4_backbone`：覆盖 `ops.block`，包 `opt_step`（`RollingOptStep`），`ctx.meta['round22_fp4_backbone']` = `Handle`（`set_policy` / `stats()`）。每个 patch 自己一个 `FP4State`，不共享 |
| `run.py` | r19 run.py 改写：预设注册、`gate_s`（带 FP4 饱和统计 `curve_run22`）、bench / profile 沿用 r16 动作；输出重定向到本 round |
| `tests/` | `test_block22.py`、`gate_b_real.py`、`graph_equiv22.py`、`summarize_r22.py`、`nsys_report22.py`、`queue_*.sh`、`retry.sh` |

## 3. nsys 分析过程

方法：`tests/nsys_queue.sh`（round16 `nsys_inner.sh`：graph 模式 5 步窗口，`--cuda-graph-trace=node`），分类器 = r19 的 `nsys_final_report.py`（原样 import）+ 两个新类别：round21 的 `act_kernel / w_kernel / amax_roll`（FP4 量化）、MMA atom m16n8k64 或 `*_ts` epilogue 的 QuACK kernel（NVFP4 GEMM；MXFP8 是 m16n8k32，bf16 是 m16n8k16，名字里能直接区分）。每个模型抓 FINAL 与 FINAL+r22 各一次（同一天、独占）。

| 类别（ms/步） | ViT-L FINAL | ViT-L P3B | 差 | ViT-B FINAL | ViT-B P3B_B | 差 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM MXFP8（骨干，前向 + 其余） | 24.28 | 7.82 | −16.46 | 14.50 | 4.58 | −9.92 |
| GEMM NVFP4（骨干 dgrad / wgrad，含 dgelu_ts） | 0 | 8.51 | +8.51 | 0 | 5.11 | +5.11 |
| MXFP8 量化（r17 quant_dual） | 5.07 | 1.63 | −3.44 | 4.11 | 1.13 | −2.98 |
| FP4 量化（act_kernel / w_kernel / roll） | 0 | 6.77 | +6.77 | 0 | 5.05 | +5.05 |
| LN / row kernels | 5.94 | 5.53 | −0.41 | 5.04 | 5.04 | 0.00 |
| 逐元素 / copy / 其他 | 6.91 | 7.07 | +0.15 | 7.74 | 7.86 | +0.12 |
| attention、head（bf16 GEMM + r14）、优化器、loss、conv、qkv bf16 前向 | 不变（±0.1） | | | 不变 | | |
| **kernel 总时间 / 步** | 98.48 | 93.49 | **−4.99** | 85.29 | 82.67 | **−2.62** |
| kernel 数 / 步 | 2475 | 2821 | +346 | 1641 | 1822 | +181 |
| 窗口 / 步，GPU 忙碌率 | 100.19 ms，0.982 | 96.10 ms，0.972 | | 86.60 ms，0.983 | 84.19 ms，0.981 | |

- GEMM：MXFP8 → NVFP4 的 dgrad + wgrad 净省 L 7.95 / B 4.78 ms，与 R20a 矩阵一致（ViT-L nv4 plain 161 次平均 41.4 µs，dgelu_ts 23 次平均 80 µs；矩阵逐块 375 µs × 23 块 = 8.6 ms）。
- **量化是剩下的损耗**：FP4 量化 6.77 / 5.05 ms 换掉 3.44 / 2.98 ms 的 mx8 量化，净 +3.3 / +2.1 ms。act_kernel 平均 47 µs（dY dual + SR，92 次）+ 17.8 µs（X 转置副本，92 次）+ w_kernel 8.7 µs（92 次）。输入大多是上一个 kernel 刚写出的、留在 96 MB L2 里的张量，R21a 已记录“小形状 L2 常驻时 dual 比 mx8 慢约 2 倍（17.4 对 8.2 µs）”，这里在真实流水线里重现。
- 窗口里 P3B 的 0.972 忙碌率主要来自 5 步窗口里一个 5.36 ms 的起始间隙（FINAL 的是 2.16 ms，同一位置：两个 elementwise kernel 之间，属 profile 窗口边界）；<20 µs 的小间隙总计只多 0.18 ms（430 个额外 kernel），所以 kernel 数增加的 launch 开销可忽略。bench 的 mid 步时（95.65 ms）与 nsys busy + 小间隙一致。

## 4. 实测结果

### 4.1 门 A（`tests/test_block22.py`，`profile/test_block22_vitb16.json` / `_vitl16.json`）

| 部分 | 内容 | 结果 |
|---|---|---|
| a. GEMM 层 | ViT-L（M=7203 / 7903）与 ViT-B（M=14406 / 15806）的 qkv / proj / fc1 / fc2：dgrad、fc2 dgrad+GELU'（含 db1）、wgrad（proj 另测不用 split-K），高斯与尖峰数据（scale-8 通道 + ×300 的 4 个 token），另有 M=1029、33；对反量化 fp32 乘积的相对 L2 | 34 + 2 例全过，0 失败。dgrad 1.59e-3 – 1.70e-3（= bf16 输出舍入下限 1.66e-3，门 2e-3），dgrad_db ≤1e-6，wgrad ≤1e-6 |
| b. RN 块对 R20b 仿真块 | 同一策略（`all.d=nv4+rn, all.w=nv4+rn`，另加 `+rht`），`amax=exact`，plain 块与 joint 块，正常数据与尖峰数据（outlier 通道 ×8 + cls / reg token ×300），块 0 / 1 / 最后一个 joint 块，两个模型 | 20 例全过：out / dx / dW 余弦都是 1.0000000（门 0.99999）；out 相对差 0.0，dx 相对差 ≤8e-5，dW 相对差 ≤2.1e-4（r13 dx 的原子累加不确定性，FINAL 对 FINAL 也有） |
| c. SR 统计一致性 | 8 个种子，plain 块 1 与最后 joint 块，Philox（真）对 torch.Generator（仿真）：dx、dW 对 FINAL 块的逐种子余弦的均值 | ViT-B：均值差 0.0005 / 0.0003（dx），0.0003 / 0.0008（dW）；ViT-L：0.0024 / 0.0006（dx），0.0002 / 0.0012（dW），都 <0.003。ViT-L plain 块的 dx 差 0.0024 偏大（真 kernel 更好，3σ 左右），不影响门，记录在案 |
| d. 其他 | nofirst 的块 0 与 FINAL 块一致（余弦 1.0）；空策略 = FINAL；捕获内首次调用 delayed site 会报错；`opt_step` 包装每步 roll 恰好 1 次 | 全过 |

### 4.2 门 B（`tests/gate_b_real.py`，同一进程内 R0 / P1 / FINAL / 仿真 / 真 kernel，同一权重、同一批、同一 RngTape；n=8 批，种子 0–7；放宽门 = 对 R0 和对 P1 都要中位 ≥0.98、最小 ≥0.97、loss_rel ≤2e-2）

ViT-L（候选 `REAL_B4` = P3B 策略；`SIM_B4` = R20b 仿真，同策略不带 RHT）：

| 候选 | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 |
|---|---|---|---:|---|---|---:|
| FINAL | 0.9969 / 0.9926 | 0.9970 / 0.9920 | 2.8e-3 | 过 | 过 | 1.0000 |
| SIM_B4（仿真） | 0.9959 / 0.9915 | 0.9957 / 0.9909 | 2.8e-3 | 过 | 过 | 0.9991 |
| **REAL_B4（真，P3B）** | **0.9959 / 0.9915** | **0.9957 / 0.9909** | 2.8e-3 | **过** | **过** | 0.9991 |
| REAL_B4x（amax=exact） | 0.9959 / 0.9915 | 0.9958 / 0.9909 | 2.8e-3 | 过 | 过 | 0.9991 |
| REAL_B4rn（RN 代替 SR） | 0.9957 / 0.9912 | 0.9956 / 0.9907 | 2.8e-3 | 过 | 过 | 0.9989 |
| REAL_B4nohr（dY headroom 2） | 0.9959 / 0.9915 | 0.9957 / 0.9909 | 2.8e-3 | 过 | 过 | 0.9990 |
| SIM_B4rht（R20b 原版带 RHT） | 0.9959 / 0.9915 | 0.9957 / 0.9910 | 2.8e-3 | 过 | 过 | 0.9991 |
| REAL_B3（ViT-L 不加 nofirst） | 0.9957 / 0.9912 | 0.9955 / 0.9907 | 2.8e-3 | 过 | 过 | 0.9988 |
| SIM_B3（同上，仿真） | 0.9928 / 0.9882 | 0.9923 / 0.9877 | 2.8e-3 | 过 | 过 | 0.9966 |

ViT-B（候选 `REAL_B3` = P3B_B 策略）：

| 候选 | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | loss_rel 最大 | 放宽门 | 严格门 | 对 FINAL 中位 |
|---|---|---|---:|---|---|---:|
| FINAL | 0.9949 / 0.9924 | 0.9950 / 0.9923 | 2.0e-3 | 过 | 过 | 1.0000 |
| SIM_B3（仿真） | 0.9919 / 0.9881 | 0.9920 / 0.9872 | 2.0e-3 | 过 | 过 | 0.9968 |
| **REAL_B3（真，P3B_B）** | **0.9921 / 0.9881** | **0.9922 / 0.9871** | 2.0e-3 | **过** | **过** | 0.9970 |
| REAL_B4（加 nofirst） | 0.9922 / 0.9881 | 0.9923 / 0.9872 | 2.0e-3 | 过 | 过 | 0.9970 |
| REAL_B3x（exact） | 0.9920 / 0.9879 | 0.9921 / 0.9871 | 2.0e-3 | 过 | 过 | 0.9970 |
| REAL_B3rn | 0.9916 / 0.9877 | 0.9918 / 0.9868 | 2.0e-3 | 过 | 过 | 0.9966 |
| REAL_B3nohr（dY headroom 2） | 0.9920 / 0.9879 | 0.9921 / 0.9869 | 2.0e-3 | 过 | 过 | 0.9970 |

- **ViT-B 最差批余量**：本次抽到的 8 批里 FINAL 自己最差 0.9924（R20b 那次抽到的最差批 FINAL 只有 0.9824），REAL_B3 最差 0.9871（对 P1）：离放宽门的 0.97 有 0.017 的余量。数据加载器跨进程不确定（6b 13.5），所以另外跑了种子 8–15 的第二次抽批，结果见 §4.8。
- 真 kernel 与仿真逐个种子的余弦几乎重合（REAL 与 SIM 中位差 ≤0.0002，ViT-L 的 B4 完全相同）。ViT-L 不加 nofirst 的 B3：真 kernel（0.9957）比仿真（0.9928）好 0.003（真 kernel 用的是 delayed amax + 2× headroom 且量化器逐位对齐参照；仿真的 B3 在 R20b 里也是最差的一档），原因没有查清，不影响推荐的 P3B（B4 真 = 仿真）。

**第二次抽批（种子 8–15，另一个进程，`profile/gateB_r22_vitl16_seeds8_15.json` / `_vitb16_seeds8_15.json`；6b 13.5：数据加载器跨进程不确定，所以这是不同的 8 个 batch）**

| 模型 | 候选 | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | 放宽门 | 严格门 |
|---|---|---|---|---|---|
| ViT-L | FINAL | 0.9945 / 0.9715 | 0.9955 / 0.9724 | 过（0.9715 ≥0.97） | 不过（最小 <0.98） |
| ViT-L | SIM_B4 | 0.9923 / 0.9673 | 0.9935 / 0.9681 | **不过（最小 <0.97）** | 不过 |
| ViT-L | **REAL_B4** | 0.9923 / 0.9675 | 0.9935 / 0.9685 | **不过（最小 <0.97）** | 不过 |
| ViT-B | FINAL | 0.9945 / 0.9874 | 0.9945 / 0.9901 | 过 | 过 |
| ViT-B | SIM_B3 | 0.9911 / 0.9836 | 0.9908 / 0.9872 | 过 | 过 |
| ViT-B | **REAL_B3** | 0.9910 / 0.9840 | 0.9908 / 0.9875 | 过 | 过 |
| ViT-B | REAL_B4 | 0.9910 / 0.9841 | 0.9906 / 0.9875 | 过 | 过 |

- **ViT-L 在这一抽里放宽门的“最小 ≥0.97”没过，只有一批（种子 13）**：逐种子对 R0：FINAL `[0.9895 0.9985 0.9956 0.9935 0.9934 0.9715 0.9972 0.9971]`，REAL_B4 `[0.9887 0.9976 0.9901 0.9928 0.9919 0.9675 0.9965 0.9958]`（对 P1 同形）。种子 13 上 P1 对 R0 = 0.9995（bf16 没问题），**FINAL（MXFP8）自己已经掉到 0.9715**，REAL_B4 对 FINAL 是 0.9956，即 FP4 在已有的 MXFP8 离群批上再降约 0.004。真 kernel 与仿真一致（0.9675 对 0.9673），所以这不是 kernel 的问题而是 FP4 策略在离群批上的余量。其余 7 批的最小是 0.9887（对 R0）。
- ViT-B：第二次抽批最差 REAL_B3 对 R0 0.9840、对 P1 0.9875，余量 ≥0.014。
- 两次抽批合并（n=16）：ViT-L REAL_B4 对 R0 中位 / 最小 0.9953 / 0.9675、对 P1 0.9949 / 0.9685（16 批里 1 批最小值未达 0.97）；ViT-B REAL_B3 对 R0 0.9914 / 0.9840、对 P1 0.9912 / 0.9871。
- 第三次抽批（种子 16–23）带诊断臂（只 FP4 dgrad / 只 FP4 wgrad / nojoint / nolast，含逐参数余弦）见 §4.8。

### 4.3 门 S（StepGraphRunner 下，候选 graph 对 R0 eager；4 场景 × 50 步 × 2 批；`profile/gateS_vit*_graph_vs_R0*.json`）

| 模型 | 场景 | EMA 对 R0 最大偏差 | 门 | 有限 | grad-norm 比值范围 | dY 饱和（exceed / calls） |
|---|---|---:|---|---|---|---|
| ViT-L P3B | schedule | 0.40% | 过 | 是 | 0.975–1.024 | 34 / 18400（0.18%） |
| ViT-L P3B | compressed | 0.19% | 过 | 是 | 1.002–1.027 | 16 / 18400（0.09%） |
| ViT-L P3B | fixed_mid | 0.22% | 过 | 是 | 0.999–1.009 | 57 / 18400（0.31%） |
| ViT-L P3B | fixed_late | 0.48% | 过 | 是 | 0.997–1.013 | 47 / 18400（0.26%） |
| ViT-B P3B_B | schedule | 0.25% | 过 | 是 | 0.995–1.047 | 0 / 9600 |
| ViT-B P3B_B | compressed | 0.05% | 过 | 是 | 0.990–1.003 | 4 / 9600（0.04%，最大比值 13.1） |
| ViT-B P3B_B | fixed_mid | 0.11% | 过 | 是 | 0.982–1.007 | 0 / 9600 |
| ViT-B P3B_B | fixed_late | 0.06% | 过 | 是 | 0.995–1.014 | 0 / 9600 |

- 饱和统计来自 R21a 的 `FP4State.stats()`（每 site、行 / 转置副本，设备端计数，graph 外读）：“exceed” = 某一步 dY 的当前 amax 超过 headroom（8）× 上一步 amax（那一步该 site 的 e4m3 scale 在 448 处裁剪）。**全部发生在 dY**（X 与权重 0 次，X 的最大 cur/prev 比值 1.3–1.9，权重 1.0）。ViT-L 超限的 site 集中在块 1、2、4、5、11 的 qkv / proj / fc2 的 dY，最大 cur/prev 比值 16.5（headroom 2 时 R20b 测得 1.11%，最大 10.15）。S 门的 50 步看不到这些裁剪的后果（EMA 偏差 ≤0.48%）。
- 全部 S 门用的是 dY headroom 8（`Handle.describe()` 与 `stats()["headroom"]` 都记了 `dy: 8.0`）：R21 的 `READY_headroom` 在 01:38 出现，早于本 round 的第一次 B / S / bench 运行（01:46 起），且 `has_headroom` 是自动检测的，所以没有需要重跑的“旧 headroom”结果；`fp4q.py` 的修改时间（01:34）也早于所有运行。
- 每个场景 50 步，graph 捕获数 1–7 个（compressed 场景 7 个 mask 签名），每步 `state.step` 恰好 +1（stats 的 `step` = 50）。

### 4.4 graph_equiv（`tests/graph_equiv22.py`，8 步，graph 回放对 eager；FP4 状态在 eager 预热之后快照，每次运行前原地恢复）

| 模型 | 场景 | 更新余弦 graph-vs-eager | eager-vs-eager（噪声底） | loss 相对差 graph-vs-eager | eager-vs-eager |
|---|---|---:|---:|---:|---:|
| ViT-B | schedule / compressed / fixed_mid / fixed_late | 0.999890 / 0.999766 / 0.999996 / 0.999996 | 0.999918 / 0.999867 / 0.999881 / 0.999996 | 2.2e-4 / 2.7e-4 / 4.8e-5 / 5.3e-5 | 2.6e-4 / 2.8e-4 / 1.8e-4 / 6.6e-6 |
| ViT-L | schedule / compressed / fixed_mid / fixed_late | 0.999990 / 0.999994 / 0.999994 / 0.999996 | 0.999988 / 0.999991 / 0.999992 / 0.999997 | 5.7e-5 / 1.3e-4 / 8.3e-5 / 9.8e-5 | 1.1e-4 / 1.5e-4 / 5.6e-5 / 4.4e-5 |

graph 与 eager 的差别和 eager 两次之间的差别同一量级（r13 的 dx 原子累加不确定性 + FP4 的放大），FP4 状态每步前进 K 次（`fp4_state_advance.ok = True`，每个场景）。`graph_equiv22.py` 还断言所有“活动 site”在预热后都已初始化（nofirst 的块没有 site）。

### 4.5 导出 / resume（r19 `run_final.py` 20 个 graph 步 + 验证 + `save_checkpoint` + `export_model`，再在新进程里跑 r19 `tests/check_export.py`：与 R0 原版路径比较 state_dict / 优化器 / scheduler、`lightly_train.load_model` 预测一张图、在原版路径上 resume 一步）

两个模型都过（`profile/export_vitl16.json` / `export_vitb16.json`，`profile/run_final_*_export.json`）。栈 = `FINAL,round22_fp4_backbone:<变体>`（ViT-L 带 nofirst，ViT-B 不带），r19 `run_final.py` 在真实 DataLoader 上跑 20 个 graph 步（1 个 graph 捕获，0 次 eager 回退）+ 一次验证（ViT-L 12.6 s / ViT-B 8.5 s）+ `save_checkpoint` + `export_model`；`check_export.py` 在新进程里按原版（R0）路径检查：

| 模型 | check_export | ckpt / export / 原版训练导出的 state_dict 键、形状、dtype | 优化器 / scheduler 结构 | 导出权重 | `load_model` 预测一张图 | 原版路径 resume 一步 |
|---|---|---|---|---|---|---|
| ViT-L | pass | 全相等（388 键） | 相等 | 全有限；19 个参数在 20 步后已变化（最大变化 2.7e-5） | 通过（[1008,1008]，int32，DINOv3EoMTSemanticSegmentation） | `load_state_dict` 全匹配，loss 29.69 有限 |
| ViT-B | pass | 全相等（208 键） | 相等 | 全有限；19 个参数已变化（最大 3.8e-5） | 通过（[1008,1008]） | 全匹配，loss 28.83 有限 |

（与 r19 的 FINAL 导出检查一致：同样 19 个参数变化、同量级的权重变化；`FP4State` 不进 checkpoint，原版路径 resume 没有 FP4。）

### 4.6 整步速度（`run.py --action bench --runner graph`，r16 `StepGraphRunner`，三相 0.2 / 0.6 / 0.2，warmup 20，50 × 3；每次一个进程；FINAL 与候选交替 A-B-A-B，同一场次）

| 模型 | 栈 | 每次 weighted img/s | 均值 img/s | 均值 mid 步时 | 对 FINAL 倍数（img/s） | mid 步时差 |
|---|---|---|---:|---:|---:|---:|
| ViT-L B=7 | FINAL | 69.06 / 68.95 | 69.00 | 101.57 ms | 1.000 | – |
| ViT-L B=7 | **FINAL + r22（P3B）** | 72.93 / 73.66 | **73.30** | **95.65 ms** | **1.062** | −5.92 ms |
| ViT-B B=14 | FINAL | 161.44 / 160.70 | 161.07 | 87.07 ms | 1.000 | – |
| ViT-B B=14 | **FINAL + r22（P3B_B）** | 166.55 / 166.58 | **166.57** | **84.26 ms** | **1.034** | −2.80 ms |

附加臂（第二场次，会话恢复后；FINAL 的绝对值比第一场次低约 1%，所以只在场次内比较；每行是单次 bench，A-B-A 的 FINAL 两次相差 ≤0.1%）：

| 模型 | 栈 | img/s | mid 步时 | 对同场次 FINAL（三次均值，ViT-L 68.46 / ViT-B 159.31 img/s，三次之间 ≤0.1%） |
|---|---|---:|---:|---:|
| ViT-L | `P3B_B`（骨干全部 FP4，**不加 nofirst**） | 72.79 | 96.30 ms | ×1.0633（−6.07 ms；第一场次 `P3B` ×1.062：nofirst 的 ~0.3 ms 在噪声内） |
| ViT-L | `P3B_rn`（RN 代替 SR；第一场次，单次） | 73.98 | 94.80 ms | ×1.072（对 P3B 的 73.30 快 0.9%） |
| ViT-B | `P3B_noFirst`（B3 + nofirst） | 165.30 | 84.87 ms | ×1.0376（−3.20 ms） |
| ViT-B | `P3B_rnB`（RN 代替 SR） | 167.59 | 83.70 ms | ×1.0520（−4.38 ms，比 SR 版快约 1.4%） |
| ViT-L | 只 FP4 wgrad（`all.w=nv4+sr+nofirst`，dgrad 仍 mx8） | 69.84 | 100.36 ms | ×1.0203（−2.01 ms） |
| ViT-B | 只 FP4 wgrad（`all.w=nv4+sr`） | 161.44 | 86.90 ms | ×1.0134（−1.18 ms，低于 2% 发布线） |

SR 的代价（Philox + SR 码的 ALU）：ViT-B 约 1.4%、ViT-L 约 0.9% 步速；B 门没有差别（§4.2 的 `REAL_B3rn` / `REAL_B4rn`），ViT-B 的 RN 臂 S 门见 §4.8。只 FP4 wgrad 只拿到约 1/3 的整步收益（ViT-L ×1.020、ViT-B ×1.013）：整步收益的大头来自 dgrad，精度代价也来自 dgrad（§4.8）。

- 各相步时（FINAL → r22）：ViT-L early 101.7 → 95.7、mid 101.6 → 95.7、late 100.8 → 94.9；ViT-B early 87.9 → 85.0、mid 87.1 → 84.3、late 85.6 → 82.5 ms。
- 对 R0（r19：ViT-L 4.22×、ViT-B 4.29×）：FINAL+r22 对应约 4.48× / 4.44×（按 FINAL 的倍数相乘，同场次 R0 没有重测）。
- 发布线（提升 ≥2%）：两个模型都过。

### 4.7 与 R20a 估计的对照

| ms / 步 | R20a 估计（骨干 ③ 净省，含 R21a 量化净增） | 实测（nsys kernel 时间） | 实测（bench mid 步时） |
|---|---:|---:|---:|
| ViT-L | 6.5（GEMM 8.13 − 量化 1.66） | 4.99（GEMM 7.95 − 量化 3.33 − 其余 −0.3） | 5.92 |
| ViT-B | 3.8（GEMM 4.93 − 量化 1.09） | 2.62（GEMM 4.78 − 量化 2.07 − 其余 +0.1） | 2.80 |

差距全部来自量化净增（R21a 用冷输入估计）。GEMM 部分与 R20a 完全吻合。bench 比 nsys 多省 0.2–0.9 ms（nsys 单次 5 步窗口的边界间隙与 bench 的 150 步均值不是同一统计量）。

### 4.8 第三次抽批（ViT-L）与诊断臂；其他附加臂

**第三次抽批（ViT-L，种子 16–23，另一个进程，带诊断臂和逐参数余弦；`profile/gateB_r22_vitl16_draw3.json`）**：

| 候选 | 对 R0 中位 / 最小 | 对 P1 中位 / 最小 | 放宽门 | 严格门 | 对 FINAL 中位 / 最小 |
|---|---|---|---|---|---|
| FINAL | 0.9956 / 0.9924 | 0.9959 / 0.9910 | 过 | 过 | – |
| SIM_B4（仿真） | 0.9930 / 0.9906 | 0.9934 / 0.9898 | 过 | 过 | 0.9985 / 0.9960 |
| **REAL_B4（P3B 策略）** | **0.9931 / 0.9903** | **0.9933 / 0.9896** | **过** | **过** | 0.9984 / 0.9958 |
| REAL_B4d（只 FP4 dgrad，wgrad 仍 mx8） | 0.9932 / 0.9902 | 0.9934 / 0.9895 | 过 | 过 | 0.9986 / 0.9957 |
| REAL_B4w（只 FP4 wgrad，dgrad 仍 mx8） | 0.9954 / 0.9922 | 0.9958 / 0.9909 | 过 | 过 | **0.9998 / 0.9994** |
| REAL_B4nj（+ nojoint，最后 3 块回到 mx8） | 0.9937 / 0.9913 | 0.9940 / 0.9905 | 过 | 过 | 0.9993 / 0.9969 |
| REAL_B4nl（+ nolast） | 0.9927 / 0.9904 | 0.9931 / 0.9896 | 过 | 过 | 0.9986 / 0.9959 |

- 这一抽 ViT-L 过（含严格门）。ViT-L 三次抽批合计 24 批：只有种子 13（第二次抽批）一批的最小值低于 0.97。
- **诊断：FP4 的精度代价几乎全部来自 dgrad，wgrad 的 FP4 几乎免费**：只 FP4 wgrad 的候选对 FINAL 的中位 / 最小余弦是 0.9998 / 0.9994（与 FINAL 对 FINAL 的重复运行噪声同量级），只 FP4 dgrad 的候选与完整 B4 几乎相同（0.9986 对 0.9984）。nojoint（3 个 joint 块回到 mx8）把对 FINAL 的最小从 0.9958 抬到 0.9969，nolast 没有帮助。这也解释了为什么 mx8 的离群批（种子 13）上 B4 再掉 0.004：掉的是 dgrad 的 FP4 误差。
- **含义**：如果以后要给 ViT-L 一个更保守的档，“只 FP4 wgrad”（`all.w=nv4+sr+nofirst`）保留 wgrad 的 GEMM 节省（ViT-L R20a 矩阵：wgrad 8.8 → 4.5 ms）而几乎不增加误差；它的整步速度见下面的 bench（若已跑完）。

**其他附加臂**：
- ViT-B 的 RN 臂 S 门（`P3B_rnB`，graph 对 R0，schedule 与 fixed_mid 两个场景；`profile/gateS_vitb16_P3B_rnB_graph_vs_R0.json`）：EMA 最大偏差 0.15% / 0.10%（SR 的 `P3B_B`：0.25% / 0.11%），无 NaN，grad-norm 比值 0.987–1.011，dY 饱和 0 / 9600。也就是说 50 步 S 门分不出 RN 与 SR，RN 整步快约 1.4%（ViT-B）。另外两个场景（compressed、fixed_late）没跑。
- bench 附加臂：见 §4.6（无 nofirst、RN、只 FP4 wgrad）；B 门里 RN、amax=exact、dY headroom 2 三个对照臂见 §4.2（第一次抽批）：与 SR / delayed / headroom 8 的差别 ≤0.0005。
- **没做（not measured）**：dY headroom 16 / 32 与 `amax=exact` 的 B 门臂、ViT-L headroom 16 的 S 门、ViT-L 不加 nofirst 的 S 门（脚本 `tests/queue_gates2.sh` / `queue_extra*.sh` 在，恢复后有一段时间 GPU 锁被另一个代理挂住的 nsys 任务占着，之后我把时间用在了上面两组更重要的对照上）。

## 5. 正确性

- **headroom 版本**：见 §4.3，所有结果都是 dY headroom 8（READY_headroom 01:38 早于第一次运行）。
- **delayed amax 与 graph**：`StepGraphRunner.warm` 在每个 mask 签名上做 eager 的 fwd + bwd（不 roll），第一次调用做 exact 初始化；捕获（不执行 kernel）时 `inited` 已为 True；`Handle.check_site` 在捕获内遇到未初始化 site 抛错（`test_block22.py` d 部分用 monkeypatch 验证）；`roll` 在 `opt_step` 包装里，随优化器一起被捕获，graph 与 eager 路径每步恰好 +1（`graph_equiv22` 的 `fp4_state_advance`）。预热之后 `next` 槽里留有预热步的 amax（预热不 roll），第一个真实步会把它并进去，只会让第二步的 headroom 略大。
- **SR 不碰 torch 默认生成器**：Philox 的 seed 是设备计数器（每次 roll +1），`sr_offset` = site 号。B 门里 RngTape 回放完整（`tape_complete=True`）。
- **Parameter 身份与 state_dict 不变**：`apply_patches` 在应用前后检查 state_dict 的键与形状；`FP4State` 不进 checkpoint（原版 resume 没有 FP4）。
- **内存泄漏已修**：第一次 S 门在第二个场景 OOM（`Handle.ctx` ↔ `ctx.meta` 的引用环让候选的 ctx 在场景之间没被释放，R0 eager 峰值 28.4 GB 放不下）。修复：`curve_run22` 结束时断开 `Handle.ctx`，并清空 `weights` / `layers`；之后每个场景前后的 `memory_allocated` 都回到 0.14–0.21 GB。
- **其他修复**：graph_equiv 里“所有 site 已初始化”的断言没考虑 nofirst（改成只查活动 site）；`gate_b_real.py` 的 `--cands` 里自定义候选必须写成 `NAME=impl:policy`（一次启动因名字没带 `=` 失败，已重跑，旧日志保留为 `logs/gateB_vitl_failed_cands.log`）。

## 6. 静态分析与 roofline

本轮没有写新的 CUDA kernel（量化器是 R21a 的、GEMM 是 QuACK 的），所以没有 ptxas / SASS 分析。roofline：ViT-L 的 nv4 GEMM 平均 41 µs / 次（plain，161 次），按矩阵形状折算 0.7–1.2 PFLOPS，与 R20a 矩阵一致；FP4 量化 kernel 在 L2 暖输入下离带宽下限还差 2–3 倍（见 §3、§7 的请求）。

## 7. 结论、推荐、失败项、契约请求、下一瓶颈

**结论**：R22 的实现和除一项外的全部门都完成。真 kernel FP4 骨干 dgrad / wgrad 在 6b 13.2 的两个策略下：门 A 通过；门 B 三次抽批：第一次（n=8）两个模型放宽门 + 严格门都过，**第二次（n=8）ViT-B 过、ViT-L 有一批（种子 13）最小余弦 0.9675 / 0.9685 <0.97**（该批 FINAL 自己只有 0.9715，FP4 再降 0.004，真 kernel = 仿真），第三次（ViT-L n=8）又全部通过；门 S（graph 下）8 个场景全过；graph_equiv 全过；导出 / resume 两个模型都过。整步 ViT-L ×1.062、ViT-B ×1.034，超过 2% 发布线。发布栈：`BEST_STACK_V2 + ("round22_fp4_backbone",)`，变体 ViT-L = `all.d=nv4+sr+all.w=nv4+sr+nofirst`，ViT-B = `all.d=nv4+sr+all.w=nv4+sr`（预设 `P3B` / `P3B_B`，在 `run.py` 里注册；新栈名由协调者写进 common/config.py）。

**需要协调者决定**
1. **ViT-L 的 B 门（放宽门最小值）**：16 批里 1 批没过（0.9675，门 0.97），这一批 FINAL 自己是 0.9715（MXFP8 的离群批，bf16 的 P1 对 R0 是 0.9995）。按 11.5 的字面规则（n≥8、中位 ≥0.98、最小 ≥0.97）：第一次抽批过、第二次抽批不过，要由你判断是接受（FP4 增量只有 0.004，且发生在 FINAL 已经临界的批上）还是收紧 ViT-L 的策略。第三次抽批（种子 16–23）ViT-L 又全部通过（含严格门），所以 24 批里只有这 1 批的最小值不过。诊断臂（§4.8）显示精度代价几乎全部来自 dgrad 的 FP4，只 FP4 wgrad 与 FINAL 几乎不可区分（对 FINAL 中位 0.9998），但只 FP4 wgrad 的整步收益只有 ViT-L ×1.020、ViT-B ×1.013；`nojoint`（最后 3 块回到 mx8）把对 FINAL 的最小从 0.9958 抬到 0.9969。
2. ViT-L 的 `nofirst`：B 门里 `REAL_B3`（不加 nofirst）也过严格门（0.9957 / 0.9912），整步速度与 `P3B` 无可测差别（×1.0636 对 ×1.062，场次不同）；R20b 仿真的 B3 在 S 门 grad-norm 比值到 1.06–1.11，真 kernel 没测 S 门。我按 6b 保留 nofirst。
3. SR 换成 RN：整步 ViT-B 快约 1.4%（`P3B_rnB` ×1.052 对 `P3B_noFirst` ×1.0376）、ViT-L 快约 0.9%，B 门没有差别（0.9957 对 0.9959），ViT-B 的 RN 臂 S 门（2 个场景）与 SR 同样通过；契约选 SR，我没有改。RN 无偏性的长期影响只有长训练能回答。
4. ViT-B 骨干 FP4 只快 3.4%，依赖下面请求 (a)–(c)；如果 R21 不改量化器，R23 可以考虑 ViT-B 只发布 head（严格档 H3）+ 骨干 FP4 作为可选档。

**对 R23 的建议**
1. 用 `P3B`（ViT-L）/ `P3B_B`（ViT-B）；head 的 FP4 由 R21b 的 patch 提供，两个 patch 各持一个 `FP4State`、各包一层 `opt_step`，应当可以叠加，但**叠加后要重跑 B / S / graph_equiv / bench**（本轮没测叠加）。
2. 剩下的骨干 FP4 开销在量化：FP4 量化 6.8 / 5.1 ms 换掉 3.4 / 3.0 ms 的 mx8 量化，净 +3.3 / +2.1 ms。对 R21 的请求（不改 R21a 的量化器，所以写成请求）：(a) `act_kernel` 在 L2 暖输入、小形状（[7203,1024] 级）下的 CTA 延迟 / 波次：真实流水线里 dY dual + SR 平均 47 µs（ViT-L），X 转置副本 17.8 µs，理想值约一半；(b) 一个 kernel 同时输出 mx8 行副本和 nv4 转置副本（X 现在读两遍）；(c) 权重量化器加 `row=False`（或同时输出 mx8 行副本）：现在 mx8 行副本和 nv4 的 `q + qt` 各读一遍 fp32 权重，`q` 用不上；(d) Philox 7 轮（R21a 已测 SR 模式快 4–6%）。
3. dY 的 headroom：8 之后 ViT-L 仍有 0.09–0.31% 的 dY 调用饱和（最大 cur/prev 16.5），ViT-B 0–0.04%（最大 13.1），S 门没看到后果。`dyhr=16` 的对照没跑（§4.8）；dY 退回 `amax=exact` 的成本是 +25–60% 量化时间（R21a），不建议，除非长训练出现发散。

**失败项**：见 §5（S 门 OOM 已修；graph_equiv 断言已修；候选解析错误已重跑；summarize 脚本的字符串 bug 已修）；会话额度用完中断一次，已从队列恢复，没有重做已完成的项；恢复后 GPU 锁被另一个代理的 nsys 任务占着约 15 分钟（我没有碰它）。没测的项：叠加 R21b 后的整步、ViT-L 不加 nofirst 的 S 门、dY headroom 16 / 32 与 exact 的臂、RN 臂的另外两个 S 场景、整步内存峰值对比。

**下一瓶颈**：FP4 量化器（L 6.8 / B 5.1 ms）、骨干剩余的 MXFP8 前向 GEMM（L 7.8 / B 4.6 ms，proj / fc2 带 fp32 残差的 epilogue 受带宽限制）、attention（L 15.9 / B 12.7 ms）、head 的 bf16 GEMM（L 19 / B 22 ms，R21b）。
