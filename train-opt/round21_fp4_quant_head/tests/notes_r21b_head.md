

---

<!-- R21b-BEGIN -->
# Round21（R21b）：head ConvT 接入真 FP4 kernel（H3 = `head.f=mx8, head.dw=nv4+sr`），负责：Sonnet，第三阶段

> 全部数字在 `infer-5090`（RTX 5090）上实测，持双锁；整步只在 round16 `StepGraphRunner` 下测，与 FINAL（`common.config.BEST_STACK_V2`）同场次对比。没测的写 null / not measured。**本章只覆盖 R21b**，R21a 的内容（§0–§8）没有改动。
> 没改 round0–20、common、PLAN、CONTRACTS；只 import 了 r14 `head` / `head_cuda`、r17 `mx8` / `gemm_mx8`、r20 `fp4_policy` / `fp4_cands` / `fp4_epi` / `fp4_ref` / `fp4_sim` / `fp4_head`（仿真 head 参照）和 r19 / r16 的 `run.py`、`run_final.py`、`check_export.py`。
> 中途因会话额度用完中断过一次（不是任务出错），之后只补缺的项；ViT-L 的整步 bench 与 B 门前 4 批是中断前跑完的，其余是之后跑的（时间戳见各 json）。

## 0. 摘要

{{SUMMARY}}

## 1. 目标与假设

| 假设（来自 PLAN_v3 / 6a 12.4 / R20a / R21a） | 实测 |
|---|---|
| head 的 dgrad+wgrad 用 nv4 GEMM 比 bf16 快约 5×（R20a：ConvT 每步净省 ① 7.1 / 7.0 ms，加前向 mx8 ② 3.3 / 3.05 ms） | {{GEMM_CLAIM}} |
| dZ 一次读入同时出 q 与 qt（R21a 量化器，dual + SR，headroom=8）；权重每步量化一次 | 实现，见 §2；dgrad / wgrad GEMM 对反量化乘积 fp32 输出误差 5e-8（bf16 输出受 bf16 舍入限制 1.7e-3） |
| RN 变体与 R20b 仿真 head 一致（6b 13.4：余弦 ≥ 0.9999） | **逐位相同**（logits 与全部 12 个输入 / 参数梯度的 relL2 = 0），见 §4.1 |
| H3 过放宽 B 门 / S 门（R20b 仿真：ViT-L / ViT-B 都过，ViT-B 过严格门） | {{GATE_CLAIM}} |

## 2. 改动（全部在 `train-opt/round21_fp4_quant_head/`，R21a 的 `impl/fp4q.py`、`impl/fp4_kernels.cu`、`tests/test_fp4q.py`、`tests/bench_fp4q.py` 没动）

新增文件：

| 文件 | 内容 |
|---|---|
| `impl/head_fp4.py` | 真 kernel head（r14 `head.py` + r20b `fp4_head.py` 为模板复制修改）：`Handle`（策略、`FP4State`、每步权重副本、call 计数）、`_ConvT4`（按 f / d / w 分派 bf16 / mx8 / nv4）、`_Upscale`、`predict`、`nv4_gemm` |
| `impl/patch.py` | patch `round21_fp4_quant_head`：覆盖 `ops.predict`（旧函数保留作回退，也用于 `torch.is_grad_enabled()==False` 和不合格形状）；包住 `ctx.ops.opt_step`：先调原函数再 `state.roll()` |
| `run.py` | r19 preset 机制：`P3H`（FINAL + H3）、`P3H_H1`、`P3H_H2`，另有 `P3H_RN / P3H_EX / P3H_HR2 / P3H_RNEX`；`gate_s` 的候选结果附 `state.stats()` 分组饱和统计；`graph_equiv`（FP4 状态在各次运行之间复位） |
| `tests/test_head_fp4.py` | Gate A（gemm / head / graph 三部分） |
| `tests/gate_b_real.py` | Gate B（运行时切换策略，同进程 R0、P1、候选；带丢弃式预热） |
| `tests/bench_head_gemm.py` | head ConvT 的 nv4 GEMM 配置对比（graph 回放，交替，3 轮中位） |
| `tests/chain.sh` `queue_bench.sh` `queue_nsys.sh` `retry.sh` `bg.sh` `bg_raw.sh` `run_final_r21.py` `make_tables21b.py` | 排队、整步 bench、nsys、导出 / resume 驱动和表格生成 |

**每个 ConvT（两个）的数据流**（H3；每步 4 次 predict = 3 次 joint 块前的 predict + final，每次 2 个 ConvT）

| 阶段 | 做法 |
|---|---|
| 权重 | 每次 forward 的**第一次 predict**（call 0）量化一次，其余 3 次 predict 复用：`wm`（bf16 [4C, C]，由 fp32 参数重排）→ r17 `mx8.quantize(row)`（前向）+ `fp4q.quant_weight(nv4)`（q / qt，2D 16×16 scale，delayed amax，headroom 2）。放在 forward 开头而不是 `opt_step` 后标脏，是因为 `StepGraphRunner.warm` 的预热 fwd+bwd 不调 `opt_step`：用"标脏"会让捕获时漏掉量化 kernel |
| 前向 | X（bf16）→ r17 `mx8.quantize(row)` → `gemm_mx8.fwd_bias`（r14 `mx8_fwd` 路径）；同时 `fp4q.quant_act(row=False, col=True)` 出转置副本 `qt`（沿 token，delayed amax，headroom 2）留给 wgrad；bf16 的 X 不再保存（比 r14 少留 y1 [4M, C]） |
| 反向 | dZ（`ht.bwd2` 输出）一次读入：`fp4q.quant_act(row=True, col=True, sr_row=True, sr_col=True, headroom=8)` 同时出 `q`（dgrad 的 A）和 `qt`（wgrad 的 A）；dgrad = `dZ.q × W.qt`，wgrad = `dZ.qt × X.qt`（fp32 输出直接作为参数梯度）；alpha = ts_a·ts_b 以设备标量传入（`fp4_epi` 的 `plain_ts` 后端） |
| 不变 | r14 融合 GELU+dwconv+LN2D kernel、einsum、mask_head、class_head、backbone.norm；Parameter 身份与 state_dict 不变 |

**状态机**（6b 13.3）：每个 patch 一个 `FP4State`（18 个 site：4 次 predict × 2 个 ConvT × {X 列副本、dZ 行 + 列副本} + 2 个权重）；`ctx.ops.opt_step` 外包一层，先调原函数再 `roll()`，所以每步恰好 roll 一次（eager 与 graph 内都成立）。dZ 的 site 用 `headroom=8`（`Handle.dy_headroom=8.0` 默认；B 门 / S 门 / bench 的 `P3H*` 预设都是这个设置，`P3H_HR2` 才是 2），X 与权重用 2。predict 序号 = `calls % 4`（opt_step 后清零），第一次 predict 做权重量化。delayed 的 site 首次调用要 eager 做 exact 初始化：`StepGraphRunner.warm`（捕获前对每个 mask 签名各跑 2 次 eager fwd+bwd，不调 opt_step）已经满足；Gate A graph 部分验证了"预热 → 复位 `next` 槽 / seed → 捕获 → 重放"与 eager 逐位相同。

## 3. nsys 分析过程

{{NSYS_PROCESS}}

## 4. 实测结果

### 4.1 Gate A（`profile/gateA_head_fp4.json`，`tests/test_head_fp4.py --part all`，0 失败）

{{GATE_A}}

{{GEMM_BENCH}}

### 4.2 Gate B（`profile/gateB_r21b_vitl16.json` / `vitb16.json`，各 8 批，同进程 R0、P1、候选，同权重 / 批 / RngTape）

候选 delayed amax 在单步里 = 第一次调用的 exact amax × headroom（X、W ×2，dZ ×8），所以这是"稳态 delayed + headroom"的真实量化条件。`H3x` = amax exact，`H3rn` = RN；三者数值相同到 4 位。每个进程先做一遍丢弃式预热（QuACK autotune / JIT 之后第一个 FINAL 的梯度与后面的不同，见 §5）。

{{GATE_B}}

### 4.3 Gate S（4 场景 × 50 步 × 2 批，StepGraphRunner 下的 P3H 对 R0 eager）

{{GATE_S}}

### 4.4 graph_equiv

{{GRAPH_EQUIV}}

### 4.5 导出 / resume（r19 `check_export.py`；checkpoint 由 P3H 栈在 graph 下跑 20 步 + 验证 + Lightly 自己的 `save_checkpoint` / `export_model` 产生）

{{EXPORT}}

### 4.6 整步速度（`profile/bench_*_all.json`，graph 回放，三相 0.2 / 0.6 / 0.2，warmup 20，50 × 3）

{{BENCH}}

### 4.7 nsys kernel 类别（FINAL 对 P3H，mid 相，5 步，graph 节点追踪）

{{NSYS_TABLE}}

## 5. 正确性与踩到的问题

{{CORRECTNESS}}

## 6. 静态分析与 roofline

{{ROOFLINE}}

## 7. 结论、推荐、失败项、契约变更请求、下一瓶颈

{{CONCLUSION}}
<!-- R21b-END -->
