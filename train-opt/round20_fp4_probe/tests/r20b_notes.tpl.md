

---

# Round20（R20b）：FP4 策略门扫描（半真实仿真：torch 量化 + 真实 QuACK FP4 GEMM），B 门 / S 门，负责：Sonnet，第三阶段

> 本章只覆盖 R20b，R20a 的内容（上面）没有改动。数字全部在 `infer-5090`（RTX 5090）上实测，持双锁，没测的写 not measured / null。**所有门都是 eager 下的仿真**：量化器是 torch 实现（`fp4_ref` 语义），GEMM 是真实的 QuACK FP4 kernel，所以门的结果能直接迁移到 R22（只需把 torch 量化器换成融合 kernel）；**这里没有任何整步速度数字**（仿真的 torch 量化器比融合 kernel 慢得多，整步时间没有意义）。节省来自 R20a 的 `per_step_estimate`（GEMM 实测 + 融合量化器估计），不是本轮实测。
> 状态：{{STATUS}}

## 0. 摘要

| 项 | 结果 |
|---|---|
| 做了什么 | `impl/fp4_policy.py`（策略串解析）、`impl/fp4_sim.py`（simulation 量化器：`fp4_ref` 语义 + 专用 SR 生成器 + 共享 RHT 符号 + `amax=exact|delayed`）、`impl/fp4_gemm.py`（真实 QuACK FP4 GEMM 的 fwd / dgrad / wgrad 封装，nv4 走 `fp4_epi.nv4_epilogues()`、mx4 走 r13 epilogue）、`impl/fp4_block.py`（r17 融合块的副本，按 GEMM 种类混合 bf16 / mx8 / nv4 / mx4）、`impl/fp4_head.py`（r14 head 的副本，ConvT f/d/w 分别选格式）、`impl/patch.py`（`BEST_STACK_V2 + ("round20_fp4_probe",)`，策略运行时切换 `ctx.meta["round20_fp4_probe"].set_policy(s)`）；`tests/` 里的块测试、`gate_b_multi.py`、`gate_s_multi.py`、`summarize_r20b.py` |
| 门结果（一句话） | {{HEADLINE}} |
| 推荐 POLICY | {{POLICY_SUMMARY}} |
| R21b（head）/ R22（骨干） | {{R_DECISION}} |
| 没测 | {{NOT_MEASURED}} |

## 1. 目标与假设

用放宽门（CONTRACTS 11.5）筛出哪些 GEMM 可以改成 FP4、用哪种配方，定下两个模型各自的 POLICY，并套用 11.6 的 go/no-go。

| 假设（R20a 之后） | 本轮要验证 |
|---|---|
| head 的 dgrad+wgrad 用 nv4+SR 能过放宽门（R20a：head wgrad 误差 0.013，dgrad 0.144–0.175） | H1 / H3 的 B 门（≥8 批）、S 门 |
| 骨干 dgrad+wgrad nv4（RHT+SR 用在 wgrad）能过放宽门；block 0 的 massive activation 要处理（prefix 或 nofirst） | B1–B5，B3 对 B4 对 B5 |
| delayed amax 在 50 步里不会让 nv4 饱和 | S 门（候选全部用 `amax=delayed`），加 `amax=exact` 对照与 cur/prev 比值诊断 |
| fc 前向 nv4（④）和 qkv 前向 mx8（⑤a）按 R20a 规则不进实现，但可以顺带筛选 | 仅记录 |

## 2. 改动（全部在 `train-opt/round20_fp4_probe/`；没有改 round0–19、common、PLAN、CONTRACTS，也没有改 R20a 已有的 `fp4_ref.py` / `fp4_epi.py` 等文件，R21a 需要的这两个文件语义不变）

| 文件 | 内容 |
|---|---|
| `impl/fp4_policy.py` | 策略串：`<lin>.<kinds>=<fmt>[+opts]`，lin ∈ qkv/proj/fc1/fc2/fc/attn/all/head，kinds ⊆ fdw，fmt ∈ bf16/mx8/nv4/mx4；条目选项 `sr rn rht w2d w1d`（作用于紧跟的条目），全局选项 `nofirst nolast nojoint prefix amax=exact|delayed`。FP4 的 d / w 条目默认 SR（梯度 dY），w2d；没写到的保持 FINAL（qkv f=bf16 d=mx8 w=mx8；proj/fc1/fc2 mx8；head bf16）。空串 = FINAL |
| `impl/fp4_sim.py` | `Sim`：专用 `torch.Generator`（SR）、`rht_signs` 符号向量（种子 20251003）、delayed-amax 表（每个调用点 `(prev_amax, inited)` 设备张量；`torch.where` 选择，无 host 同步）+ 诊断（超 2× 次数、cur/prev 最大比值、最差调用点）。`q_matrix / q_matrix_2d / q_row / q_col / q_weight`：`amax="exact"` 时与 `fp4_ref.quantize_matrix / quantize_matrix_2d / quantize / quantize_weight` 逐位一致（RN、给定 u 的 SR、RHT、2D/1D 权重；测试 b） |
| `impl/fp4_gemm.py` | 真实 QuACK FP4 GEMM：`fwd_plain/bias/res/gelu`、`dgrad`、`dgrad_dgelu`、`wgrad`（fp32 输出）；nv4 的 `ts = ts_a·ts_b` 以设备标量传给 epilogue |
| `impl/fp4_block.py` | r17 `fused_block_mx8.py` 的副本（复用 r13 的 `_LN1/Shadows/_Bundle/eligible`、r17 的 `mx8` / `gemm_mx8`）。`LayerQ`：每个 Linear 的 f/d/w 各自选格式；X 在前向只量化需要的副本（fwd 行副本、wgrad 转置副本），dY 只量化一次（dgrad 行副本、wgrad 转置副本）；权重每步在 block 0 量化（`Handle.begin_step`，同 r17 `Mx8Weights.refresh`）；`prefix`：wgrad 的 FP4 收缩只覆盖每张图的后 1024 个 patch token，prefix token（plain 块 5 个，joint 块 105 个）用 bf16 小 GEMM（fp32 输出）精确算后加回 |
| `impl/fp4_head.py` | r14 `head.py` 的副本：每个 ConvT 的 f/d/w 各自选 bf16（r14 的 QuACK / cuBLAS）/ mx8（r17）/ nv4 / mx4；d、w 支持 `+sr`，w 支持 `+rht`（沿像素维）。策略空串时与 r14 `fused` 逐位一致（测试 e） |
| `impl/patch.py` | `round20_fp4_probe`：覆盖 `ops.block` / `ops.predict`（旧的作回退），不改 Parameter 身份和 state_dict（apply_patches 会检查），`variant` = 策略串 |
| `tests/test_fp4_block.py` | a 策略解析；b `fp4_sim` 对 `fp4_ref` 逐位、delayed 语义、默认生成器未被碰；c 真实 QuACK FP4 GEMM 对反量化 fp64 乘积；d prefix 复现 R20a 的 block 0 数字；e 空策略对 r17 块 / r14 head；f 块级合理性 |
| `tests/gate_b_multi.py` | 从 r19 复制改写：一个进程里 R0、P1、FINAL 和所有候选共用同一次 build / 同一初始权重 / 同一批 / 同一 RngTape；candidate 在同一个 round20 ctx 里运行时切换策略；候选前后各跑一次 FINAL（`FINAL`、`FINAL_again`）确认运行间无漂移 |
| `tests/gate_s_multi.py` | 4 个场景 × 50 步 × 2 批；R0、FINAL 和候选同一进程；候选 `amax=delayed`；对 R0 和对 FINAL 的 EMA 偏差、NaN/Inf、grad-norm 比值、delayed 诊断 |
| `tests/cands.py`, `queue_*.sh`, `retry.sh`, `run_gateB.sh`, `summarize_r20b.py`, `show.py` | 候选表、队列（`retry.sh` 处理 `assert_gpu_exclusive` 误报）、汇总表 |
| `profile/` | `gateB_r20b_*_screen.json`、`*_finals.json`、`*_arms.json`、`gateS_r20b_*.json`、`test_fp4_block.json`、`tables_r20b.md` |

**与 PLAN 的差别**：PLAN_v3 说 nv4 / mx4 走“fake-quant 后的 bf16 GEMM”；按本任务和 6a 的要求改成“torch 量化 + **真实** QuACK FP4 GEMM”（R20a 已验证 fp4_ref 的操作数喂 QuACK 结果正确），所以累加顺序也和真 kernel 相同。量化噪声（RN / SR / RHT / delayed amax）是真的，只有“量化 kernel 的速度”是仿真的。

## 3. nsys 分析过程

**本轮没有 nsys，也没有整步速度测量**：R20b 是精度门（B / S），全部在 eager 下用 torch 量化器仿真，不能代表融合后的 kernel 时间，整步 graph 下的速度要等 R21/R22 的真 kernel。节省估计仍用 R20a 的 `per_step_estimate.json`（GEMM 实测 + 融合量化器估计）：

| 项 | 来源 | ViT-L | ViT-B |
|---|---|---:|---:|
| ① head dgrad+wgrad → nv4 | R20a §4.6（量化器 amax 由生产者给出） | 7.24 ms | 7.04 ms |
| ②a head 前向 → mx8 | 同上 | 3.30 ms | 3.05 ms |
| ③ 骨干 dgrad+wgrad → nv4 | 同上 | 7.86 ms | 4.29 ms |
| prefix 侧 GEMM 成本（plain 5 + joint 105 token） | R20a §4.3 实测 | 1.84 ms | 0.96 ms |
| nofirst 损失 ≈ ③ 的 1/24（L）/ 1/12（B） | 估计 | 0.33 ms | 0.36 ms |
| FINAL 步时（同机，PLAN §0） | | 101.8 ms | 86.6 ms |

另外，仿真过程中记录了每个候选一步 fwd+bwd 的 eager 时间（`gateB` json 的 `seconds`），只用于确认没有异常，**不是速度数据**。

**命令**（每个 python 进程从启动到退出持双锁，≤10 分钟；队列脚本见 `tests/queue_*.sh`）：

```bash
python tests/test_fp4_block.py --only a,b,c,d,e,f                  # -> profile/test_fp4_block.json
python tests/gate_b_multi.py --model vitl16 --cands H1 H2 H3 B1 B2 B3 B4 B5 B3+mx4 ... --seeds 0 1 2 --out profile/gateB_r20b_vitl16_screen.json   # 筛选，全部候选 + 对照臂 + 仅记录项
python tests/gate_b_multi.py --model vitl16 --cands H1 H2 H3 B3 B4 B5 --seeds 0..7 --per-param --out profile/gateB_r20b_vitl16_finals.json  # 决赛，≥8 批
python tests/gate_b_multi.py --model vitl16 --cands B4 B4+mx4 B4+rn B4+w1d B4+norht --seeds 0..7 --out profile/gateB_r20b_vitl16_arms.json   # 对照臂，≥8 批
python tests/gate_s_multi.py --model vitl16 --cands B3 H3 B4 --scenarios schedule compressed --out profile/gateS_r20b_vitl16_a.json            # S 门（两个进程各 2 个场景）
python tests/summarize_r20b.py --results                              # -> profile/tables_r20b.md, results.json["r20b"]
```

## 4. 实测结果

### 4.1 仿真块 / head 的单元测试（`tests/test_fp4_block.py`，`profile/test_fp4_block.json`）

| 测试 | 结果 |
|---|---|
| a 策略解析 | 通过（FINAL 默认值、别名、SR 默认、nofirst/nolast/nojoint 回到 FINAL、非法输入报错） |
| b `fp4_sim` 对 `fp4_ref` | `amax=exact`：nv4 / mx4 的 RN、RHT+SR（给定同一个 u）、转置副本 + RHT、2D / 1D 权重，数据字节与 scale 字节全部逐位一致；`delayed`：第一次调用 = exact，第二次 ts = 2 × 上一次 amax / 2688（逐位）；默认 CPU / CUDA 生成器状态在量化之后不变（RngTape 对齐不受影响） |
| c 真实 QuACK FP4 GEMM 分派（ViT-B plain 块形状 qkv / proj / fc1 / fc2，nv4 和 mx4） | fwd bias / res / gelu 对反量化 fp64 乘积相对 L2 = 1.66e-3（= bf16 输出舍入下限；fp32 输出的 `fwd_res` 的 out 为 0）；dgrad 1.66e-3（mx4 1.7e-3）；dgrad+GELU' 及 db 对参考 1.66e-3 / 0；wgrad（fp32 输出）与 prefix 版 wgrad 对反量化乘积 = 0（逐位）。门槛 ≤2e-3 |
| d prefix 复现 R20a block 0（ViT-L 缓存张量 `cap_vitl16.pt`） | nv4 wgrad 相对 L2：block0.fc1 0.2594 → prefix 后 0.0549；fc2 0.6045 → 0.0416；fc2+RHT 9.0914 → 0.0449；proj 0.1279 → 0.0948；qkv 0.0473 → 0.0496；block22.fc2 0.0919 → 0.0113。与 R20a `debug_prefix_tokens.json` 的数字在 1e-6 内一致（`fp4_block.LayerQ` 里的 prefix 实现 = R20a 的侧 GEMM 方案） |
| e 空策略（= FINAL）对 r17 块 / r14 head（ViT-B，plain 块 1 和 joint 块 11） | 块输出相对差 0.0（逐位），dx 相对差 1e-5（r17 / r13 dx 的原子累加不确定性，两次运行 r17 对 r17 也有）, 所有参数梯度余弦 ≥0.99999（`dW_cos_min`=1.0）；head：mask logits、dx、全部参数梯度、cls logits 与 r14 **逐位相同** |
| e head 策略对 r14（单次、合成输入，ViT-B） | `head.dw=nv4+sr`：mask logits 余弦 1.0000、dx 0.984、参数梯度余弦最小 0.9715（`upscale.0.conv1.weight`）；`head.f=mx8,head.dw=nv4+sr+rht`：0.9985 / 0.983 / 0.970；`head.dw=mx8`（退路）：1.0000 / 0.9993 / 0.9986 |
| f 块级合理性（对 R0 的默认 torch 块在 bf16 autocast 下的输出和梯度；输入 = 随机图经 `prepare_tokens` 得到的 token，**不含**真实训练里的 massive activation；ViT-B 块 0 / 1 / 11（joint），ViT-L 块 0 / 1 / 23（joint）；无 autocast 的 fp32 块在这里 SDPA 报 `No available kernel`，所以参考是 bf16 autocast 的 R0 块） | FINAL 块：out ≥0.9992，dx 0.9977–0.9988（ViT-L 块 0 为 0.9857），dW 最小 0.996–0.997（ViT-L 块 0 为 0.886，LN 参数）。`all.d=nv4+sr,all.w=nv4+sr+rht`：out 不变；**dx 0.970–0.982（ViT-B）、0.965–0.984（ViT-L）**；dW 最小 plain 块 0.907–0.975，joint 块 0.894（ViT-B）/ 0.961（ViT-L）；加 `prefix`：ViT-B joint 块 dW 最小 0.894 → 0.958，plain 块和 ViT-L 基本不变。`all.f=nv4`（前向）：out 0.942–0.9998，**dx 0.67–0.99**（ViT-B 块 0 为 0.67、块 1 为 0.74；ViT-L 块 1 为 0.83），dW 最小 0.07–0.92 → 前向 nv4 对单块反向的破坏远大于 dgrad / wgrad 的 nv4 |

### 4.2 筛选（B 门，同一进程、同一初始权重 / 批 / RngTape，n 见表）

候选表（`tests/cands.py`）：

| 编号 | 策略 |
|---|---|
| H1 | `head.dw=nv4+sr` |
| H2 | `head.dw=mx8`（退路） |
| H3 | `head.f=mx8,head.dw=nv4+sr` |
| B1 | H3 + `all.w=nv4+sr+rht` |
| B2 | B1 + `prefix` |
| B3 | H3 + `all.d=nv4+sr,all.w=nv4+sr+rht` |
| B4 | B3 + `nofirst` |
| B5 | B3 + `prefix` |
| 对照臂 | 以 B3 / B4 / B5 为底：`+mx4`（nv4→mx4）、`+rn`（SR→RN）、`+w1d`（权重 1D scale）、`+norht`（wgrad 不用 rht）；仅记录：`+fcF4`（`fc.f=nv4`）、`+qkvF8`（`qkv.f=mx8`） |

注意：**数据加载器在不同进程里对同一个 seed 给出的 batch 不完全相同**（R0 的 loss 在两次进程里不同：例如 ViT-B seed 0 为 30.475 / 30.098），所以只有“同一进程内”的比较是配对的；表里的行是各自进程内对 R0 / P1 的值，行与行之间 <0.001 的差别要当噪声。

{{TABLE:screen_vitl16}}

{{TABLE:screen_vitb16}}

### 4.3 决赛（B 门，≥8 批）

放宽门（对 R0 和对 P1 分别）：全局梯度余弦中位数 ≥0.98、最小 ≥0.97、loss_rel ≤2e-2；严格门只记录。ViT-L 的各候选 8 批来自 3 个进程（seed 0 / 2、seed 1、seed 3–7），每个种子内部所有候选同一批；ViT-B 的决赛重新在**一个进程**里跑了 seed 0–7（旧的混合进程版本保留为 `gateB_r20b_vitb16_finals_mixedproc.json`）。

{{TABLE:finals_vitl16}}

{{TABLE:finals_vitb16}}

### 4.4 对照臂（以 B4 为底，≥8 批，同一进程配对）

{{TABLE:arms_vitl16}}

{{TABLE:arms_vitb16}}

筛选阶段（3 批）对 B3 / B4 / B5 的所有对照臂结果见 §4.2 的两张筛选表（每行的底是名字前缀）。

### 4.5 S 门（4 场景 × 50 步 × 2 批，全部 eager，候选 `amax=delayed`；ViT-B 表里的 B3x / B4x 是 `amax=exact` 对照）

R0、FINAL、候选在同一进程里（ViT-L：schedule / compressed 一个进程、fixed_mid / fixed_late 一个进程；ViT-B：B3 / H3 两个进程，B4 / B3x / B4x 另外两个进程；**不同进程的批不同**，所以 B3 / H3 与 B3x / B4x 之间不是配对比较，B4 对 B4x 才是）；EMA(0.9) 损失偏差对 R0 和对 FINAL 都要 ≤5%，不能有 NaN / Inf；grad-norm 比值记录（门要求 0.1–10）。“检查点”= 每 10 步，“全 50 步”= 逐步取最大。

{{TABLE:gateS_vitl16}}

{{TABLE:gateS_vitb16}}

{{SGATE_NOTES}}

## 5. 正确性

- fp4_sim 对 fp4_ref 逐位一致，空策略对 r17 块 / r14 head 的一致性、prefix 对 R20a 的复现见 §4.1。
- 本轮没有改任何 Parameter / state_dict / 训练状态（`apply_patches` 在应用 patch 前后检查了 state_dict 键与形状），所以没有 export / resume；graph_equiv 不适用（仿真量化器是 eager 的 torch 算子，且 R20b 不发布栈）；这些检查属于 R22 / R23。
- B 门里 FINAL 在候选前后各跑一次（`FINAL` / `FINAL_again`），余弦对 FINAL 恒为 1.0000（逐位的 GEMM 和 r17 dx 的原子累加不确定性 ≤1e-5），所以候选间的差别不是运行间漂移。
- 逐参数最小余弦在 FINAL 自己身上就有很低甚至为负的值（例如 ViT-L 的 FINAL 对 R0 逐参数最小 −0.64）：这是梯度极小的参数（`qkv.bias` 的 k 部分被掩码、LN 偏置等）的噪声，对全局梯度没有贡献；按 Amendment 4 / 6 只记录不作门。

## 6. 静态分析与 roofline

不适用（没有写 kernel）。

## 7. 结论、推荐、失败项、契约变更请求、下一瓶颈

{{CONCLUSIONS}}
