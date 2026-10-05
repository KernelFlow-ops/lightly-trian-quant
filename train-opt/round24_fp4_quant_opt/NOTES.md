# Round24（R24）：FP4 量化器在真实步里的优化（热输入）（第三阶段，Sonnet 5.5 执行）

> 数字全部在 `infer-5090`（RTX 5090）上实测，持双锁；没测的写 null / not measured。`results.json` 机器可读。边做边写，被打断时已有内容可用。
> 没改 round0–23、common、PLAN、CONTRACTS、README；build 目录 `/root/autodl-tmp/train-opt-tmp/round24_fp4_quant_opt/`（没动 R21a 的 `.so`）。
> 状态（随时更新，14:10）：§1–§3、§4.1–4.5 完成，nsys（§4.2b）已出；整步 bench q2 对 P3B：L +1.28 %、B +0.74 %（不到 2 %）；graph_equiv 已过；门 A 全部分、B、S、q2d / q2p7 整步在队列里（`tests/chainA2.sh` `chainB.sh` `chainC.sh`），未出结果的项标 not measured。

## 0. 摘要（随做随更新，最终版见 §7）

| 项 | 状态 |
|---|---|
| 交付 | `impl/fp4_kernels2.cu`、`impl/fp4q2.py`（`fp4q` 的 drop-in，新参数 `philox_rounds`、`quant_weight(row=)`、`WEIGHT_ROW`、`set_fast`）、`tests/`（热输入基准 `bench_hot.py`、门 A 的 `test_fp4q2.py`、逐位对比 `quick_cmp.py`、SASS 分析 `sass_mix.py`、nsys 分解 `quant_breakdown.py`） |
| 热输入基线 | 见 §1.2（R21a `fp4q`；每步按调用点汇总 §1.1） |
| 优化 | 进行中，见 §2 |
| 门 / 整步 | not measured（排在 R23 的计时队列之后） |

## 1. 目标与基线

### 1.1 真实步里的 FP4 量化（R22 nsys，`profile/r22_quant_breakdown_*.json`，由 `tests/quant_breakdown.py` 读 R22 的 nsys 逐 kernel trace 得到；每步）

ViT-L（P3B，nofirst：20 个 plain 块 + 3 个 joint 块有 FP4 dgrad/wgrad），总计 **6.77 ms/步**：

| 调用点 | 形状 [M, C] | 次数/步 | 平均 µs | ms/步 |
|---|---|---:|---:|---:|
| dY 行+转置（dual，SR×2）fc1 的 dz | [7203,4096] | 20 | 78.2 | 1.564 |
| dY dual qkv 的 dqkv | [7203,3072] | 20 | 58.4 | 1.168 |
| dY dual proj / fc2 的 g | [7203,1024] | 40 | 24.8 | 0.990 |
| X 转置（col only）fc2 的 h | [7203,4096] | 20 | 37.2 | 0.744 |
| X 转置 qkv / proj / fc1 | [7203,1024] | 60 | 11.0 | 0.661 |
| joint 块（M=7903）上述各项合计 | – | 3 块 | – | 0.84 |
| 权重（fp32 → W、Wᵀ，2D scale）4 个形状 | – | 23 × 4 | 3.9–13.7 | 0.80 |

ViT-B（P3B_B，12 块），总计 **5.05 ms/步**：dY dual（3072 / 2304 / 768 / 768）1.03 + 0.78 + 0.58 ms（plain）+ joint 0.88，X col（3072 / 768）0.70 + 0.40 + joint 0.41，权重 0.27，roll 0.003。
结论：dY 的 dual + SR 占 L 的 63%（4.3 ms），X col 占 21%，权重占 12%。

### 1.2 热输入微基准（`tests/bench_hot.py`：CUDA graph 里，每次量化前先放一个写输入的 kernel `x.uniform_()`；只统计量化 kernel 的时间 = (producer+quant − producer)/reps；候选在同一进程交替；中位数 5 轮）

基线 = R21a `fp4q`（`profile/bench_hot_base0.json`，`profile/bench_hot_decomp.json`，µs）：

| 形状 | dY dual+SR（headroom 8） | dual（RN） | row（RN） | row+SR | col+SR | X col（RN） |
|---|---:|---:|---:|---:|---:|---:|
| [7203,4096] | 71.6 | 53.8 | 30.6 | 43.3 | 38.4 | 33.5 |
| [7203,3072] | 54.3 | 41.7 | 22.9 | 30.2 | 32.0 | – |
| [7203,1024] | 20.9 | 17.1 | 9.9 | 11.6 | 12.0 | 10.9–12.1 |

与 nsys 真实步相比（78.2 / 58.4 / 24.8 µs 和 37.2 / 11.0 µs），热输入微基准低约 10%（真实步还有别的 kernel 同时占 L2 和 SM），所以可以用它排优化顺序。
每步总量（热输入基准按次数加权，dy + x + 权重）：ViT-L 6.15 ms，ViT-B 4.33 ms（nsys 为 6.77 / 5.05）。
参照：r17 的 mx8 `quant_dual` 暖输入 [7203,1024] 7.7 µs，`mx8.quantize(row)` 热输入 [7203,1024] 4.95 µs、[7203,4096] 18.0 µs。

## 3. nsys / SASS 分析过程（瓶颈在哪）

1. **每步分解**：用 R22 的 nsys 逐 kernel trace（`/root/autodl-tmp/train-opt-tmp/round22_fp4_backbone/nsys/*_cuda_gpu_trace.csv`）按 grid 形状分调用点（§1.1）。
2. **静态指令分布**（`cuobjdump -sass` + `tests/sass_mix.py`，R21a 的 act_kernel）：dual（bf16）4096 条静态指令，row-only 3480，col-only 3056；LOP3 / FADD / FFMA / IMAD 为主，CALL 65 / FCHK 60 / MUFU 60（`__fdiv_rn` 的慢路径展开），因为 RHT / SR / prefix / div_true 都是运行时开关，各分支都在同一个 kernel 里。
3. **热输入下的每元素时间**：[7203,4096] row-only 30.6 µs，≈ 2.4 TB/s；同形状 r17 mx8 row 18 µs，≈ 4.9 TB/s。

## 2. 改动（全部在 `train-opt/round24_fp4_quant_opt/`；`impl/fp4_kernels2.cu` + `impl/fp4q2.py` 从 R21a 复制后修改，R21a 文件未动）

API 与 `fp4q` 完全兼容（`quant_act` / `quant_weight` / `FP4State` / `FP4Q` / `headroom` / 监控 / `philox_uniform` 签名不变），新增的全部带默认值：

| 新增 | 默认 | 说明 |
|---|---|---|
| `quant_act(..., philox_rounds=None)` / `philox_uniform(..., rounds=None)` | 模块级 `PHILOX_ROUNDS`（env `FP4Q2_PHILOX`，默认 10） | 7 = Philox4x32-7，只改 kernel 内部产生的 SR 随机流；显式 `u_row` / `u_col` 路径不受影响；10 轮与 R21a 逐位相同 |
| `quant_weight(..., row=None)` | 模块级 `WEIGHT_ROW`（env `FP4Q2_WEIGHT_ROW`，默认 True = R21a 行为） | `row=False` 只产出 W^T 副本（`qt`、`sft`、`ts`），`q` / `sf` 为 None。R22 的 `prep_weight` 取完 `quant_weight` 立刻丢弃 `q` / `sf`；`run.py` 在 `q2` 预设里设 `WEIGHT_ROW=False` |
| `FP4State.parts` / `FP4State.fold()` | – | amax 发布改为分散到 16 个 32 B 扇区（见 §3），`roll()` 在 graph 内折回 `amax[1]`；`fold()` 供测试 / 查看（把 `parts` 折进 `amax[1]` 并清零） |
| `extension().set_fast(bool)` | True | False = 只用通用 kernel（A/B 与测试） |
| env `R24_VARIANT` / `R24_DEFS` / `R24_SRC` | – | 仅实验用：独立 build 目录 / nvcc 宏 / 临时 .cu |

### 2.1 优化项

| # | 改动 | 状态 | 热输入收益（µs，R21a → fp4q2） |
|---|---|---|---|
| (e) | **amax 发布去热点**（本轮最大的一项，见 §3）：R21a 每个 warp 对同一个地址做 `volatile load + atomicMax`，整个 grid 的所有 warp 串行到同一条 L2 线上；改成 CTA 内先做 warp / smem 归约，每 CTA 每副本一个 fire-and-forget `RED.MAX`，分散到 `NPART = 16` 个 32 B 扇区；`roll()` kernel 折回 | 完成 | [7203,4096] row-only 31.9 → 14.0（2.3×），dual（RN）54.4 → 33.8（1.6×），X col 33.5 → 24.7 |
| (a) | **`act_kernel` 的每元素指令数**：新增快路径 `actf_kernel`（nv4、bf16 输入、无 RHT、无 prefix、无显式 u；RN 或 Philox-SR）：ROW / COL / SR / Philox 轮数全是编译期参数，不带 RHT / SR / prefix / `div_true` 的死代码；线程 = 一个 16 元素块（R21a 是两个 lane 各 8 个，block scale、`1/denom` 在两个 lane 上重复算），amax 用 `VIMNMX.U16x2` 在 bf16 对上做；块 scale 的 `t / ts` 用提前算好的 `RN(1/ts)` 做 Markstein（`ts ∈ [2^-60, 2^60]` 内与 IEEE 除法逐位相同，之外退回逐元素 `__fdiv_rn`）；tile 以原始 bf16 进 smem（16 B 存），列副本读相邻两列的 32 位字（XOR 16 swizzle，无 bank 冲突）；SR 用整数进位公式（|v|≥1：`m22 > 32(2h+1)` ⇔ 一次加法 + 移位）和 `FADD.RP` 取整比较（|v|<1），与 R21a 同一个决策 `u < p`、同一个 Philox 流；sign 修复用单条 `LOP3`；X 的 tile 加载放在 per-tensor scale 加载之前（`__ldg`，ts_out 的写放到 kernel 末尾） | 完成；RN / SR / 状态与 R21a 逐位相同（§4.1） | 静态指令数：row-only RN 3480 → 872，dual RN 4096 → 1016（含 `__fdiv_rn` 回退路径 2128）；SR 变体见 §4.2 |
| (c) | 权重量化器 `row=False`（不写用不到的 nv4 行副本 `q` / `sf`）；kernel 对 `WROW` 做模板 | 完成 | 热输入微基准里权重已在 L2 且 `row=True` 默认，没有测到收益；真实步里权重来自 fp32 master（DRAM），收益只在整步里体现 |
| (d) | Philox 7 轮开关（`philox_rounds=7`；编译期模板 `philox_t<NR>`） | 完成，默认关 | SR 路径见 §4.2；统计检验见 §4.4 |
| (b) | X 的 mx8 行副本 + nv4 转置副本合并成一个 kernel | **没做** | 见 §7：热输入下 r17 `mx8.quantize(row)` [7203,1024] 4.95 µs、[7203,4096] 18.0 µs（每步 L 约 24 块 × 27.9 µs = 0.67 ms，B 同量级）是合并能省的上限，合并后还要付 mx8 转换，预计节省不到整步 1%，要在 R22 的块代码里换调用点，预算内放弃 |

## 3. nsys / SASS 分析过程（瓶颈在哪，如何确认）

1. 起点（§1）：热输入下 act_kernel 比 L2 下限慢 2 倍，R21a 的假设是"每元素指令数"。
2. **静态 SASS**（`cuobjdump -sass` + `tests/sass_mix.py`）：R21a 的 dual 4096 条、row-only 3480 条静态指令，LOP3 / FADD / FFMA 为主，大量 `FCHK` / `CALL`（`__fdiv_rn` 慢路径）和 RHT / SR 的死分支。新快路径 row-only RN 872、dual RN 1016（含回退路径的 2128，去掉回退路径的编译实验见 `R24_ASSUME_OK`）。
3. **指令数不是主瓶颈的证据**（骨架消融，`profile/bench_hot_abl*.json`、`bench_hot_skel.json`、`bench_hot_nopub.json`，ViT-L，热输入，µs；快路径 kernel 的变体，宏 `R24_SKEL`（去掉全部量化计算，只剩加载 → 异或 → 存储）、`R24_NOQ` / `R24_NOSF`（不存 q / 不存 scale 字节）、`R24_NOLD`（不读 X）、`R24_COAL`（4 个完全合并的 16 B 加载）、`R24_NOPUB`（不发布 amax））：

| 变体 | row [7203,4096] | row [7203,1024] | X col [7203,4096] | X col [7203,1024] | dual RN [7203,4096] |
|---|---:|---:|---:|---:|---:|
| 完整 kernel | 27.1 | 7.3 | 36.9 | 9.6 | 56.3 |
| 骨架（无量化计算） | 28.4 | 7.4 | 37.7 | 9.2 | 59.4 |
| 骨架 + 不存 q、不存 sf | 26.2 | 5.8 | 37.7 | 9.4 | – |
| 骨架 + 不读 X | 20.5 | 5.8 | 35.0 | 8.2 | – |
| 骨架 + 合并加载（`COAL`）、不存 | 22.5 | 6.0 | 40.8 | 9.5 | – |
| **完整 kernel，不发布 amax** | **13.8** | **4.2** | **23.6** | **5.6** | **31.0** |

   - 去掉全部计算，时间不变 → 不是 ALU 受限；合并 / 非合并加载一样、去掉存储也不降 → 不是访存模式；**只有去掉 amax 发布才降（row 2.3×、dual 1.8×）**。
   - 原因：`amax_publish` = 每个 warp 一次 `volatile` 读 + 可能的 `atomicMax`，目标是同一个 32 位地址（dual 时同一条 128 B 线上的两个地址）：[7203,4096] 有 7296 个 CTA × 4 warp = 2.9 万次请求落在同一个 L2 扇区上，约 1 次 / L2 时钟的串行吞吐，正好等于 R21a 的 act kernel 时间。R21a 的 generic kernel 有同样的结构，所以 §1 里"比 L2 下限慢 2 倍"的真正原因是这个而不是指令数。
   - 处理：每 CTA 一次（`__syncthreads` 后 thread 0），写到 `parts[site*2+copy][cta % 16]`（每个子槽独占一个 32 B 扇区），RED 不带返回值；`amax_roll_mon_kernel` 读 16 个子槽取最大再清零，折进 `next`，其余语义（`prev <- next`、监控、seed / step）不变。
4. **量化 SR 之后仍是指令受限**：去掉 amax 热点后，SR 变体 dual 59.4 µs（RN 31.0），row-only SR 33.6（RN 14.0）：SR 路径每元素约 28 条额外指令（Philox 约 7 + 决策公式约 20），issue 利用率约 85%（按热输入吞吐和静态指令数估计），所以 SR 的开销只能靠减指令。已做：SR 路径不再做符号修复、h 只提取一次、阈值常量合并；未做：见 §7。

## 4. 实测结果（持续更新；没有测的写 not measured）

### 4.1 逐位对比（门 A 的核心，`tests/quick_cmp.py`，`tests/test_fp4q2.py --part fast`）

`fp4q2` 快路径 = `fp4q2` 通用路径（`set_fast(False)`）= R21a `fp4q`，逐位相同：q / qt / sf / sft / ts / tst 字节，delayed amax 状态（`fold()` 之后）、`roll()` 之后的 amax / seed；形状 10 种（7203×1024、7903×3072、1029×768、33×128、257×192、14406×768、7168×4096、128×64、300×320、15806×2304）× 数据 8 种（高斯、尖峰、尖峰 + 1e3 通道 + 1e4 massive 行、全 0、1e-30、1e4、-0.0、2 的幂）× row / col / dual × RN / SR（Philox-10）× delayed（3 步，headroom 8，每步 `roll`）/ exact；权重 7 种形状 × delayed / exact，`row=True` 与 R21a 逐位相同，`row=False` 的 W^T 副本逐位相同。
结果：`quick_cmp.py` 818 例，0 差异（`logs/quick_cmp5.log`）；完整的 `--part fast` 见 §4.3 门 A 表。

### 4.2 热输入微基准（`profile/bench_hot_v5.json`，µs，R21a 与 fp4q2 同进程交替，graph 回放，5 轮中位数）

见 `profile/tables_r24.md` 的完整表。要点（ViT-L / ViT-B 的真实调用点形状）：

| 调用点 | R21a | fp4q2 | 倍数 |
|---|---:|---:|---:|
| row-only RN [7203,4096] | 31.9 | 14.0 | 2.28× |
| dual RN [7203,4096] | 54.4 | 33.8 | 1.61× |
| **dY dual + SR（headroom 8）[7203,4096]** | 73.6 | 62.3 | 1.18× |
| dY dual + SR [7203,3072] | 58.3 | 48.8 | 1.19× |
| dY dual + SR [7203,1024] | 21.7 | 17.1 | 1.27× |
| X col [7203,4096] | 33.5 | 24.7 | 1.36× |
| X col [7203,1024] | 11.3 | 7.2 | 1.57× |
| 每步合计（dy + X + 权重，按真实次数加权） | ViT-L 6.29 ms | 5.09 ms | 1.23× |
| 每步合计 | ViT-B 4.44 ms | 3.49 ms | 1.27× |

dY 的 SR 路径仍是指令受限（每元素约 44 条静态指令，dual SR 102 寄存器），占剩余量化时间的大头。

### 4.3 整步速度（StepGraphRunner，三相 0.2/0.6/0.2，warmup 20，50×3；每个候选一个进程，A-B-C-C-B-A 交替，同一场次；`profile/bench_vit*_graph_all*.json`，汇总 `profile/tables_r24.md`）

| 模型 | 栈 | img/s（各次） | 平均 img/s | mid 步时 ms | 对 FINAL | 对 R22 的 P3B |
|---|---|---|---:|---:|---:|---:|
| ViT-L | FINAL | 68.14, 68.51 | 68.33 | 102.53 | 1.000 | – |
| ViT-L | P3B（R22 fp4q） | 72.49, 72.58 | 72.54 | 96.65 | 1.062 | 1.000 |
| ViT-L | **P3B_q2（fp4q2）** | 73.45, 73.48 | **73.46** | **95.41** | 1.075 | **1.0128（+1.28 %）** |
| ViT-B | FINAL | 159.43 | 159.43 | 87.97 | 1.000 | – |
| ViT-B | P3B_B | 166.45 | 166.45 | 84.32 | 1.044 | 1.000 |
| ViT-B | **P3B_B_q2** | 167.65, 167.69 | **167.67** | **83.70** | 1.052 | **1.0074（+0.74 %）** |

（上表是已完成的次数；队列里还有 P3B_B / FINAL 的重复，最终数字以 `results.json["bench_step_summary"]` 为准。）
**整步比 P3B 快 1.28 % / 0.74 %，都不到发布线 2 %。** 量化器本身（热输入）每步省 1.2 ms（L）/ 0.95 ms（B）：与整步的节省（mid 步时 −1.24 ms / −0.62 ms）一致，微基准没有高估。ViT-L 要到 ≥2 % 需要再省 0.7 ms，ViT-B 再省 1 ms。

### 4.4 Philox 7 轮（选项 d，`philox_rounds=7`，默认 10）

热输入微基准（`profile/bench_hot_p7.json`，µs，同进程交替，只有 SR 路径受影响）：

| 调用点 | 10 轮 | 7 轮 | 变化 |
|---|---:|---:|---:|
| dY dual+SR [7203,4096] | 61.8 | 57.8 | −6.5 % |
| dY dual+SR [7203,3072] | 48.8 | 46.1 | −5.6 % |
| dY dual+SR [7203,1024] | 16.7 | 15.6 | −6.2 % |
| dY dual+SR [14406,3072]（B） | 91.9 | 86.3 | −6.1 % |
| dY dual+SR [14406,2304] | 74.6 | 70.6 | −5.4 % |
| dY dual+SR [14406,768] | 25.6 | 22.3 | −12.7 % |

统计检验（`tests/test_fp4q2.py --part philox7`，结果见 §4.6 门 A 表；BigCrush 没有跑，依据 Salmon et al. 2011：Philox4x32-7 通过 BigCrush，10 轮是保守余量）：核内 7 轮流 == 显式 u = `philox_uniform(rounds=7)` 经通用 kernel 的结果（逐位）；7 轮只改 SR 噪声（scale 字节、ts、RN 输出不变）；7 轮流的均值 / 标准差 / 16 位比特平衡 / 256 格卡方 / 64×64 对卡方 / 序列相关 / 跨 word1、跨 seed 相关 / 计数器与密钥位的雪崩（理想 0.5）。
**默认 10 轮不变**（与 R21a 逐位相同）；7 轮只在 `philox_rounds=7` 或 env `FP4Q2_PHILOX=7` 或 `run.py` 的 `*_q2p7` 预设下启用。

### 4.5 SR "dither" 变体（`impl/fp4q2_srd.py` → 正式文件 `impl/fp4q2d.py` + `impl/fp4_kernels2d.cu`；**可选，默认不启用**）

**改了什么**：SR 的舍入判决公式。R21a / fp4q2 默认（`sr_mode="exact"`）：16 位随机数 h → u = (2h+1)/2^17，向上取整当且仅当 u < p（p = 到下邻点的归一化距离；|v|≥1 用尾数位技巧，|v|<1 用 FADD.RP 取整比较）。dither（`sr_mode="dither"`）：|v|≥1 同样的进位技巧，直接得到取整后的网格值（无需再比较）；|v|<1 时把 `|v| + (2h−65535)·2^-18` 交给硬件 RN（噪声 ±0.25，不会跨到别的间距的格子）。目的：每元素少约 15 条指令。
**与标准 SR 的统计区别**（`profile/sr_dither_check.json`：对 3688 个 |v|（网格点、中点、±1 ulp、随机、极小、>6）穷举全部 65536 个 h）：

| 规则 | 最大 \|偏差\|/格距 | 最大 \|P(上取整) − p\| | 最大 \|方差 − p(1−p)s²\|/s² | 网格点映射到自身 |
|---|---:|---:|---:|---:|
| exact（R21a） | 7.6e-6 | 7.6e-6 | 7.6e-6 | 100 % |
| dither | 7.6e-6 | 7.6e-6 | 7.6e-6 | 100 % |

两条规则的偏差、上取整概率、方差**完全一致**（都等于 2^-17 的随机数粒度），所以 dither 仍然无偏，统计性质与 exact 相同；区别只是对同一个 h，两者的具体判决可能不同（逐元素不逐位相同；Philox 流、计数器、种子、scale、RN 输出都不变）。核内无偏性（256 个种子的均值 / 预期噪声）见 `profile/test_q2d.json`（§4.6）。
**微基准**（`profile/bench_hot_srd.json`，ViT-L，µs，同进程交替，`srd_exact` 是同一份 .cu 里的 exact 规则作对照）：

| 调用点 | fp4q2 exact | dither | 变化 |
|---|---:|---:|---:|
| dY dual+SR [7203,4096] | 61.3 | 50.5 | −17.5 % |
| dY dual+SR [7203,3072] | 46.5 | 38.7 | −16.8 % |
| dY dual+SR [7203,1024] | 16.3 | 13.5 | −17.1 % |
| 每步合计（dy+x，plain 块，ViT-L） | 2.81 ms | 2.32 ms | −0.48 ms |

**是否接受**：它没有改变 SR 的统计性质（穷举结果与 exact 相同），但它不是"同一个 u 下与 R21a 逐位相同"，所以**默认关闭**，作为独立可选项（预设 `P3B_q2d` / `P3B_B_q2d`，环境变量 `FP4Q2_SR=dither`）；整步数字见 §4.7，B / S 门**没有对 dither 单独跑**（not measured，若要采用需要补）。

### 4.2b nsys：真实步里的 FP4 量化器（`profile/nsys_report24.{json,txt}`、`profile/q1_quant_breakdown_*.json`；graph 模式，5 步窗口，同一会话先后抓 P3B 与 P3B_q2；每步 ms）

| 类别 / 指标 | ViT-L P3B（R22 fp4q） | ViT-L P3B_q2 | ViT-B P3B_B | ViT-B P3B_B_q2 |
|---|---:|---:|---:|---:|
| **FP4 量化（act / w kernel + roll）** | **6.78** | **5.36**（−1.42，−21 %） | **5.03** | **4.35**（−0.68，−13 %） |
| MXFP8 量化（r17 quant_dual，不变） | 1.63 | 1.64 | 1.11 | 1.12 |
| GEMM NVFP4（不变） | 8.50 | 8.53 | 5.10 | 5.11 |
| kernel 总时间 | 93.77 | 92.38 | 82.71 | 82.06 |
| 窗口 / 步 | 95.77 | 94.72 | 84.18 | 83.38 |
| kernel 数 / 步 | 2821 | 2821 | 1822 | 1822 |

真实步里的分项（µs / 次，R22 fp4q → fp4q2；`tests/quant_breakdown.py`）：

| 调用点 | ViT-L | ViT-B |
|---|---|---|
| dY dual+SR，大形状（fc1 的 dz / qkv 的 dqkv） | 78.4 → 63.6（[7203,4096]）、59.0 → 46.2（[7203,3072]） | 114.3 → 94.3（[14406,3072]）、87.7 → 77.6（[14406,2304]） |
| dY dual+SR，小形状 [M,C] C = 1024 / 768 | 24.8 → 16.1 | 32.7 → 23.0 |
| X col，大形状（fc2 的 h） | 36.9 → 35.6（[7203,4096]） | 75.6 → **82.3**（[14406,3072]，**变慢**） |
| X col，小形状 | 11.0 → 6.9 | 14.7 → 10.7 |
| 权重 w_kernel（不写行副本） | 13.7 → 13.0、9.0 → 8.1、8.3 → 7.4、3.9 → 3.6（4 个形状；合计 0.80 → 0.74 ms） | 0.27 → 0.25 ms |

**真实步与热输入微基准的差别**：X col 的大形状（输入 59–88 MB，接近或超过 96 MB L2）在真实步里受 DRAM 限制，kernel 已在带宽下限上，改指令数没有收益（冷输入基准 `profile/bench_hot_cold1.json`：[7203,4096] 46.5 → 46.4 µs，[14406,3072] 70.8 → 70.9 µs，dY dual+SR 76 → 64 µs、115 → 98 µs；热输入微基准里 X col 的 33.5 → 24.7 µs 是输入完全驻留 L2 时才有）。按真实顺序（producer → r17 mx8 行量化 → nv4 列量化，`--sites xseq`，只计列量化）：[7203,4096] 41.0 → 25.8 µs、[14406,3072] 60.0 → **62.2** µs（略慢 3.7 %）、[14406,2304] 39.2 → 22.2、小形状 12.2 → 7.7、15.5 → 10.1。B 的 [14406,3072] 这一点（每步 12 次，约 +0.07 ms）没有找到原因（`R24_COAL` 完全合并加载的实验在 L2 命中时没有差别，DRAM 受限时的差别没单独测）。

### 4.2c 热输入微基准里没有收益的点（实测）

- `quant_weight(row=False)`（`profile/bench_hot_wr0.json`，权重已在 L2 的热输入）：ViT-B 全部权重每步 224 → 209 µs（−15 µs），可忽略；真实步里 nsys 的权重量化 0.80 → 0.74 ms（L）/ 0.27 → 0.25 ms（B）。
- 去掉 `__fdiv_rn` 回退路径（`-DR24_ASSUME_OK`，`profile/bench_hot_nosl2.json`）：只快约 1–2 %，不值得为此改语义。
- 骨架消融、合并加载、去 amax 发布的结论见 §3。
