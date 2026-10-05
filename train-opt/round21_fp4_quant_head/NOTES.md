# Round21（R21a）：融合 FP4 量化器（nv4 / mx4，一次读入出行副本 + 转置副本）（负责：Sonnet，第三阶段）

> 全部数字在 `infer-5090`（RTX 5090）上实测，持双锁，CUDA graph 回放、同一进程内候选交替、3 轮取中位数。没测的写 null / not measured。**本文件只覆盖 R21a**（量化器 + 微基准）；head 集成（R21b）、骨干（R22）、B/S 门、整步 bench、nsys 都不在本任务内，没有数据。
> 没改 round0–20、common、PLAN、CONTRACTS；只 import 了 `round20_fp4_probe/impl/fp4_ref.py`（逐位参照）和 `fp4_cands.py / fp4_epi.py`（GEMM 端到端测试）。

## 0. 摘要

| 项 | 结果 |
|---|---|
| 交付 | `impl/fp4_kernels.cu`（sm_120a）、`impl/fp4q.py`（接口）、`tests/test_fp4q.py`（Gate A）、`tests/bench_fp4q.py`、`tests/per_step_quant.py`、`tests/make_results21a.py` |
| Gate A | **全部通过**（`profile/gateA_fp4q.json`，8 个部分 0 失败）：RN 逐位对 fp4_ref（721 组）、SR 显式 u 逐位 + Philox 流对拍 + 无偏性 + 同种子同结果 + 换 step 结果变、exact ts 逐位、2D 权重逐位（84 组）、状态机 / CUDA graph 重放等价、QuACK nv4 GEMM 端到端相对 L2 ≤ 6e-8（门槛 2e-3）、硬件 RN 对 2^32 个 fp32 位型穷举 0 差异 |
| 微基准（冷输入，DRAM 常驻；28 个形状） | dual（行 + 转置）= **0.98–0.99× DRAM 下限**（16 个大形状 R·C ≥ 2.5e7），1.03–1.23×（12 个小形状，中位 1.14×）；dual + RHT(col) 与 dual 相同（免费）；加 SR 后 1.22–1.28×（大）/ 1.29–1.53×（小）；RHT + SR 1.27–1.33×（大）/ 1.34–1.58×（小）；独立 amax 遍历（exact）1.25–1.72×；mx4 dual 0.93–0.98×。对 r17 mx8 `quant_dual`：同一形状快 1.03–1.33×（字节少）；对 `x.t().contiguous()` 快 3.8–4.6× |
| 每步量化开销（按 R20a 的计数方式，冷输入） | ① head d+w：ViT-L 1.77 ms / ViT-B 2.65 ms（纯新增）；③ 骨干 d+w：FP4 量化 6.74 / 4.95 ms，相对 FINAL 的 mx8 量化净增 +1.66 / +1.09 ms（R20a 的估计是 +0.27 / +0.65） |
| 静态 | nv4 act kernel 81–96 寄存器（mx4 dual 136）/ 权重 ≤ 58，0 spill，smem 17 KB / 20 KB |
| 重要发现 | `fp4_ref` 在 CUDA 张量上算的 ts 与契约 12.2 的"fp32 除法"**不是逐位一致**（见 §5、§7 契约请求）；kernel 两种都支持（`div_true`） |

## 1. 目标与假设

| 假设（来自 R20a / PLAN_v3） | 实测 |
|---|---|
| 融合 dual 量化器能到 DRAM 下限附近（R20a 估计用 r17 mx8 × 字节比 × 1.25） | 大形状 0.98–0.99×，**比 R20a 的估计更好**；小形状 1.03–1.23× |
| nv4 的 amax 不能额外整遍扫描 | 实现了 `delayed`（上一步 amax × 2，当前 amax 由 kernel 顺带 atomicMax）；exact 模式（额外 amax 遍历）实测 +25–70% |
| RHT / SR 的额外开销 | RHT（沿 token 轴 FWHT16）免费；SR（软件 Philox）+22–28%（大形状），见 §4 |

## 2. 改动（全部在 `train-opt/round21_fp4_quant_head/`）

### 2.1 接口（`impl/fp4q.py`）

```python
q = fp4q.quant_act(x, fmt="nv4"|"mx4", row=True, col=True, rht_row=False, rht_col=False, sr_row=False, sr_col=False,
                   prefix=(B, N, npre)|None, rpad=None, amax="delayed"|"exact"|"ts", state=FP4State, site=int, ts=None,
                   seed=None, sr_offset=0, u_row=None, u_col=None, div_true=False) -> FP4Q
q.q, q.sf, q.ts          # 行副本 [R, C/2], blocked scale, ts[1]（nv4）
q.qt, q.sft, q.tst, q.kt # 转置副本 [C, Kt/2], blocked scale, ts[1], Kt
q.xp                     # prefix 模式：bf16 [B*npre, C]，prefix 行原样收集
w = fp4q.quant_weight(w, fmt, amax="delayed"|"exact"|"ts", state=, site=, ts=)   # q [N,K/2], qt [K,N/2] (= W^T 沿 N)，2D scale，RN
st = fp4q.FP4State(n_sites)   # amax [2(prev,next), n_sites, 2(row,col)], 设备端 seed / step 计数器
st.roll()                      # 每步一次：prev<-next, next<-0, seed+=1, step+=1（设备端，可进 CUDA graph）
```

- 输入 bf16 / fp32 连续 `[R, C]`，`C % 64 == 0`；权重 `N, K % 128 == 0`。输出的 blocked scale 布局与 `fp4_ref.to_blocked` 一致（nv4 `rk = ceil(C/64)`，mx4 `rk = ceil(C/128)`，尾部不足 128 行 / 4 块的部分写 0 字节）。
- **prefix 模式**：token `b·N + t`；每张图前 `npre` 个 token 为 prefix，不进转置副本（`Kt = B·(N-npre)`，token `b·N+npre+p` → 列 `b·(N-npre)+p`），转置副本的 `tst`（amax）只统计 patch token；prefix 行以 bf16 收集到 `xp`。行副本覆盖全部 R 个 token。要求 `(N-npre) % 64 == 0` 且 `B·(N-npre) % 128 == 0`（ViT-L/B 的 1024 个 patch 满足）。
- **amax 模式**：`delayed`（默认）ts = 上一步 amax × 2 / 2688，首次（python 标志，需在 graph 捕获前 eager 跑一次）用 exact 初始化 prev；当前 amax（RHT 之后的值，转置副本只含 patch token）atomicMax 写 next 槽。`exact`：独立 amax kernel（`quant_act(..., amax_only=True)`）后再量化。`ts`：显式传入（测试用）。
- **SR**：Philox4x32-10，key = 设备 int64 seed（`state.seed`，`roll` 每步 +1），counter = `(flat_index/8, 2·sr_offset + copy, 0, 0)`，每次 Philox 调用的 4 个 32-bit 字拆成 8 个 16-bit 半字，uniform = `(2h+1)/2^17`（开区间，粒度 2^-17）。`sr_offset` 是调用点固定偏移。`u_row / u_col`（fp32 [0,1)）覆盖生成器，用来和 `fp4_ref.e2m1_sr(v, u)` 逐位对拍；`fp4q.philox_uniform(n, seed, word1)` 给出 kernel 同一条流（测试里据此逐位对拍）。
- **div_true**：见 §5。默认 False（与 `fp4_ref` 在 CUDA 张量上的结果逐位相同）。

### 2.2 kernel 设计（`impl/fp4_kernels.cu`）

- CTA 128 线程，tile = 64 token × 64 列。读：每线程 4 行 × 8 连续列（16 B bf16 / 32 B fp32），行副本直接在寄存器里做（RHT-16 用两线程 shfl 蝶形、amax 用 shfl、`cvt.rn.satfinite.e2m1x2.f32` 打包），整 tile 同时以 fp32 写入 smem（ld = 68，无 bank 冲突）。
- 转置副本：每线程取一列的 32 个 token（nv4 两个 16 组 / mx4 一个 32 组），从 smem 读出，做 FWHT16（寄存器蝶形，h = 1,2,4,8，与 fp4_ref 相同的加减顺序）→ ×0.25 → 符号，求 amax、量化，一次写 16 B 到 `qt`（同一 32 B sector 的另一半由同 CTA 另一个 warp 写，所以每列每 tile 恰好整 sector）。scale 字节直接写 blocked 布局。
- nv4 的 `v = x / (float(sf)·ts)` 要求 IEEE 除法位级一致：同一 block 共用分母，用 `r = RN(1/d)`、`q = RN(y·r)`、`res = fma(-d, q, y)`、`q' = fma(res, r, q)`（Markstein 修正，正规范围内与 `__fdiv_rn` 位级相同，范围外退回 `__fdiv_rn`），`tests/test_fp4q.py --part div` 在 2.8×10^10 个样本上与 `__fdiv_rn` 比较 0 差异（d 的尾数含全 1 / 全 0，y 取 e2m1 判决阈值 ±3 ulp）。
- SR 的 e2m1 码用位技巧：|v|≥1 时下邻点码 = `(bits>>22) - 252`、概率 = 低 22 位尾数 / 2^22；|v|<1 时间距 0.5。与 `fp4_ref.e2m1_sr` 在 4×10^7 个随机 / 特殊输入上逐位一致。
- 权重 kernel：64×64 tile 读 fp32 / bf16 一次，同时出 `q`（沿 K）和 `qt`（W^T 沿 N，码字就是 `q` 码字的转置，经 smem 转置）；2D scale（nv4 16×16、mx4 32×32）使两份副本数值一致。
- 所有 kernel 无 host 同步、无数据相关形状，可被 CUDA graph 捕获（Gate A `state` 部分捕获 roll + 激活量化 + 权重量化，重放结果与 eager 等价，状态机一致）。

## 3. nsys 分析过程

本任务是算子级（量化器 + 微基准），没有整步改动，**没有 nsys**。分析方法：
1. 以实测 D2D copy 带宽（1528 GB/s，读 + 写）作为 DRAM 下限，下限 = (读 + 写字节数) / 带宽；字节数含 scale（nv4 每元素 0.5 + 1/16 B，mx4 0.5 + 1/32 B）、prefix 模式的 bf16 `xp`。
2. "冷"：每个形状轮换足够多份互不相同的输入缓冲（总量 > 4.2 × 96 MB L2），测的是 DRAM 路径（可与下限直接比较）；"暖"：同一输入反复量化（≤ 15 MB 的小张量留在 L2，对应真实流水线里生产者 → 量化器的 L2 命中）。
3. 对照：r17 `mx8.quantize`（`quant_dual`，同一进程同一 graph 回放）、`x.t().contiguous()`、R20a `profile/fp4_quant_cost.json`（Spark / QuTLASS，其他 session 测的）。
4. 瓶颈判断依据：大形状 dual = 下限，说明 DRAM 受限；SR 开销随元素数线性（+22–28%）说明是 ALU 发射受限；小形状（≤15 MB）冷 / 暖差别小而 dual 比行 / 列各自慢，说明是波次 / 单 CTA 延迟受限（见 §4.3）。

## 4. 实测结果

### 4.1 Gate A（`profile/gateA_fp4q.json`，`tests/test_fp4q.py --part all`）

| 部分 | 内容 | 用例数 | 失败 |
|---|---|---:|---:|
| rn | nv4 / mx4 × 行 / 转置 × RHT 开 / 关 × prefix 开 / 关；R ∈ {7203, 7903, 14406, 15806, 1029, 33, 28672, 57344} × C ∈ {768, 1024, 2304, 3072, 4096}（数据类型轮换：高斯 / 尖峰 scale-8 / 尖峰 + ×1e3 通道 + ×1e4 massive 行 / 全 0 / 1e-30 / 1e4）；小形状上所有数据类型 × RHT 的 4 种组合 × 只行 / 只列 / fp32 输入 / C%128==64；e2m1 判决阈值上的值；比较 q、scale 字节、ts、prefix 的 `xp` | 721 | 0 |
| sr | 显式 u 与 `fp4_ref.e2m1_sr` 逐位一致（行 / 列 / 混合 RN+SR，prefix 开关）；Philox 已知答案向量 + kernel 流 = numpy 实现；kernel 内 Philox = 显式 `philox_uniform` 流（逐位）；无偏性（256 个种子，不饱和元素的均值偏差 / 期望噪声 = nv4 行 1.002 / 列 0.999，mx4 行 0.997 / 列 0.999）；同种子同比特（0 字节差）；`roll()` 后 q / qt 约 53% / 48% 的字节变化；换 `sr_offset` 也变 | 37 | 0 |
| exact | 设备端 ts 与 `amax/(6·448)` 的 fp32 除法逐位一致（`div_true=True`，600/600 随机 amax）；默认模式与 `fp4_ref` CUDA 逐位一致（600/600）；`div_true=True` 整个量化（q、scale、ts）与 `fp4_ref` 在 **CPU** 张量上逐位一致；prefix 时转置副本的 amax 只含 patch token（且与含全部 token 的 ts 不同） | 1820 | 0 |
| weight | 2D 权重量化对 `fp4_ref.quantize_weight(two_d=True)`：q / qt / sf / sft / ts 逐位（10 种形状 × nv4 / mx4 × 5 类数据，另有 bf16 权重；exact 模式 ts 与 q 也逐位） | 84 | 0 |
| state | `roll` 搬运 + 清零 + seed / step 计数；首次调用 exact 初始化；delayed ts = 2 × 上一步 amax；CUDA graph 捕获 roll + act + weight，重放 4 步 = eager 4 步（输出、amax、seed、step 全相同） | 14 | 0 |
| gemm | 量化器输出喂 QuACK nv4 GEMM（fwd、dgrad、wgrad 三种布局；fc1 形 7203×1024×4096、qkv 形 7903×1024×3072 带 prefix、head 形 7168×1024×4096），对反量化 fp32 乘积的相对 L2：≤ 5.7e-8（门槛 2e-3）；`fp4_epi.plain_ts` 设备标量 ts 的 epilogue 同样 | 3 | 0 |
| div | Markstein 除法与 `__fdiv_rn` 逐位一致：2.79×10^10 个样本 0 差异 | 1 | 0 |
| codes | 硬件 `cvt.rn.satfinite.e2m1x2` 对 `fp4_ref.e2m1_rn` 判决阈值在**全部 2^32 个 fp32 位型**上 0 差异；SR 码对 `e2m1_sr` 在 4×10^7 个输入上 0 差异 | 2 | 0 |

量化器输出对**真实 fp32 乘积**的整链误差（信息项，不是门）：fwd 0.146、dgrad 0.176（与 R20a 误差表的 nv4 单 GEMM 误差一致）。

### 4.2 微基准（`profile/bench_fp4q.json`；copy 带宽 1528 GB/s；µs，冷输入；括号为对 DRAM 下限的倍数）

| 形状 | nv4 行 | nv4 转置 | **nv4 dual** | dual+RHT(col) | dual+SR | dual+RHT+SR | prefix dual | prefix+RHT+SR | mx4 dual | nv4 dual exact | r17 mx8 dual |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ViT-L X [7203,1024] | 13.4 (1.08) | 12.9 (1.04) | **18.8 (1.25)** | 18.9 | 23.1 (1.53) | 23.7 (1.57) | 18.9 | 24.1 (1.59) | 13.9 | 26.0 (1.72) | 19.2 |
| ViT-L h [7203,4096] | 46.0 (0.93) | 46.4 (0.94) | **60.3 (1.00)** | 60.2 | 76.7 (1.27) | 79.6 (1.32) | 61.7 | 81.2 (1.34) | 57.0 | 77.4 (1.28) | 78.6 |
| ViT-L dY [7203,3072] | 34.0 | 34.6 | **47.6 (1.05)** | 47.8 | 59.1 (1.30) | 61.4 (1.35) | 48.6 | 62.2 (1.37) | 42.6 | 59.2 | 58.9 |
| ViT-L dY [7203,1024] | 13.5 | 12.8 | **18.8 (1.24)** | 18.8 | 23.1 | 23.7 (1.57) | 19.0 | 24.1 | 13.8 | 26.0 | 19.2 |
| ViT-L dY [7203,4096] | 45.9 | 46.4 | **60.4 (1.00)** | 60.2 | 76.8 | 79.9 (1.32) | 61.8 | 81.3 | 57.0 | 77.5 | 78.5 |
| head dZ [28672,4096] | 185.7 (0.94) | 190.2 (0.97) | **236.4 (0.98)** | 236.1 | 291.2 (1.21) | 303.7 (1.26) | – | – | 230.1 | 374.6 (1.56) | 312.6 |
| head dZ [7168,4096] | 45.8 | 46.1 | **59.5 (0.99)** | 59.8 | 76.6 | 79.7 (1.33) | – | – | 56.6 | 76.4 | 78.0 |
| convT in [28672,1024] | 45.8 | 45.9 | **59.3 (0.99)** | 59.5 | 76.6 | 79.6 | – | – | 56.6 | 75.1 | 77.9 |
| ViT-B X [14406,768] | 18.2 | 18.2 | **25.9 (1.14)** | 26.1 | 32.0 | 33.1 (1.46) | 26.5 | 33.6 | 20.8 | 33.8 | 29.1 |
| ViT-B h [14406,3072] | 69.8 | 70.4 | **89.8 (0.99)** | 89.5 | 113.5 | 118.0 (1.30) | 92.5 | 120.3 | 86.1 | 140.0 | 117.9 |
| ViT-B dY [14406,2304] | 51.9 | 52.1 | **67.6 (1.00)** | 67.7 | 86.3 | 89.6 (1.32) | 70.1 | 91.4 | 64.1 | 84.7 | 88.2 |
| ViT-B head dZ [57344,3072] | 278.4 | 284.2 | **353.7 (0.98)** | 353.5 | 437.5 (1.21) | 456.0 (1.27) | – | – | 344.4 | 560.7 | 469.2 |
| ViT-B head dZ [14336,3072] | 69.4 | 69.8 | **89.1 (0.99)** | 89.3 | 113.3 | 118.0 (1.31) | – | – | 85.4 | 139.8 | 117.0 |
| ViT-B convT in [57344,768] | 69.3 | 69.5 | **88.7 (0.98)** | 88.2 | 113.2 | 117.7 | – | – | 85.4 | 138.5 | 117.1 |

（所有其他形状 [7903 / 15806 系列] 见 `profile/bench_fp4q.json` / `results.json`；`x.t().contiguous()` 对照：[7203,1024] 71.8 µs、[7203,4096] 269 µs、dZ [28672,4096] 1050 µs，即 dual 的 3.8–4.4 倍。R20a 的 Spark 转置路径 97.7 / 414 / 1525 µs，QuTLASS `backward_t_bf16` 59.4 / 233 / 916 µs，相同形状。）

权重（fp32 读，双副本，冷）：nv4 delayed 11.7 µs（[3072,1024]，1.11×）/ 14.3 µs（[4096,1024]，1.01×）/ 5.95 µs（[1024,1024]，1.69×）；ViT-B 4.5–9.5 µs；mx4 约快 8%；exact（多一遍 amax）约 2 倍；r17 mx8 dual 对照 5.2–15.6 µs。

暖输入（同一输入，≤15 MB 留在 L2）：nv4 dual X[7203,1024] 17.4 µs（r17 mx8 dual 8.2 µs），ViT-B X[14406,768] 23.8 µs（r17 14.3 µs）。**小形状 L2 常驻时我们的 kernel 比 mx8 慢约 2 倍**（单 CTA 延迟 / 波次受限），大形状 DRAM 受限时则快 1.0–1.33×。

### 4.3 与目标的对照和瓶颈

- 目标"dual 输出 ≤ 约 1.3× DRAM 下限"：**纯 dual 和 dual + RHT 达到**（28 个形状全部 0.98–1.23×）；**dual + SR 大形状达到**（1.22–1.28×，RHT + SR 1.27–1.33×，个别 1.33 略超），**小形状没达到**（SR 1.29–1.53×，RHT + SR 1.34–1.58×）。
- SR 额外 +22–28%（大形状）：Philox + SR 码 + 16 位 uniform 转换的 ALU 发射（随元素数线性，而不是延迟）。
- 小形状：[7203,1024] 的 dual 18.2 µs 对下限 15.1 µs（行 / 列单独各约 13 µs），1808 个 CTA、每 SM 5 个常驻（96 寄存器）= 2.13 波次；mx4 dual（无除法）14.3 µs，说明 nv4 多出的约 4 µs 是每元素的位精确除法 + e4m3 scale 的指令数，不是访存。

### 4.3b 优化一轮（`profile/variants_summary.json`，快速子集，冷 µs，同一 session）

实验开关在 `impl/fp4q.py`：`R21_VARIANT=<名字> R21_DEFS="-D..."`（独立 build 目录，不影响默认 kernel）。

| 变体 | X[7203,1024] dual / RHT+SR | dY[7203,4096] dual / prefix+RHT+SR | head dZ[28672,4096] dual / RHT+SR | ViT-B X[14406,768] dual / RHT+SR | 结论 |
|---|---|---|---|---|---|
| 默认（基线） | 18.2 / 23.5 | 59.6 / 80.7 | 236.5 / 301.9 | 25.7 / 32.8 | – |
| `-DACT_MINB=6`（≤80 寄存器，每 SM 6 个 CTA；一个 kernel 有 8 B 栈 spill） | 18.5 / 23.5 | 59.8 / 80.8 | 236.4 / 301.3 | 25.9 / 32.8 | **无收益**（转置 only 反而慢 1 µs）：占用率不是瓶颈，没采用 |
| `-DPHILOX_ROUNDS=7` | 18.2 / 22.6 | 59.9 / 76.5 | 236.2 / 283.6 | 25.6 / 31.2 | SR 模式 −4…−6%（dZ 1.26× → 1.18× 下限）；**未采用为默认**：偏离标准 Philox4x32-10（已知答案向量测试针对 10 轮）、总收益小（约 0.3 ms/步） |
| 两者 | 18.4 / 22.7 | 60.1 / 77.2 | 236.4 / 286.4 | – | 同上 |

没有做的优化（预期收益小、预算不够）：SR 的整数比较（省约 4 条指令 / 元素）、nv4 block scale 的 `t/ts` 除法用常量倒数的 Markstein（省约 3%）、小形状按 32 token 切 tile。

### 4.4 每步量化开销估计（`profile/per_step_quant_cost.json`；冷输入 → 小形状上界）

按 R20a 的计数方式（ViT-L 21 plain M=7203 + 3 joint M=7903 块，ViT-B 9 + 3；head 每步 4 次 predict × ConvT1 / ConvT2）：

| ms/步 | ViT-L | ViT-B | 说明 |
|---|---:|---:|---|
| ① head d+w（nv4）：dZ dual（RHT col + SR）+ ConvT 输入 col | **1.77** | **2.65** | FINAL 的 head 是 bf16，无量化器，整笔是新增；RN（无 SR）则 1.42 / 2.12；R20a 估计 1.64 / 2.66 |
| ③ 骨干 d+w（nv4）：FP4 量化总时间（dY dual（SR + RHT col + prefix）+ X col（RHT + prefix）） | 6.74 | 4.95 | 其中 dY 部分 5.5 / 4.1 ms |
| ③ 相对 FINAL 的 mx8 量化净增（dY mx8 dual 被替换；X 保留 mx8 行给 MXFP8 前向，转置副本改成 FP4；qkv 前向是 bf16，X 转置副本为净增） | **+1.66** | **+1.09** | R20a 估计 +0.27 / +0.65；差别来自 SR + RHT + prefix 开销和 ViT-L 小形状 |
| 权重量化（nv4 delayed，24 / 12 块 × 4 个 Linear 的 W 与 Wᵀ） | 1.11 | 0.38 | FINAL 已经在量化 W 到 MXFP8，视为抵消（与 R20a 相同处理） |

对 R20a 的 ①、③ 节省（GEMM 节省 ① 8.88 / 9.69 ms，③ 8.13 / 4.93 ms）：① 扣除后 7.1 / 7.0 ms；③ 扣除量化净增后约 6.5 / 3.8 ms（再扣 prefix 侧 GEMM 1.5–1.8 / 0.6–1.0 ms，那是 R22 的事）。

## 5. 正确性

- Gate A 见 §4.1，全部通过，没有放宽任何一项。
- **ts 的位级定义不一致（契约请求，见 §7）**：`fp4_ref._nv4_ts` 里的 `amax / (6.0 * 448.0)` 在 CUDA 张量上是 torch 的"标量除法 = 乘以 fl32(1/2688)"（实测 400 万个随机 amax 中 89.9 万个与真正的 fp32 除法差 1 ulp，与乘倒数 0 差异；`amax_blk/6` 同理 133 万个差 1 ulp），在 CPU 张量上才是 IEEE 除法。CONTRACTS 12.2 要求"fp32 除法"。kernel 提供 `div_true`：False（默认）= 与 `fp4_ref` CUDA 逐位一致；True = 字面的 fp32 除法 = `fp4_ref` CPU 逐位一致。两者的整个量化输出都做了逐位对拍。差别只是极少数块的 e4m3 scale 舍入 / ts 差 1 ulp。
- 没有改任何参数 / state_dict / 训练状态，所以没有 export / resume 检查。B 门、S 门、graph_equiv（整步）不在本任务内。

## 6. 静态分析与 roofline

`common/kernel_static.sh` 在服务器上不能直接用（脚本依赖 `rg`，服务器没有；`cu` 模式也不能带 torch 头），所以用构建日志里的 `ptxas -v` 和 `cuobjdump -res-usage`（`profile/static_summary.json`、`profile/static/*.sass`）：

| kernel（`cuobjdump -res-usage` 最终 .so；bf16 / fp32 输入相同） | 寄存器 | spill / stack | smem（静态） |
|---|---:|---:|---:|
| act nv4 行 + 转置（dual） | 95 | 0 | 17 408 B |
| act nv4 仅行 | 81–82 | 0 | 0 |
| act nv4 仅转置 | 96 | 0 | 17 408 B |
| act mx4 行 + 转置 / 仅转置 | 136 | 0 | 17 408 B |
| act mx4 仅行 | 81–82 | 0 | 0 |
| 权重 w_kernel（nv4 / mx4，fp32 / bf16） | 58 / 58 / 47 / 47 | 0 | 20 496 B |
| amax_abs / amax_roll / amax_roll_mon / philox_uniform | 18–22 / 12 / 24 / 22 | 0 | 0 |

（早先在 00:28 的构建日志里 nv4 dual 是 96、mx4 是 99–102；后来加了 Philox 16 位半字和 `__launch_bounds__(128, ACT_MINB=1)` 之后重新编译，mx4 的 dual / 转置变成 136 寄存器（占用 3 CTA / SM），nv4 基本不变；上面是最终 .so 的数。mx4 不是推荐格式。）occupancy（128 线程）：nv4 dual 95 寄存器 → 每 SM 5 个 CTA；smem 不是限制。SASS 计数（早先构建，全部 kernel 合计）：LDG 293、STG 223、LDS 514、STS 116、SHFL 564、F2FP 580、BAR 20、ATOMG 1。

roofline：大形状下 dual 与 DRAM 下限重合（0.98–0.99×），没有余量可挖；SR 路径 ALU 受限（见 §4.3）。

## 7. 结论、推荐、失败项、契约变更请求、下一瓶颈

**结论**：融合 dual 量化器按要求完成，Gate A 全部通过；大形状 dual 在 DRAM 下限上（0.98–0.99×），比 r17 mx8 dual 快 1.03–1.33×；RHT 免费，SR +22–28%；每步量化开销 ① 1.77 / 2.65 ms，③ 相对 FINAL 净增 1.66 / 1.09 ms。

**失败项 / 没做到的**
- dual + SR 的小形状未达到 1.3×（1.29–1.53×；RHT + SR 1.34–1.58×），大形状 1.22–1.28×（RHT + SR 1.27–1.33×，见 §4.3）。优化一轮（§4.3b）：min-blocks 无收益；7 轮 Philox 有 4–6% 但未采用。
- 暖（L2 常驻）输入下小形状 dual 比 r17 mx8 dual 慢约 2 倍（17.4 vs 8.2 µs）。
- `kernel_static.sh` 在服务器上不能用（缺 `rg`），改用 ptxas / cuobjdump。
- 没有 mx8 行副本的同步输出（PLAN 里"head 前向 mx8、反向 nv4"时的需求）：行副本 mx8 可以直接用 r17 `mx8.quantize(row=True, col=False)`（暖 / 冷都在下限），R21b 不需要新 kernel。

**给 R21b / R22 的接入注意事项**
1. 每步开头调用一次 `state.roll()`（设备端，graph 内）；每个调用点（site）一个固定编号和一个固定 `sr_offset`；`delayed` 首次调用需要先 eager 跑一遍（`state.inited[site]`），不能在 graph 捕获里第一次调用。
2. 权重：`quant_weight(master_fp32, "nv4", amax="delayed", state, site)` 同时给 W / Wᵀ；GEMM 的 alpha = `ts_a · ts_b`，用 `fp4_epi.plain_ts` 等 epilogue 传设备标量（`q.ts * w.ts`）；dgrad 的 B 操作数用 `w.qt / w.sft`，ts 同一个 `w.ts`。
3. wgrad 的收缩维 Kt 是 `q.kt`（plain 时 Rpad，prefix 时 B·1024）；两个操作数（dYᵀ 与 Xᵀ）必须用相同的 prefix 和相同的 RHT 符号（符号固定，种子 20251003，常量 `RHT_SIGN_MASK` 已断言等于 `fp4_ref.rht_signs()`）。prefix 侧 GEMM 用 `q.xp`（X 的 prefix 行）与 dY 的 `xp`（对 dY 同样调用 `prefix=` 即可得到）。
4. `delayed` 的 ts 是上一步 amax 的 2 倍：若某个 site 的激活 amax 突增 >2 倍，e4m3 scale 会饱和到 448（kernel 里 `min(…, 448)`），那一步的该块有量化误差；S 门（50 步）才能验证，B 门（单步）与 exact 等价。
5. C 必须是 64 的倍数；prefix 模式要求 `(N-npre) % 64 == 0`、`B·(N-npre) % 128 == 0`。
6. 输出缓冲由 kernel 内部分配（torch 缓存分配器，graph 内走私有池），与 r17 `mx8.quantize` 的使用方式相同。

**契约变更请求**
- CONTRACTS 12.2 / `fp4_ref`：明确 ts 与 `amax_blk/6` 是"IEEE fp32 除法"还是"torch CUDA 标量除法（乘倒数）"。当前 `fp4_ref` 在 CUDA 上是后者；建议要么把 `fp4_ref` 里这两处改成对 0 维 tensor 的除法（`amax / torch.tensor(2688.0, device=...)`），要么把契约改成"与 `fp4_ref` CUDA 逐位一致"。kernel 已两者兼容（`div_true`），默认 False = fp4_ref CUDA。
- 无其它契约变更。

**下一瓶颈**：(a) 小形状 L2 常驻时的 CTA 延迟 / 波次（暖 dual 17 µs vs mx8 8 µs）；(b) SR 的 ALU 开销（7 轮 Philox 可省 4–6%）；(c) 每步量化总量（③ 6.7 / 5.0 ms），减少量化次数（例如 qkv 与 fc1 共享 LN 输出的转置副本）比继续优化单个 kernel 更值。

## 8. 后续任务：`headroom` 参数 + `roll()` 饱和监控（Amendment 6b 13.5；`READY_headroom` 已创建）

**新接口**（没有改任何现有接口；新参数都有默认值，现有调用不变）

```python
fp4q.quant_act(x, "nv4", ..., amax="delayed", state=st, site=i, headroom=2.0)    # delayed: ts = headroom * prev_amax / 2688
fp4q.quant_weight(w, "nv4", amax="delayed", state=st, site=i, headroom=2.0)
st.roll()          # 原语义不变 (prev<-next, next<-0, seed+=1, step+=1) + 饱和监控；设备端，无 host 同步，可进 CUDA graph
st.stats()         # host 读取（不要在 graph 内调用）
st.set_headroom(site, h); st.reset_stats(); st.reset()   # reset() 现在也清监控计数
```

- `headroom` 只在 delayed 模式生效，默认 2.0；调用时把该 site 的 headroom 记进 `st.headroom`（设备端 fp32 [n_sites]；值不变时不做设备写，所以默认调用没有任何额外 kernel）。同一个 site 的两个槽（行副本 / 转置副本）共用一个 headroom。headroom 必须 > 0；`headroom=8` 的 ts = `fl(fl(8·prev) · fl(1/2688))`（`div_true=True` 时为 IEEE 除法）。
- **监控语义**（`amax_roll_mon_kernel`，每个 slot 一个线程，不进量化 kernel）：roll 时 `prev` 是给上一步 ts 用的 amax，`next` 是上一步实际产生的 amax；若 `next > prev · headroom`，上一步的 nv4 block scale 在 e4m3 的 448 处饱和。每个 slot 累计 `calls`（比较次数）、`exceed`（超限次数，严格大于，所以恰好 2 倍在 headroom=2 时不算超限）、`max_ratio`（`next/prev` 的最大值）。`prev == 0 且 next > 0`（上一步 amax 为 0）计为超限，`max_ratio = inf`；两者都是 0（从未调用的 site）跳过。
- `st.stats()` 返回 dict：`n_sites, step, headroom[sites]`，`row` / `col` 各有 `calls / exceed / max_ratio`（列表，按 site；权重 site 只用 `row`），以及 `total_calls, total_exceed, exceed_rate`。**第一次 roll 之后的比较才有意义**：第一次调用的 site 用 exact 初始化 prev，所以紧接着的第一次比较比值恒为 1。
- 用法建议（13.5）：dY 的调用点 `headroom=8`，X 和权重保持默认；每隔若干步在 host 上 `st.stats()`（图外）看 `exceed_rate` 和 `max_ratio / headroom`。

**测试**（`tests/test_fp4q.py --part headroom`，写入 `profile/gateA_fp4q.json`，现在共 9 个部分，`all_pass = true`）

| 项 | 结果 |
|---|---|
| 默认值与改动前逐位相同 | 改动前的 build 上用 `tests/headroom_baseline.py --save` 存了 72 个输出的 sha256（nv4 / mx4 × 3 个形状 × dual / RHT+SR / prefix+RHT+SR × 3 步 delayed、amax 状态、权重 3 步）；改动后默认调用和显式 `headroom=2.0` 都是 **72/72 一致**；完整 Gate A 其余 8 个部分重跑全部 0 失败 |
| headroom = 8 / 3 / 0.5 / 2 的 ts 位 | act（行 + 转置）和 weight、`div_true` 两种模式：delayed 的 ts 与 `fl(fl(headroom·prev)·fl(1/2688))`（或 IEEE 除法）逐位相同；delayed 调用的 q / qt / sf / sft 与同一 ts 的显式 ts 模式逐位相同（后者对 fp4_ref 的逐位一致已由原 Gate A 保证）|
| 监控计数 | 人为构造的 amax 序列（比值 1、4、0.25、16、1、0.03、2、2（恰好等于 headroom 2 的情形）、0（全零步）、inf（prev=0）、1、64…），site0 headroom 2、site1 headroom 8、site2 权重、site3 从未调用：计数（calls 14 / exceed 4 / 3 / 4）和与独立的 host 模型完全一致，`max_ratio` 一致，未调用的 site 和权重的 col 槽保持 0；另一条无全零步的序列给出有限的 `max_ratio = 64.0`（headroom 2：exceed 3，headroom 8：exceed 2，与手算一致）；`reset_stats` 清零 |
| graph 回放 = eager | roll + act（SR + RHT col，headroom 8）+ weight 捕获进 CUDA graph，输入经静态缓冲在回放之间变化（缩放 1、12、0.5、1、40、1），回放 6 次 = eager 6 步：输出、amax / seed / step、监控计数 / `max_ratio` / headroom 数组全部逐位相同；该序列触发了 4 次 exceed（graph 与 eager 相同） |

**性能影响**（`profile/bench_headroom.json`，冷输入，CUDA graph 回放，同一进程交替，3 轮中位数）

| 项 | 结果 |
|---|---|
| `headroom=8.0` / `2.0` 显式 / 默认调用，5 个形状 × dual / dual+SR / dual+RHT+SR | 比值 0.95–1.006（大部分 ±0.5%；dZ[28672,4096] 的 SR 行里默认调用那一次偏慢 5%，是同一进程内的噪声，同形状的其它模式 ±0.1%）。量化 kernel 本身没有改动，headroom 只是 kernel 标量参数 |
| 改动后的完整 `bench_fp4q`（28 个形状）对改动前的完整 bench（`profile/bench_fp4q.json`，较早的 session） | nv4 dual 中位 0.998（0.9945–1.001）；dual+RHT 0.998；dual+SR 0.990；dual+RHT+SR 0.991；对照 r17 mx8 dual 1.0006 → 差别在噪声内（SR 模式快 1% 是 session 漂移，不是改动带来的） |
| `roll()` | 旧 `amax_roll` 与新 `amax_roll_mon`，64 / 300 / 1000 个 site：都是 1.13 µs / 次（graph 内，launch 受限），监控没有可测的额外开销；每步 roll 一次，ViT-L 约 300 个 site 约 1.1 µs |
| 静态 | 量化 kernel 的寄存器 / smem 不变（见 §6）；`amax_roll_mon_kernel` 24 寄存器 |

---

<!-- R21b-BEGIN -->
# Round21（R21b）：head ConvT 接入真 FP4 kernel（H3 = `head.f=mx8, head.dw=nv4+sr`），负责：Sonnet，第三阶段

> 全部数字在 `infer-5090`（RTX 5090）上实测，持双锁；整步只在 round16 `StepGraphRunner` 下测，与 FINAL（`common.config.BEST_STACK_V2`）同场次对比。没测的写 null / not measured。**本章只覆盖 R21b**，R21a 的内容（§0–§8）没有改动。
> 没改 round0–20、common、PLAN、CONTRACTS；只 import 了 r14 `head` / `head_cuda`、r17 `mx8` / `gemm_mx8`、r20 `fp4_policy` / `fp4_cands` / `fp4_epi` / `fp4_ref` / `fp4_sim` / `fp4_head`（仿真 head 参照）和 r19 / r16 的 `run.py`、`run_final.py`、`check_export.py`。
> 中途因会话额度用完中断过一次（不是任务出错），之后只补缺的项；ViT-L 的整步 bench 与 B 门前 4 批是中断前跑完的，其余是之后跑的（时间戳见各 json）。

## 0. 摘要

| 项 | 结果 |
|---|---|
| 交付 | 真 kernel 的 head ConvT：`impl/head_fp4.py`、`impl/patch.py`（patch 名 `round21_fp4_quant_head`，覆盖 `ops.predict`，旧函数保留作回退）、`run.py`（预设 `P3H` / `P3H_H1` / `P3H_H2` 等）、Gate A / B 脚本、排队 / bench / nsys / 导出驱动 |
| 策略 | H3 = `head.f=mx8, head.dw=nv4+sr`（前向 r17 MXFP8，dgrad / wgrad NVFP4 + SR，dZ 的 `headroom=8`）；对照 H1 = `head.dw=nv4+sr`（前向 bf16），H2 = `head.dw=mx8`（r14 的 MXFP8 反向） |
| Gate A | **全部通过**：RN 变体对 R20b 仿真 head 的 logits 和 12 个梯度**逐位相同**（余弦 1.0，门槛 0.9999）；GEMM 对反量化乘积 fp32 输出 ≤ 7.5e-8；SR 统计与仿真生成器一致（误差比 0.999–1.001）；capture 进 CUDA graph 与 eager 逐位相同 |
| Gate B（n = 8，同进程对 R0 和 P1） | H3：ViT-L 中位 0.9948 / 最小 0.9884，ViT-B 0.9923 / 0.9871；**两个模型都过放宽门，也过严格门**（中位 ≥ 0.99、最小 ≥ 0.98，对 R0 和 P1）；H1、H2 同样都过 |
| Gate S（4 场景 × 50 步 × 2 批，StepGraphRunner） | **8 个场景全过**：对 R0 的 EMA 最大偏差（检查点）ViT-L 0.61%、ViT-B 0.75%（门 5%；所有步最大 0.82% / 0.75%），无 NaN / Inf，grad-norm 比 0.89–1.14；`state.stats()`：dZ 的 delayed amax 在 `headroom=8` 下 **0 次饱和**（最大 amax 比 2.9），X 与权重 0 次 |
| graph_equiv | 过：第一步 loss 逐位相同，8 步 loss 相对差 ≤ 2.9e-4（FINAL 在 r19 是 ≤ 1.7e-4 同量级），更新方向余弦 ≥ 0.99986 |
| 导出 / resume | ViT-L、ViT-B 的 `check_export.py` 都 **pass**：P3H 栈在 graph 下训练 20 步 + 验证 + Lightly 自己的 save / export，checkpoint 与导出的键、形状、dtype 与 R0 和原训练导出一致，optimizer / scheduler 结构一致，权重已更新，原版 `load_model` 加载并 predict，原版路径 resume 一步 loss 有限 |
| 整步速度（graph，三相加权，同场次） | ViT-L 69.34 → **77.59 img/s（×1.119，对 R0 约 4.72×）**，mid 101.04 → 90.31 ms；ViT-B 160.30 → **183.82 img/s（×1.147，对 R0 约 4.92×）**，mid 87.51 → 76.33 ms；H1 ×1.076 / ×1.101，H2 ×1.048 / ×1.059；**远超 2% 的发布线** |
| 与 R20a 估计的对照 | 估计 head 净省 L 约 7.1 + 3.3 = 10.4 ms、B 约 7.0 + 3.05 = 10.05 ms；实测 mid 步时 L 省 10.73 ms、B 省 11.18 ms（H1 单独：7.08 / 8.04 ms），nsys kernel 总时间 L −10.41 ms、B −10.36 ms，**与估计一致（实测比估计多省：L +3%、B +11%）** |
| 没做到 / 没测（not measured） | P3H 与 R22 骨干 patch 的组合栈（R23）；e2e 200 步（R23）；ViT-L 的 bench 最后一次 FINAL（中断前的队列里有，中断后没补，FINAL 已有 2 次，漂移 0.04%）；dZ 量化并进 `bwd2` 只做了评估（≤ 约 1 ms / 步，≈1.1%，低于 2% 线，不建议做） |

## 1. 目标与假设

| 假设（来自 PLAN_v3 / 6a 12.4 / R20a / R21a） | 实测 |
|---|---|
| head 的 dgrad+wgrad 用 nv4 GEMM 比 bf16 快约 5×（R20a：ConvT 每步净省 ① 7.1 / 7.0 ms，加前向 mx8 ② 3.3 / 3.05 ms） | dgrad / wgrad 的 nv4 GEMM 对 bf16（r14）快 4.5–5.9×（`bench_head_gemm.json`，见 §4.1 末尾）；nsys 中 head 的 bf16 GEMM 类别 L 18.97 → 3.09 ms、B 21.58 → 4.21 ms（剩下的是 einsum、mask_head 等），新增 NVFP4 GEMM L 2.07 / B 2.18 ms，MXFP8 前向 GEMM L +1.78 / B +2.05 ms |
| dZ 一次读入同时出 q 与 qt（R21a 量化器，dual + SR，headroom=8）；权重每步量化一次 | 实现，见 §2；dgrad / wgrad GEMM 对反量化乘积 fp32 输出误差 5e-8（bf16 输出受 bf16 舍入限制 1.7e-3） |
| RN 变体与 R20b 仿真 head 一致（6b 13.4：余弦 ≥ 0.9999） | **逐位相同**（logits 与全部 12 个输入 / 参数梯度的 relL2 = 0），见 §4.1 |
| H3 过放宽 B 门 / S 门（R20b 仿真：ViT-L / ViT-B 都过，ViT-B 过严格门） | 两个模型 n = 8：H3 对 R0 与 P1 的全局梯度余弦中位 / 最小为 L 0.9948 / 0.9884、B 0.9923 / 0.9871，loss_rel ≤ 2.6e-3，**放宽门与严格门都通过**；S 门 8 个场景全过（≤ 0.75%） |

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

1. 用 r16 的 `nsys_inner.sh`（`--cuda-graph-trace=node`，capture range = `cudaProfilerApi`）在 mid 相各抓 5 个 graph 步：ViT-L / ViT-B × FINAL / P3H（`queue_nsys.sh`，原始 `.nsys-rep` 在 `/root/autodl-tmp/train-opt-tmp/round21_fp4_quant_head/nsys/`，kernel 汇总 csv 和 trace csv 同目录）。
2. 分类用 r19 `nsys_final_report.py` 的分类器，前面加三类（`tests/nsys_report21.py`）：`act_kernel`（R21a 融合量化器）、`w_kernel` + `amax_*`（权重量化与 roll）、QuACK NVFP4 GEMM（kernel 名里的 MMA atom 为 16864）。
3. 看三件事：（a）head 的 bf16 GEMM 类别（QuACK 前向 / dgrad + cutlass sm80 wgrad）是否真的被替掉；（b）新增类别各占多少：FP4 量化 L 1.69 / B 2.60 ms，NVFP4 GEMM L 2.07 / B 2.18 ms，权重量化 + roll 0.01–0.02 ms；（c）窗口和忙碌率：FINAL 的窗口 L 100.51 / B 86.37 ms，P3H 89.74 / 76.06 ms，忙碌率 0.978–0.984 不变（仍然是纯 kernel 时间受限，没有引入 host 间隙；`roll()` 每步 1 个 24 寄存器的小 kernel）。
4. 判断：head 量化类 kernel 占整步 L 1.9% / B 3.5%；ConvT 的 nv4 GEMM 在 1.0–1.25 PFLOPS（`bench_head_gemm.json`），配置已最优；剩下的 head 时间是 r14 的融合 elementwise kernel（`fwd` / `bwd1` / `bwd2`，L 6.7 / B 10.4 ms，内存受限）和 bf16 的 einsum / mask_head（L 3.1 / B 4.2 ms）。

## 4. 实测结果

### 4.1 Gate A（`profile/gateA_head_fp4.json`，`tests/test_head_fp4.py --part all`，0 失败）

| 部分 | 内容 | 结果 |
|---|---|---|
| gemm | head 的 nv4 GEMM（`plain_ts` 后端与 BlockScaledOperand 两种；dgrad / wgrad / fwd；两个模型的 ConvT1 / ConvT2 真实形状 M = 7168 / 28672（ViT-L）、14336 / 57344（ViT-B）；gauss 与 scale-8 尖峰数据）对反量化 fp32 乘积 | 16 例，**0 失败**：wgrad（fp32 输出）相对 L2 最大 7.5e-08；fwd / dgrad（bf16 输出）1.66e-03（= bf16 输出舍入的下限 2^-9/√3，门槛 2e-3） |
| head RN | RN 变体（`head.f=mx8,head.dw=nv4+rn,amax=exact`、`head.dw=nv4+rn,amax=exact`、`head.dw=mx8`）对 R20b 仿真 head（`fp4_head`）：logits 和 12 个输入 / 参数梯度（tok、e、w1、b1、d1、g1、be1、w2、b2、d2、g2、be2） | 12 组（2 模型 × gauss / spike × 3 策略）：**logits relL2 最大 0.0e+00，梯度 relL2 最大 0.0e+00，梯度余弦最小 1.0000000（逐位相同）**，门槛 0.9999 |
| bf16 恒等 | 空策略 = r14 head | 4 组全部逐位相同：True |
| SR 统计 | 6 个种子；对 bf16 head 的误差（w1 / w2 / tok 梯度）与种子间标准差，kernel Philox 对仿真生成器 | 误差比 0.9994–1.0005，标准差比 0.9995–1.0014（门：0.9–1.1）；同种子逐位相同 True，换种子结果不同 True；6 个种子平均后误差明显下降（w1：0.247 → 0.174），SR 无偏 |
| graph | 4 次 predict（SR + delayed + roll）捕获进 CUDA graph：预热（eager，`StepGraphRunner.warm` 同款）→ 复位 `next` 槽 / seed → 捕获 → 重放两次，对 eager 的第 1、2 步 | vitl16: 第 1 步逐位 True、第 2 步逐位 True；vitb16: 第 1 步逐位 True、第 2 步逐位 True |

#### head ConvT 的 nv4 GEMM 配置（`profile/bench_head_gemm.json`，graph 回放，交替，3 轮中位数，µs；括号 TFLOPS）

| 形状 | dgrad `plain_ts` | dgrad operand | dgrad cuBLASLt | dgrad bf16（r14） | wgrad `plain_ts` | wgrad operand | wgrad operand split-K | wgrad cuBLASLt | wgrad bf16（r14） |
|---|---|---|---|---|---|---|---|---|---|
| vitl16/convT1 M=7168 | 48.0 (1253) | 49.3 (1221) | 52.5 (1144) | 283.5 (212) | 54.1 (1112) | 55.8 (1077) | 55.1 (1091) | 57.2 (1051) | 298.8 (201) |
| vitl16/convT2 M=28672 | 225.2 (1068) | 230.5 (1044) | 236.6 (1017) | 1066.5 (226) | 231.1 (1041) | 232.8 (1033) | 229.4 (1049) | 228.8 (1051) | 1131.4 (213) |
| vitb16/convT1 M=14336 | 54.0 (1253) | 55.3 (1224) | 59.4 (1139) | 287.6 (235) | 58.6 (1155) | 57.3 (1180) | 62.6 (1080) | 59.5 (1136) | 320.0 (211) |
| vitb16/convT2 M=57344 | 258.7 (1046) | 256.2 (1056) | 273.1 (991) | 1177.9 (230) | 260.4 (1039) | 260.3 (1040) | 270.8 (999) | 269.7 (1003) | 1286.5 (210) |

结论：QuACK `plain_ts`（设备标量 ts 的 epilogue）在 8 个 dgrad / wgrad 里 5 个最快或并列最快，其余与最快差 1.0–2.2%；split-K 对 head 的 wgrad 没有收益（ViT-B ConvT1 慢 7%，ViT-L ConvT1 持平）；cuBLASLt 的 dgrad 慢 5–10%、wgrad 差 −1%…+4%；bf16 → nv4 的 GEMM 加速 4.5–5.9×（与 R20a 的 5.1× 一致）。**配置已是最优，不需要再调**（后端默认 `epi`）。

### 4.2 Gate B（`profile/gateB_r21b_vitl16.json` / `vitb16.json`，各 8 批，同进程 R0、P1、候选，同权重 / 批 / RngTape）

候选 delayed amax 在单步里 = 第一次调用的 exact amax × headroom（X、W ×2，dZ ×8），所以这是"稳态 delayed + headroom"的真实量化条件。`H3x` = amax exact，`H3rn` = RN；三者数值相同到 4 位。每个进程先做一遍丢弃式预热（QuACK autotune / JIT 之后第一个 FINAL 的梯度与后面的不同，见 §5）。

#### vitl16: B gate (relaxed: median >= 0.98, min >= 0.97, loss_rel <= 2e-2; strict recorded), same process, same weights / batch / RngTape

| cand | policy | n | vs R0 median / min | vs R0 loss_rel max | vs P1 median / min | relaxed (R0 & P1) | strict (R0 & P1) | vs FINAL median | head param cos min |
|---|---|---:|---|---:|---|---|---|---:|---:|
| FINAL | `` | 8 | 0.9976 / 0.9926 | 2.6e-03 | 0.9974 / 0.9926 | True | True | 1.0000 | 0.991 |
| H3 | `head.f=mx8,head.dw=nv4+sr` | 8 | 0.9948 / 0.9884 | 2.4e-03 | 0.9937 / 0.9875 | True | True | 0.9959 | 0.979 |
| H1 | `head.dw=nv4+sr` | 8 | 0.9961 / 0.9911 | 2.6e-03 | 0.9958 / 0.9911 | True | True | 0.9984 | 0.986 |
| H2 | `head.dw=mx8` | 8 | 0.9976 / 0.9926 | 2.6e-03 | 0.9973 / 0.9925 | True | True | 0.9999 | 0.991 |
| H3x | `head.f=mx8,head.dw=nv4+sr,amax=exact` | 8 | 0.9948 / 0.9884 | 2.4e-03 | 0.9937 / 0.9875 | True | True | 0.9959 | 0.979 |
| H3rn | `head.f=mx8,head.dw=nv4+rn` | 8 | 0.9948 / 0.9885 | 2.4e-03 | 0.9937 / 0.9875 | True | True | 0.9959 | 0.979 |
| FINAL_again | `` | 8 | 0.9976 / 0.9926 | 2.6e-03 | 0.9974 / 0.9926 | True | True | 1.0000 | 0.991 |

#### vitb16: B gate (relaxed: median >= 0.98, min >= 0.97, loss_rel <= 2e-2; strict recorded), same process, same weights / batch / RngTape

| cand | policy | n | vs R0 median / min | vs R0 loss_rel max | vs P1 median / min | relaxed (R0 & P1) | strict (R0 & P1) | vs FINAL median | head param cos min |
|---|---|---:|---|---:|---|---|---|---:|---:|
| FINAL | `` | 8 | 0.9945 / 0.9932 | 2.4e-03 | 0.9945 / 0.9938 | True | True | 1.0000 | 0.986 |
| H3 | `head.f=mx8,head.dw=nv4+sr` | 8 | 0.9923 / 0.9871 | 2.6e-03 | 0.9909 / 0.9870 | True | True | 0.9960 | 0.974 |
| H1 | `head.dw=nv4+sr` | 8 | 0.9932 / 0.9919 | 2.4e-03 | 0.9931 / 0.9925 | True | True | 0.9986 | 0.981 |
| H2 | `head.dw=mx8` | 8 | 0.9944 / 0.9931 | 2.4e-03 | 0.9943 / 0.9937 | True | True | 0.9999 | 0.986 |
| H3x | `head.f=mx8,head.dw=nv4+sr,amax=exact` | 8 | 0.9923 / 0.9870 | 2.6e-03 | 0.9909 / 0.9870 | True | True | 0.9960 | 0.974 |
| H3rn | `head.f=mx8,head.dw=nv4+rn` | 8 | 0.9923 / 0.9870 | 2.6e-03 | 0.9909 / 0.9870 | True | True | 0.9961 | 0.974 |
| FINAL_again | `` | 8 | 0.9945 / 0.9932 | 2.4e-03 | 0.9945 / 0.9938 | True | True | 1.0000 | 0.986 |


### 4.3 Gate S（4 场景 × 50 步 × 2 批，StepGraphRunner 下的 P3H 对 R0 eager）

#### vitl16: S gate (50 steps x 2 batches, candidate P3H under StepGraphRunner vs R0 eager; EMA(0.9) deviation <= 5%, no NaN)

| scenario | max_rel (ckpts) | pass | finite | grad-norm ratio min / max | loss first (R0 / P3H) | loss last (R0 / P3H) | captures | dY exceed (calls) / max ratio | X exceed | W exceed |
|---|---:|---|---|---|---|---|---:|---|---|---|
| schedule | 0.0059 | True | True | 0.999 / 1.013 | 30.660 / 30.697 | 19.357 / 19.248 | 1 | row 0/200 col 0/200 max ratio 1.85 | 0/200 (max 1.19) | 0/50 |
| fixed_mid | 0.0061 | True | True | 0.977 / 1.010 | 30.531 / 30.560 | 19.052 / 18.971 | 1 | row 0/200 col 0/200 max ratio 2.71 | 0/200 (max 1.17) | 0/50 |
| compressed | 0.0019 | True | True | 0.988 / 1.009 | 30.393 / 30.377 | 16.128 / 16.125 | 7 | row 0/200 col 0/200 max ratio 2.91 | 0/200 (max 1.11) | 0/50 |
| fixed_late | 0.0010 | True | True | 0.999 / 1.017 | 30.287 / 30.305 | 16.113 / 16.139 | 1 | row 0/200 col 0/200 max ratio 2.13 | 0/200 (max 1.12) | 0/50 |

#### vitb16: S gate (50 steps x 2 batches, candidate P3H under StepGraphRunner vs R0 eager; EMA(0.9) deviation <= 5%, no NaN)

| scenario | max_rel (ckpts) | pass | finite | grad-norm ratio min / max | loss first (R0 / P3H) | loss last (R0 / P3H) | captures | dY exceed (calls) / max ratio | X exceed | W exceed |
|---|---:|---|---|---|---|---|---:|---|---|---|
| schedule | 0.0033 | True | True | 0.991 / 1.045 | 31.526 / 31.483 | 16.429 / 16.470 | 1 | row 0/200 col 0/200 max ratio 2.02 | 0/200 (max 1.12) | 0/50 |
| fixed_mid | 0.0049 | True | True | 0.992 / 1.141 | 31.305 / 31.333 | 16.040 / 16.258 | 1 | row 0/200 col 0/200 max ratio 2.19 | 0/200 (max 1.09) | 0/50 |
| compressed | 0.0075 | True | True | 1.013 / 1.061 | 30.188 / 30.181 | 16.664 / 16.949 | 7 | row 0/200 col 0/200 max ratio 1.95 | 0/200 (max 1.14) | 0/50 |
| fixed_late | 0.0058 | True | True | 0.892 / 1.023 | 29.957 / 29.988 | 16.701 / 16.729 | 1 | row 0/200 col 0/200 max ratio 2.50 | 0/200 (max 1.12) | 0/50 |


### 4.4 graph_equiv

#### vitl16: graph_equiv (8 steps; eager vs eager = run-to-run noise, graph vs eager)

| scenario | loss rel max graph-vs-eager | loss rel max eager-vs-eager | gnorm rel max g-v-e | update cos g-v-e | update rel L2 g-v-e | update rel L2 e-v-e | first-step loss bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed_mid | 7.99e-05 | 1.89e-05 | 1.98e-04 | 0.999987 | 5.01e-03 | 2.02e-03 | True |
| schedule | 1.06e-04 | 3.51e-04 | 6.81e-04 | 0.999974 | 7.16e-03 | 1.05e-02 | True |
| fixed_late | 4.16e-05 | 1.67e-05 | 5.03e-04 | 0.999990 | 4.48e-03 | 1.26e-03 | True |

#### vitb16: graph_equiv (8 steps; eager vs eager = run-to-run noise, graph vs eager)

| scenario | loss rel max graph-vs-eager | loss rel max eager-vs-eager | gnorm rel max g-v-e | update cos g-v-e | update rel L2 g-v-e | update rel L2 e-v-e | first-step loss bitwise |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed_mid | 3.34e-05 | 8.35e-05 | 2.21e-04 | 0.999991 | 4.24e-03 | 3.74e-03 | True |
| schedule | 2.91e-04 | 1.47e-04 | 2.27e-03 | 0.999858 | 1.69e-02 | 3.35e-03 | True |
| fixed_late | 1.03e-04 | 4.23e-05 | 2.97e-04 | 0.999989 | 4.60e-03 | 1.31e-03 | True |


### 4.5 导出 / resume（r19 `check_export.py`；checkpoint 由 P3H 栈在 graph 下跑 20 步 + 验证 + Lightly 自己的 `save_checkpoint` / `export_model` 产生）

- **vitl16**：`pass` = **True**；checkpoint 步数 20；checkpoint / export 的 train_model 键、形状、dtype 与 R0 一致 = True / True，与原训练导出权重一致 = True；optimizer 结构一致 True，scheduler 键一致 True；权重确实更新（改变的参数 19 个，最大改动 2.71e-05）；导出全部有限 True；原版 `lightly_train.load_model` 加载并 predict 成功（[1008, 1008]）；原版路径 resume 一步 loss = 29.0639（有限 True）。训练侧：graph 捕获 1 张，真实 loader 20 步，验证 + 保存成功
- **vitb16**：`pass` = **True**；checkpoint 步数 20；checkpoint / export 的 train_model 键、形状、dtype 与 R0 一致 = True / True，与原训练导出权重一致 = True；optimizer 结构一致 True，scheduler 键一致 True；权重确实更新（改变的参数 19 个，最大改动 3.84e-05）；导出全部有限 True；原版 `lightly_train.load_model` 加载并 predict 成功（[1008, 1008]）；原版路径 resume 一步 loss = 25.0664（有限 True）。训练侧：graph 捕获 1 张，真实 loader 20 步，验证 + 保存成功

### 4.6 整步速度（`profile/bench_*_all.json`，graph 回放，三相 0.2 / 0.6 / 0.2，warmup 20，50 × 3）

#### vitl16: whole-step bench under StepGraphRunner (warmup 20, 50 x 3 per phase, weighted 0.2 / 0.6 / 0.2); order = run order

| run | stack | early ms | mid ms | late ms | weighted img/s | mid img/s | repeat spread mid (ms) |
|---|---|---:|---:|---:|---:|---:|---|
| FINAL_s11 | FINAL | 101.16 | 101.02 | 100.47 | 69.35 | 69.29 | 100.84, 101.01, 101.21 |
| FINAL_s13 | FINAL | 101.20 | 101.06 | 100.47 | 69.33 | 69.27 | 100.88, 101.05, 101.24 |
| P3H_H1_s14 | P3H_H1 | 93.88 | 93.96 | 93.39 | 74.61 | 74.50 | 93.76, 93.98, 94.13 |
| P3H_H2_s15 | P3H_H2 | 96.43 | 96.49 | 95.99 | 72.63 | 72.55 | 96.29, 96.51, 96.67 |
| P3H_s12 | P3H | 90.53 | 90.31 | 89.64 | 77.59 | 77.51 | 90.17, 90.27, 90.47 |

vitl16 per stack (mean over its runs; ratio vs FINAL = mean of the FINAL runs of the same session):

| stack | runs | mid ms | weighted img/s | vs FINAL (weighted) | vs FINAL (mid ms) |
|---|---:|---:|---:|---:|---:|
| FINAL | 2 | 101.04 | 69.34 | 1.0000x | 1.0000x |
| P3H_H1 | 1 | 93.96 | 74.61 | 1.0759x | 1.0754x |
| P3H_H2 | 1 | 96.49 | 72.63 | 1.0475x | 1.0471x |
| P3H | 1 | 90.31 | 77.59 | 1.1190x | 1.1188x |

#### vitb16: whole-step bench under StepGraphRunner (warmup 20, 50 x 3 per phase, weighted 0.2 / 0.6 / 0.2); order = run order

| run | stack | early ms | mid ms | late ms | weighted img/s | mid img/s | repeat spread mid (ms) |
|---|---|---:|---:|---:|---:|---:|---|
| FINAL_s11 | FINAL | 87.94 | 87.42 | 86.05 | 160.46 | 160.15 | 87.22, 87.45, 87.59 |
| FINAL_s13 | FINAL | 88.17 | 87.59 | 86.18 | 160.13 | 159.83 | 87.43, 87.60, 87.75 |
| P3H_H1_s14 | P3H_H1 | 80.46 | 79.47 | 77.68 | 176.52 | 176.16 | 79.42, 79.47, 79.52 |
| P3H_H2_s15 | P3H_H2 | 83.20 | 82.64 | 81.16 | 169.79 | 169.41 | 82.51, 82.65, 82.77 |
| P3H_s12 | P3H | 77.10 | 76.33 | 74.72 | 183.82 | 183.42 | 76.18, 76.34, 76.47 |

vitb16 per stack (mean over its runs; ratio vs FINAL = mean of the FINAL runs of the same session):

| stack | runs | mid ms | weighted img/s | vs FINAL (weighted) | vs FINAL (mid ms) |
|---|---:|---:|---:|---:|---:|
| FINAL | 2 | 87.51 | 160.30 | 1.0000x | 1.0000x |
| P3H_H1 | 1 | 79.47 | 176.52 | 1.1012x | 1.1011x |
| P3H_H2 | 1 | 82.64 | 169.79 | 1.0592x | 1.0589x |
| P3H | 1 | 76.33 | 183.82 | 1.1468x | 1.1464x |


### 4.7 nsys kernel 类别（FINAL 对 P3H，mid 相，5 步，graph 节点追踪）

#### vitl16（FINAL → P3H，ms/步，5 步平均）

| 类别 | FINAL | P3H | 差 |
|---|---:|---:|---:|
| GEMM MXFP8 (QuACK block-scaled, backbone) | 24.33 | 26.11 | +1.78 |
| attention (flash fwd/bwd, RoPE) | 16.02 | 15.95 | -0.07 |
| GEMM bf16 head (QuACK: ConvT fwd/dgrad, einsum bwd) | 11.76 | 1.45 | -10.32 |
| optimizer (r15 flat: sumsq_prep + update) | 7.72 | 7.72 | +0.00 |
| GEMM bf16 head (cutlass sm80: ConvT wgrad, einsum, mask head) | 7.21 | 1.64 | -5.57 |
| elementwise / copy / reduce / other | 6.90 | 6.71 | -0.19 |
| r14 head fused kernels (GELU+dwconv+LN2D fwd/bwd) | 6.63 | 6.70 | +0.07 |
| LN / row kernels (r13 CUDA + torch LN) | 5.88 | 5.90 | +0.01 |
| MXFP8 quantize (own CUDA quant_dual) | 5.07 | 5.17 | +0.10 |
| GEMM bf16 QuACK backbone (qkv forward, bias epilogue) | 4.77 | 4.76 | -0.01 |
| GEMM NVFP4 (QuACK mxf4nvf4) | 0.00 | 2.07 | +2.07 |
| FP4 quantize (R21 act_kernel: dZ row+col, X col) | 0.00 | 1.69 | +1.69 |
| loss / matcher / mask build | 1.49 | 1.49 | +0.01 |
| conv (cuDNN: patch embed / residual head convs) | 0.72 | 0.71 | -0.00 |
| **kernel 总计** | 98.49 | 88.08 | -10.41 |
| 窗口（首 kernel 到末 kernel） | 100.51 | 89.74 | -10.78 |
| GPU 忙碌率 / >20 µs 间隙 | 0.978 / 0.78 ms | 0.980 / 0.43 ms | |
| kernel 数 / 步 | 2475 | 2514 | +39 |

#### vitb16（FINAL → P3H，ms/步，5 步平均）

| 类别 | FINAL | P3H | 差 |
|---|---:|---:|---:|
| GEMM MXFP8 (QuACK block-scaled, backbone) | 14.39 | 16.44 | +2.05 |
| GEMM bf16 head (QuACK: ConvT fwd/dgrad, einsum bwd) | 13.36 | 2.10 | -11.27 |
| attention (flash fwd/bwd, RoPE) | 12.75 | 12.75 | +0.00 |
| r14 head fused kernels (GELU+dwconv+LN2D fwd/bwd) | 10.31 | 10.40 | +0.09 |
| GEMM bf16 head (cutlass sm80: ConvT wgrad, einsum, mask head) | 8.22 | 2.11 | -6.11 |
| elementwise / copy / reduce / other | 7.78 | 7.64 | -0.15 |
| LN / row kernels (r13 CUDA + torch LN) | 5.05 | 5.03 | -0.02 |
| MXFP8 quantize (own CUDA quant_dual) | 4.13 | 4.37 | +0.24 |
| loss / matcher / mask build | 3.17 | 3.17 | -0.00 |
| GEMM bf16 QuACK backbone (qkv forward, bias epilogue) | 2.71 | 2.72 | +0.01 |
| FP4 quantize (R21 act_kernel: dZ row+col, X col) | 0.00 | 2.60 | +2.60 |
| optimizer (r15 flat: sumsq_prep + update) | 2.25 | 2.25 | -0.00 |
| GEMM NVFP4 (QuACK mxf4nvf4) | 0.00 | 2.18 | +2.18 |
| conv (cuDNN: patch embed / residual head convs) | 1.10 | 1.10 | +0.00 |
| **kernel 总计** | 85.23 | 74.87 | -10.36 |
| 窗口（首 kernel 到末 kernel） | 86.37 | 76.06 | -10.31 |
| GPU 忙碌率 / >20 µs 间隙 | 0.984 / 0.64 ms | 0.982 / 0.41 ms | |
| kernel 数 / 步 | 1641 | 1680 | +39 |


## 5. 正确性与踩到的问题

- **dZ 的 `headroom=8` 确认（协调者的复核要求）**：`Handle.dy_headroom` 默认 8.0，patch 的 `P3H*` 预设没有覆盖它；B 门脚本对每个候选显式设置 `handle.dy_headroom = 8.0`（`H3hr2` 才是 2）；整步 bench 与 nsys 用 `P3H` 预设（默认 8）；S 门的 `state.stats()["headroom"]` 设备端数组读回：dZ 的行 / 列 site = 8.0，X 与权重 = 2.0（8 个场景全部如此，见 `profile/gateS_*_P3H_*.json` 的 `fp4_stats`）。所以所有 B / S / 速度结果都是在 headroom=8 下得到的，没有需要重跑的部分。
- **Gate A 的"逐位相同"来自同一份量化数据 + 同一个 GEMM**：RN + `amax=exact` 时 R21a 量化器对 `fp4_ref`（CUDA）逐位一致，R20b 仿真 head 用的就是 `fp4_ref` 语义的量化数据 + 真 QuACK GEMM，所以整个 head 的 logits 与梯度逐位相同（relL2 = 0）。
- **B 门里候选的 amax 是"delayed"**：单步 = 首次调用的 exact amax × headroom（X、W ×2，dZ ×8），这是稳态 delayed 的真实量化条件，不是 R20b 的"首步 exact"。`H3x`（`amax=exact`）与 `H3rn`（RN）的 B 门数值与 H3 在 4 位小数内相同，说明 headroom 8 与 SR 对单步梯度质量几乎无影响。
- **预热问题（已处理）**：进程里第一次跑 FINAL 时 QuACK autotune / JIT 会让第一批梯度与后面不同（B 门 seed 0 的第一个 FINAL 与 `FINAL_again` 的余弦差 0.004，其余批相同）。`gate_b_real.py` 现在在种子循环前做一遍丢弃式预热；seed 0 在预热版里重算，最终 n = 8（种子 0–7）全部来自预热后的进程。
- **权重量化放在 forward 的第一次 predict，而不是 opt_step 之后标脏**：`StepGraphRunner.warm` 的 eager 预热不调 `opt_step`，如果用"标脏"，捕获第二张 graph 时可能已被预热清掉脏标记而漏掉量化 kernel（重放就会用陈旧权重）。按 forward 刷新天然适应 eager、预热、捕获三种情况，每步仍然只量化一次（predict 序号 = `calls % 4`，`opt_step` 后清零）。
- **graph_equiv 需要复位 FP4 状态**：r16 的 `action_graph_equiv` 不知道 `FP4State`，eager 的两次运行之间状态会前进（seed、amax 历史）。`run.py` 里自己的 `action_graph_equiv` 在每次 eager 运行前 `reset_state(full=True)`，graph 运行在预热后 `reset_state(full=False)`（保留 exact 初始化的 prev、清 `next` 槽、seed 复位）。
- **Gate A graph 部分最初失败**：把 `torch.cuda.graph` 捕获放在默认流上创建的 AccumulateGrad 之后会报 `cudaErrorStreamCaptureImplicit`（测试写法问题，不是 kernel 问题）；改成侧流预热（PyTorch 标准的整网捕获写法）后通过。`StepGraphRunner` 本身不受影响（整步 graph 的 S 门、graph_equiv、bench 都正常）。
- **共享构建目录上的陈旧 baton 锁（环境问题，需要协调）**：nsys 的 P3H 抓取在加载 r15 flat 优化器扩展时挂住（`/root/autodl-tmp/train-opt-tmp/round15_optimizer_flat/build/lock` 是 0 字节的 torch cpp_extension baton 文件，时间戳 = 我那个 nsys 进程启动后 14 秒），挂住的进程持有双 GPU 锁，R22 的排队任务也被堵了约 14 分钟。我确认没有任何活着的 ninja / 编译进程后杀掉自己的挂起进程，**删除了这个 0 字节 lock 文件**（时间戳和进程启动时间都表明是我的进程留下的），之后 R22 与我的任务都恢复。r15 的 `build/` 目录是各 round 共用的：两个代理并发加载同一个扩展、中途被杀都会留下这种锁；建议 R23 / 协调者不要在持锁的 nsys 里首次加载共用扩展，或者按代理给 r15 扩展单独的 build 目录（需要改 r15 的 `flat.py`，我没有权限改）。
- **headroom 对照**（6b 13.5）：dZ 用 headroom 2 时（`P3H_HR2`，fixed_mid / compressed 各 50 步，两个模型）dZ 的 delayed amax 在每个场景里有 1 / 200 次超限（最大比值约 2.1）；用 headroom 8 时 8 个场景（含这两个）0 次超限（最大比值 1.85–2.91），而且两种设置的 S 门 EMA 偏差都 ≤ 0.8%，没有可测的区别。8 的代价：dZ 的 ts 大 4 倍，e4m3 block scale 的动态范围少用 2 bit，B 门里 H3 与 `amax=exact` 的 H3x 数值相同到 4 位，说明这个代价看不出来；所以保留 `dy_headroom=8`（不需要退回 `amax=exact`）。
- **不在 checkpoint 里的状态**：`FP4State`（delayed amax 历史、seed 计数）不在 checkpoint 里，resume 后每个 site 的第一次调用重新做 exact 初始化（与首步行为一致），不影响参数 / optimizer 的兼容性；`check_export.py` 的 resume 一步在原版路径上进行。

## 6. 静态分析与 roofline

- head ConvT 的 nv4 GEMM（`bench_head_gemm.json`）：dgrad 1.05–1.25 PFLOPS，wgrad 1.04–1.16 PFLOPS（对 FP4 指令峰值 2.03 PFLOPS 的 51–62%，与 R20a 的 1.1–1.25 PFLOPS 一致；ConvT2 的形状更大，受 L2 / 带宽限制略低）；bf16 对照 0.20–0.24 PFLOPS（限速）。
- FP4 量化 kernel：沿用 R21a 的 roofline（大形状 dual = 0.98–0.99× DRAM 下限，SR +22–28%）。整步里 FP4 量化 L 1.69 / B 2.60 ms，与 R21a 估计（1.77 / 2.65）吻合，所以集成没有引入额外开销。
- `static`：本任务没有新 kernel，没有新的 ptxas / SASS 数据（R21a 的 §6 仍然适用）。

## 7. 结论、推荐、失败项、契约变更请求、下一瓶颈

**结论**：H3 在两个模型上都达到并超过 R20a 估计，通过全部门（A、B n=8 放宽 + 严格、S 4 场景 × 50 步在 StepGraphRunner 下、graph_equiv、导出 / resume），整步 ViT-L ×1.119（77.59 img/s）、ViT-B ×1.147（183.82 img/s）。**建议发布为 `P3H_STACK`（FINAL + `round21_fp4_quant_head`，策略 H3，`dy_headroom=8`）**。

**H1 / H2 对照**：H1（前向 bf16，只做 dgrad / wgrad nv4）L ×1.076 / B ×1.101；H2（r14 MXFP8 反向）L ×1.048 / B ×1.059；H3 的前向 MXFP8 再多给 +4.0% / +4.1%（对 H1）。三者 B 门都过严格门，H3 在 B 门里最差（对 R0 最小 0.9884 / 0.9871，仍高于严格门的 0.98）。

**失败项**：无门失败。Gate A graph 部分第一版的测试写法问题（见 §5）、B 门第一个 FINAL 的预热污染（已处理）、陈旧 baton 锁（已处理并上报）。

**契约变更请求**：无。两点说明：（1）6b 13.3 说"`StepGraphRunner` 的预热步已经满足 delayed 初始化"——实测成立（预热的 eager fwd+bwd 做 exact 初始化），但预热**不调用 `opt_step`**，所以"每步开头标脏"的做法在捕获时不安全；本实现改为"每次 forward 的第一次 predict 刷新权重副本"，建议 R22 的骨干权重量化也按这个方式处理（R22 如果已经在 `begin_step` 里做，需要自查捕获第二张 graph 的情形）。（2）建议 `common/config.py` 新增 `P3H_STACK = BEST_STACK_V2 + ("round21_fp4_quant_head",)`，`P3H_KWARGS` 里该 patch 的 `variant` 为 `head.f=mx8,head.dw=nv4+sr`。

**dZ 量化并进 r14 `bwd2` 输出的评估（只评估，没有实现）**：nsys 里 dZ 的 FP4 量化约占 FP4 量化类的 85%（L ≈ 1.45 ms、B ≈ 2.2 ms / 步；X 列副本 L 0.24 / B 0.35 ms），`bwd2` 本身 L 2.53 / B 3.70 ms。融合能省掉：量化 kernel 重读 dZ（L ≈ 0.78 ms）和 `bwd2` 写 bf16 dZ（2 B / 元素，改写 FP4 两份 ≈ 1.06 B / 元素，L ≈ 0.4 ms），上界约 1.2 ms（L）/ 1.8 ms（B）；扣掉在 `bwd2` 里增加的 SR / Philox ALU（R21a：SR 使量化 kernel 慢 22–28%，约 0.35 ms）和对 `bwd2` 占用率的影响（`bwd2` 已是 4 通道 / 线程、每 SM 2 个 CTA 的配置），实际可得 ≤ 约 1 ms / 步（L ≈ 1.1%、B ≈ 1.3%），低于 2% 的发布线。实现复杂度高：`bwd2` 的 CTA 按图像行（w = 32 / 64 个像素）切，行副本的 16 元素块沿 4C 维是 16 个相邻通道，而 `bwd2` 每线程只持有 4 个通道，要 4 个线程 shuffle；转置副本的 16 token 块 = 一行里连续 16 个像素（可行）但每列只写 8 B，需要 smem 转置才能凑满 32 B sector，同时要产出行 / 列两份、做 delayed amax 的 atomicMax、SR 的 Philox 计数与 R21a 的 flat index 对齐（Gate A 的逐位参照要重做）。**不建议在 R23 做**；如果以后骨干也到瓶颈，再考虑与骨干 dY 的量化一起做。

**下一个瓶颈（ViT-L / ViT-B，P3H 的 nsys，ms / 步）**：骨干 MXFP8 GEMM 26.1 / 16.4（R22 在做骨干 dgrad / wgrad 的 FP4）、attention 16.0 / 12.8、优化器 7.7 / 2.3（`update_kernel` 单 kernel 6.6 ms，L）、r14 的融合 elementwise 6.7 / 10.4（B 上比 FP4 GEMM 还大，内存受限）、LN / row 5.9 / 5.0、MXFP8 量化 5.2 / 4.4、head 里还剩的 bf16 GEMM（einsum、mask_head）3.1 / 4.2。

**给 R23 的建议**：（1）先把 P3H_STACK 与 R22 的骨干 patch 叠起来：两个 patch 各自持有一个 `FP4State`，都包住 `ctx.ops.opt_step`（按应用顺序各 roll 一次），替换的是不同的 op（`block` 与 `predict`），应当可组合，但**需要实测组合栈的 B / S / graph_equiv**（我没有测）；（2）两个模型的 head 策略相同（H3）；（3）ViT-B 骨干 FP4 在最差批上离放宽门只有约 0.005（6b 已提醒），叠加 H3 后 B 门要盯最小值；（4）head 的 r14 融合 kernel 在 ViT-B 上 10.4 ms，是 head 里下一个值得做的（例如 `bwd1` + `bwd2` 合并，减少 dz1 / du 的往返），但不属于 FP4 任务。
<!-- R21b-END -->
