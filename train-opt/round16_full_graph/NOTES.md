# Round16：整步 CUDA Graph（负责：Agent A2，2026-10-03）

> 全部数字为新容器（RTX 5090，575 W 上限）实测，B 固定（ViT-L 7 / ViT-B 14），三相 early/mid/late 按 0.2/0.6/0.2 加权，warmup 20，50×3，`common.bench.time_steps`。栈 = `P2_STACK + round15_optimizer_flat`（简称 **P2,r15**）。原始 json 在 `profile/`，nsys 原始文件在 `/root/autodl-tmp/train-opt-tmp/round16_full_graph/nsys/`。
> 同机另有其它代理的任务在交替使用 GPU（round14 的队列）：所有测量都持双锁，并用 `tests/locked_run.sh` 在“对方只是带着空闲 CUDA 上下文在等锁”时放行（见 §7 的契约变更请求）；每个 bench json 里的 `gpu_snapshot_before/after` 显示计时期间 GPU 上只有本进程。

## 1. 目标与假设

- P2（MXFP8）把 GPU kernel 时间压到 120 ms，但 ViT-L 墙钟 136 ms：GPU 忙碌率 0.80，>20 µs 间隙 17 ms，每块 host 时间超过 GPU 时间（QuACK 的 Python 派发、每块 12 次量化 launch、每步 7–8 次同步）。整步 CUDA Graph 把 host 开销整体移出关键路径。
- 目标：一次 `graph.replay()` 完成 前向、loss（r3 padded targets）、反向、clip、AdamW（r15）、设备端 lr 与 step；静态输入加侧 stream H2D；按 mask 阶段签名惰性捕获、共享池；修复 round10 的捕获失败；eager 对 graph 对比、nsys 前后、门槛；列出不可捕获部件与接入方式。

## 2. 改动

| 文件 | 内容 |
|---|---|
| `impl/graph_step.py` | `StepGraphRunner`：静态输入（image、padded `masks uint8 [B,T,H,W]`/`labels`/`valid`）；`stage(next_batch)` 在侧 stream 上 host(pinned) 或 GPU 批 → staging，下一步开始前计算流只做一次 D2D（staging → static）并记录 `consumed` 事件；按 mask 签名（每个 joint 块 `one/middle/zero`，共 7 种出现顺序）惰性捕获，**共享一个 memory pool**，LRU `max_resident=8`；捕获前对该调度会访问的每个签名做一次 eager 预热（fwd+bwd，不更新优化器，之后梯度清零）；捕获本身不执行 kernel，因此不会动权重；`middle` 状态下每步只做一个 `fill_` 更新设备上的 `attn_mask_probs`；train metrics 默认 lazy（见 §3）；形状不符/目标数超过 `max_targets` 的批退回普通 eager 步。 |
| `run.py` | 与 round15 共用：`--action bench/profile/gate_s/graph_equiv/gate_b/opt_bench`，`--runner eager|graph`，`--metrics step|lazy|off`，`--cpu-batches`（批留在 host，走 pinned H2D 路径）。 |
| `tests/` | `probe_r10_capture_failure.py`（round10 捕获失败的根因）、`test_fallback_and_swap.py`（溢出回退、LRU）、`locked_run.sh`、`nsys_inner.sh`/`nsys_queue.sh`、`trace_busy.py`（由原始 kernel trace 算忙碌率/间隙）、`summarize.py`、`finalize_results.py`（由 profile json 和 nsys 导出重建 results.json）、`queue_run.sh`/`list_*.txt`/`driver3.sh`/`final_queue.sh`（实际跑过的队列）。 |

**round10 失败的根因（已修，没有改 round10）**：`round7_optimizer/impl/fused_optimizer.py:25` 的 `scale = torch.as_tensor(max_norm, dtype=..., device=cuda)` 把 Python float 变成 CUDA 张量，即捕获中从 pageable 内存做 H2D，报 “Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned”。`tests/probe_r10_capture_failure.py` 在 P2 栈的 ViT-B 上复现：不加 `--fix` 时在该行失败（前向、r3 padded loss、反向都已捕获成功），加 `--fix`（`(total+eps).reciprocal().mul(max_norm).clamp(max=1)`，无 H2D）后整步可捕获并回放（日志 `profile/r10_failure_*_vitb16.log`）。round16 直接用 round15 的 flat 优化器（clip 系数在设备上算），不依赖 r7 的 clip。

## 3. nsys 分析过程

**命令**（持双锁；原始文件在 tmp，项目里只放 csv/summary；`tests/nsys_queue.sh` = `locked_run.sh` + `nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-graph-trace=node --capture-range=cudaProfilerApi` + `nsys_analyze.py --iters 5` + r17 的 `nsys_step_categories.py`）：

```bash
tests/nsys_queue.sh v1 vitl16:P2:eager vitl16:P2,r15:eager vitl16:P2,r15:graph vitb16:P2:eager vitb16:P2,r15:eager vitb16:P2,r15:graph   # v1：graph 每步更新 train_metrics
tests/nsys_queue.sh v2 vitl16:P2,r15:graph vitb16:P2,r15:graph                                                                          # v2：metrics lazy（最终）
python tests/trace_busy.py <prefix> ...      # 窗口 = 首个 kernel 起点到最后一个 kernel 终点（5 步），NVTX 的 step 区间对异步 graph 回放只是 CPU 时间，不能算忙碌率
```

**3.1 整步（mid 相，5 步窗口，由原始 kernel trace 计算；nsys 开销使 eager 的 host 更慢，窗口时间比 bench 长）**

| 模型 / 配置 | kernel 数/步 | kernel 总时间 ms/步 | GPU 忙碌率 | >20 µs 间隙 个/步 → ms/步 | 窗口 ms/步 |
|---|---|---|---|---|---|
| ViT-L P2 eager（r7 优化器，现状） | 3294 | 120.0 | **0.797** | 324.8 → 21.4 | 150.5 |
| ViT-L P2+r15 eager | 2809 | 117.3 | 0.845 | 304.0 → 16.4 | 138.8 |
| ViT-L P2+r15 graph v1（每步更新 train_metrics） | 2580 | 115.9 | 0.958 | 7.4 → 3.5 | 120.9 |
| ViT-L P2+r15 **graph v2（最终，metrics lazy）** | 2573 | 115.7 | **0.981** | 1.0 → 0.75（其中一个 3.5 ms 间隙在窗口起点） | 117.9 |
| ViT-B P2 eager | 2150 | 114.4 | 0.862 | 173.6 → 11.6 | 132.8 |
| ViT-B P2+r15 eager | 1892 | 112.8 | 0.874 | 193.0 → 12.5 | 129.1 |
| ViT-B P2+r15 **graph v2** | 1767 | 112.7 | **0.984** | 2.2 → 0.77 | 114.5 |

**3.2 NVTX / 同步**：eager 的 NVTX（ViT-L，mid）`clip_opt_sched` GPU 投影 17.4 ms、CPU 区间 27.8 ms（P2）→ 8.4 / 6.1 ms（P2+r15）；graph 回放内部没有 NVTX 区间（kernel 节点以 `--cuda-graph-trace=node` 展开）。eager 每步同步 8 → 7 次（nsys 的 `cudaStreamSynchronize` + 同步型 `cudaMemcpyAsync`）；graph 回放本身 0 次同步，v1 的 graph 里每步仍有 1.5–3.4 ms 的间隙，与 `train_metrics.update_with_losses` 同时出现：v1 每步 5 次同步（3 次 `cudaStreamSynchronize` + 2 次同步型 `cudaMemcpyAsync`），host 等 GPU 之后才发下一次 replay；改为 lazy（日志步才更新）后同步降到 0 次/步、间隙消失（v2：忙碌率 0.958 → 0.981，bench +0.45 img/s，见 §4）。同步的具体来源（torchmetrics 内部的 nan/inf 检查）没有逐调用验证，只验证了“去掉这次调用后全部同步消失”。

**3.3 前后 top kernel（ViT-L graph v2）**：kernel 构成与 eager 相同（flash bwd 10.2 ms、QuACK MXFP8/bf16 GEMM、head 的 cuDNN 卷积、量化 5.1 ms），`update_kernel` 6.65 ms 取代 480 个优化器 kernel；graph 少掉的 kernel 主要是 host 侧小 kernel（r7 的 235 个 `zero_grad` foreach 等）。类别（ViT-L，ms/步）：GEMM（QuACK）29.8、elementwise/copy/其他 26.5（含 K2 6.65、梯度累加 1.5）、head 卷积 22.3、attention 16.3、LN/行 kernel 9.1、bf16 GEMM（cuBLAS）6.2、量化 5.1、loss 0.2。ViT-B：elementwise 31.0、head 卷积 30.5、GEMM 17.6+6.2、attention 12.9、LN 10.0、量化 4.1。

**3.4 结论**：graph 把忙碌率从 0.80（ViT-L）/ 0.86（ViT-B）推到 0.98，几乎兑现了全部 host 间隙；残余空闲 < 2%（窗口 117.9 ms 对 kernel 115.7 ms）。下一步只能靠减少 kernel 时间。

## 4. 实测结果

**4.1 整步（三相加权 img/s；括号为 mid 步时 ms）**

| 栈 / runner | ViT-L B=7 | ViT-B B=14 |
|---|---|---|
| P2 eager（r7 优化器，现状） | 51.68（135.6） | 109.92（127.6） |
| P2 + r15 eager | 56.00（125.1） | 116.57（120.3） |
| **P2 + r15 graph（metrics lazy）** | **59.51（117.8）** | **123.11（113.9）** |
| P2 + r15 graph（每步更新 train_metrics） | 59.06（118.8） | 122.55（114.4） |

倍数：graph / P2 现状 = **1.152×（ViT-L）/ 1.120×（ViT-B）**；graph / (P2+r15 eager) = 1.063× / 1.056×；对 P1R13（43.7 / 100.7）1.36× / 1.22×，对 P1（41.4 / 96.0）1.44× / 1.28×，对 R0（16.2 / 37.2）3.67× / 3.31×。三个 repeat 的均值很稳（ViT-L graph mid：118.3/118.9/119.1 ms，lazy 版见 `profile/bench_*_graph_lazy_all.json`）。

**4.2 捕获**：3 相 bench 共 3 次捕获（early `[one,one,one]`、mid `[zero,middle,one]`、late `[zero,zero,zero]`）：ViT-L 0.27 / 0.09 / 0.09 s，ViT-B 0.22 / 0.06 / 0.07 s；7 个签名的 eager 预热合计 6.3 s（ViT-L）/ 6.2 s（ViT-B，一次性）。共享池：ViT-L 捕获后 reserved 17.6–18.1 GB（eager 峰值 17.3 GB allocated），ViT-B 15.2–15.4 GB；压缩调度（7 个签名都出现）的捕获次数见 §5。真实调度下 7 个签名按顺序各出现一次，不会回头，总捕获时间 < 1 s。

**4.3 其它配置**（`--phases mid`）：见下表，未测的写 pending。

| 配置 | ViT-L mid ms（img/s） | ViT-B mid ms（img/s） |
|---|---|---|
| graph，批在 GPU（基准） | 117.8（59.4） | 113.9（122.9） |
| graph，批在 host（pinned H2D 走侧 stream） | 116.4（60.1） | 113.3（123.5） |
| graph，每步更新 train_metrics（`--metrics step`） | 117.8（59.4） | 114.0（122.8） |

（只测 mid 相，50×3；与基准相差 ≤1.2%，在共享 GPU 上的 run-to-run 噪声内：pinned 内存的 H2D 在侧 stream 上与回放重叠，没有可见代价；每步更新 train_metrics 在 bench 里只贵 0.5–0.8%，但在 nsys 里让每步多出 1.5–3.4 ms 间隙，见 §3.2。）

## 5. 正确性

**5.1 图 vs eager，同一 RNG（`--action graph_equiv`，每步 `torch.cuda.manual_seed(7000+k)`，eager 参考也用 padded targets，因为 r3 的 padded 与非 padded 路径抽取的点采样随机数形状不同；ViT-B，12 步）**

| 场景 | 首步 loss 逐位相同 | graph vs eager：loss 相对差最大 / grad_norm 相对差最大 / 更新相对 L2 / 更新余弦 | eager vs eager（同样两次运行）的更新相对 L2 / loss 相对差最大 |
|---|---|---|---|
| fixed mid | 是 | 1.1e-4 / 1.4e-4 / 3.3e-3 / 0.999995 | 3.7e-3 / 1.2e-4 |
| compressed（4 个签名，每次切换重新捕获） | 是 | 1.4e-4 / 3.4e-3 / 9.2e-3 / 0.999958 | 1.2e-2 / 2.1e-4 |
| fixed late | 是 | 4.2e-5 / 7.4e-4 / 2.2e-3 / 0.999998 | 1.9e-3 / 3.5e-5 |
| schedule（真实调度，前 12 步） | 是 | 3.1e-4 / 1.3e-3 / 1.3e-2 / 0.999918 | 1.2e-2 / 2.4e-4 |

结论：首步 loss 逐位相同（说明 graph 与 eager 的随机数序列、前向、loss 完全一致）；之后的差异与 **eager 自己两次运行之间的差异同量级**（backward 里有非确定性 kernel：flash bwd / split-K 原子加；Adam 的更新量对梯度噪声敏感，所以更新相对 L2 在 1e-3–1e-2），所以参数不是逐位相同，而是在 eager 的噪声内（参数最大绝对差 ≤ 5e-6，权重量级 ~1，一步更新量 ~1e-4）。ViT-L（`fixed_mid`，6 步）：首步 loss 逐位相同；loss 相对差最大 4.0e-5（eager 自己两次 1.2e-5），grad_norm 相对差最大 2.6e-4，更新相对 L2 2.5e-3 / 余弦 0.9999968（eager vs eager 2.6e-3 / 0.9999966）；参数最大绝对差 3.6e-7。ViT-L 的另外三个场景没有单独跑等价测试（第一次尝试在统计阶段 OOM：三份 1.26 GB 参数快照 + 11 GB 的 graph 池，已改为 CPU 分块统计），其覆盖由 5.2 的 Gate S 提供。

**5.2 Gate S（`--action gate_s`，2 个真实 batch、50 步，graph 路径对 eager 路径，`gate_c(interval=10, max_rel=0.05)`，无 NaN/Inf；两边都是 P2+r15；eager 走标准 list-targets 路径，graph 走 padded targets）**

| 场景 | ViT-B：EMA 最大相对偏差 / 捕获数 | ViT-L：EMA 最大相对偏差 / 捕获数 | grad_norm 比值范围（B / L） |
|---|---|---|---|
| schedule（真实调度，50 步内只处于 `one` 签名） | 1.0e-3 / 1 | 9.2e-4 / 1 | 0.998–1.007 / 0.981–1.030 |
| compressed（退火压缩到 3–21 步，7 个签名全部出现） | 1.2e-3 / 7 | 1.3e-3 / 7（捕获 0.46 s + 6×约 0.1 s） | 0.939–1.019 / 0.990–1.009 |
| fixed mid | 7.8e-4 / 1 | 1.6e-3 / 1 | 0.978–0.998 / 0.957–1.023 |
| fixed late | 4.8e-4 / 1 | 1.0e-3 / 1 | 0.997–1.014 / 0.983–1.012 |

8 个组合全部 PASS（阈值 5e-2，实际 ≤1.6e-3）。`profile/gateS_vit{b,l}16_P2_r15_graph_vs_eager.json`。

**5.3 边界（`tests/test_fallback_and_swap.py`，ViT-B，`max_resident=1`）**
- 目标数超出静态容量（一张图 3 个目标 > `max_targets=2`）：自动退回 eager 步，loss 有限（30.04），`eager_fallbacks=1`；之后同签名的 graph 继续回放，**没有重新捕获**。eager 步与驻留的 graph 池同时存在时峰值 reserved 28.4 GB（ViT-B）。**ViT-L 上同样的回退需要约 18 GB（池）+ 约 17 GB（eager）> 32 GB，预计 OOM，未测**；所以 `max_targets` 必须覆盖数据集的最大目标数（本轮观察到的 4 个 batch 每图 ≤2 个目标，整个数据集的最大值没有统计；容量上限 4）。
- LRU 淘汰/重捕获：phase `mid → late → mid` 在 `max_resident=1` 下捕获 3 次、淘汰 2 次，reserved 回到 15.3 GB，losses 有限。第一次运行在这里崩了：最后一个驻留 graph 被淘汰后 allocator 释放了共享池，再往旧的 pool handle 捕获触发 `it->second->use_count > 0 INTERNAL ASSERT FAILED`（`CUDACachingAllocator.cpp:3049`）。已修复：池变空时重新申请 `torch.cuda.graph_pool_handle()`。默认 `max_resident=8` 足够放下全部 7 个签名，真实调度从不淘汰。注意：共享池中淘汰一个 graph 不会立即释放显存，只有池里所有 graph 都没了才释放。
- r15 状态一致性：捕获不执行 kernel、预热后梯度清零，所以捕获前后权重、m、v、计数器不变（graph_equiv 的 eager 与 graph 从同一个快照开始并得到相同首步 loss）。

## 6. 静态分析

本轮没有新的自研 kernel（flat 优化器的静态分析在 round15）。PDL 没有尝试：PyTorch 捕获出的 kernel 节点没有 programmatic-stream-serialization 属性，逐节点补属性不是“一行改动”。

## 7. 结论、失败的尝试、不可捕获部件、接入方式、下一瓶颈

**结论**
- P2+r15 整步 graph：ViT-L 51.68 → 59.51 img/s（1.152×，其中 r15 贡献 1.084×，graph 再 1.063×），ViT-B 109.92 → 123.11（1.120×，r15 1.060×，graph 再 1.056×）；GPU 忙碌率 0.80 → 0.98。捕获很便宜（每个签名约 0.1–0.3 s），7 个签名共享一个池，显存与 eager 同量级。
- 图 vs eager 的首步 loss 逐位相同，其后差异在 eager 自身的非确定性之内。

**发现的坑 / 失败**
1. round10 的失败根因是 r7 的 clip 里 `as_tensor(python float, device=cuda)`（见 §2），不是整步里别的部件：fwd、loss、bwd、MXFP8/r13 融合块、head、attention 都能捕获。
2. `train_metrics.update_with_losses`（torchmetrics）每步让 host 等 GPU：graph v1 每步 1.5–3.4 ms 的间隙；lazy 后消失。eager 的 `harness.train_step` 里同样有这次同步（P2 eager 每步 8 次同步之一，未单独量化）。训练 loss 指标每步被覆盖，所以只在日志步调用 `runner.update_metrics()` 即可。
3. 共享池 + 多个 eager 预热：必须在第一次捕获**之前**把所有签名预热完（`runner.prepare`），否则第二个签名的 eager 预热会和第一个图的池（~17 GB）同时驻留而 OOM（ViT-L 32 GB 卡）。
4. 淘汰最后一个驻留 graph 后必须换新的 pool handle（allocator 内部断言，已修，见 §5.3）；graph 驻留时 eager 回退在 ViT-L 上会 OOM（见 §5.3）。
5. `bench.time_steps` 之后 `torch.cuda.max_memory_allocated()` 不含 graph 私有池（ViT-L 6.1 GB、ViT-B 2.1 GB 的“峰值”是假的）：results.json 的 graph 变体改记捕获后的 reserved（ViT-L 18.1 GB、ViT-B 15.4 GB，对照 eager 峰值 17.3 / 14.9 GB）。

**不可捕获 / 受限的部件**
- r7 `fused_clip_grad_norm_`（H2D 标量）：用 r15 替代；r7 `capturable_torch` 的其余部分可捕获，但 `P1R13`/`P2` 栈里的 r7 优化器路径不要用于 graph。
- host 侧：train metrics 的每步更新（改 lazy）、host scheduler（graph 内不需要，lr 是设备函数；`flat.sync_host()` 在存 checkpoint 时同步）、mask 概率更新（`middle` 状态每步一个 `fill_`，图外）、数据加载与 H2D（侧 stream）、nancapture 监控（Lightly 循环里的 `check_and_capture`）。
- 数据相关形状：r3 padded loss 每张图最多 4 个目标（`max_targets` ≤ 4）；超过容量或形状变化的批退回 eager（`report.eager_fallbacks`，见 §5.3）。
- JIT / 首次初始化（QuACK、Triton、cuDNN SDPA 计划、CuTe）必须在捕获前完成：`runner.prepare()` 对每个签名做 eager 预热；不要在捕获期间触发新的 shape。
- `ctx.rng` 必须是 `RngTape(mode="off")`；graph 模式随机数由 torch 默认 CUDA generator 的 philox（捕获时注册，回放时推进偏移）生成，B 检查（逐元素对 tape）仍应跑 eager。

**怎么接别的栈**
```python
ctx = build(cfg); apply_patches(ctx)            # 栈必须含 r3 padded loss、r15 flat 优化器（或实现了 adopt_grads/zero_grad/graph_mode 的可捕获 ops.opt_step）
batches = preload_batches(ctx, 2)
runner = StepGraphRunner(ctx, batches[0], max_targets=2)
runner.prepare(batches[0])                      # 预热全部 mask 签名（eager fwd+bwd，不更新优化器）
for step, batch in enumerate(loader):
    out = runner.step(batch, step, next_batch)  # StepOut(loss, {}, grad_norm)；next_batch 用于侧 stream 预取
runner.update_metrics()                         # 日志步前调用一次，再 train_metrics.compute_aggregated_values()
# 基准：bench.time_steps(ctx, batches, step_fn=runner.as_step_fn(batches), segments=False)；time_steps 之后调用 runner.sync_probs_state()
```
要求：step 路径无 host 同步、无 pageable H2D、无数据相关形状；所有 kernel 在当前 stream；workspace 常驻；随机数只来自 torch。

**契约变更请求**
1. 发布栈：`P2R15_STACK = P2_STACK + ("round15_optimizer_flat",)`，`P2R15_PATCH_KWARGS = {**P2_PATCH_KWARGS, "round15_optimizer_flat": {"variant": "flat"}}`（run.py 里写 `P2,r15`）；整步 graph 的入口是 `round16_full_graph/impl/graph_step.py::StepGraphRunner`（`run.py --runner graph`），需要 `ctx.rng = RngTape(mode="off")`。
2. `common/bench.assert_gpu_exclusive` 会被“别的进程只是持有空闲 CUDA 上下文在等锁”打断（同机并行代理时必现；r17 已提过）。本轮用 `tests/locked_run.sh` 绕过：持锁后若 GPU 利用率 ≤ 3% 连续 3 秒，则通过 `TRAIN_OPT_GPU_ALLOWED_PIDS` 放行这些空闲上下文。建议把这个逻辑放进 `common/`。
3. `common/harness.train_step` 每步调用 `train_metrics.update_with_losses`（host 同步）；建议 lazy 化。
4. nsys：`common/nsys_analyze.py` 的 `gpu_busy_ratio`/`syncs_per_step` 以 NVTX step 区间为分母，对异步 graph 回放无意义（出现 15.3、25.8 这样的“忙碌率”）；本轮用 `tests/trace_busy.py` 由原始 kernel trace 计算。

**下一瓶颈**（graph 之后 GPU 忙碌 98%，只剩 kernel 时间）：ViT-L：head 卷积 22.3 ms（19%，r14）、elementwise/copy 26.5 ms（含 K2 6.65、梯度累加 1.5、head 的逐元素）、attention 16.3 ms（r12）、MXFP8/bf16 GEMM 36 ms、量化 5.1 ms（现在才值得做生产者融合）、LN/行 kernel 9 ms。ViT-B：head 卷积 30.5 ms + elementwise 31.0 ms 占一半以上。优化器还可以去掉梯度累加与清零写（约 2.5 ms，见 round15 §7）。

**未测**：`P1R13,r15` 的 graph 回退档（P2 能整步捕获，不需要回退）；PDL 开/关对照；ViT-L 在 graph 驻留时的 eager 回退（预计 OOM，§5.3）；ViT-L 的 fixed_late / compressed / schedule 等价测试（Gate S 覆盖）。nsys 的原始 `.nsys-rep/.sqlite` 在 `/root/autodl-tmp/train-opt-tmp/round16_full_graph/nsys/`，项目里 `profile/nsys/` 只放 summary、analysis json 和 kernel/NVTX csv。
