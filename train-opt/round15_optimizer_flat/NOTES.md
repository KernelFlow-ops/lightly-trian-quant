# Round15：扁平优化器（flat fp32 master/grad + 两个 kernel 的 clip+AdamW）（负责：Agent A2，2026-10-03）

> 全部数字为新容器（RTX 5090，575 W 上限）实测，B 固定（ViT-L 7 / ViT-B 14）。命令都在 `run.py` / `tests/` 下，原始 json 在 `profile/`，nsys 原始文件在 `/root/autodl-tmp/train-opt-tmp/round16_full_graph/nsys/`（round16 与本轮共用同一批 nsys）。

## 1. 目标与假设

- 目标：把 P2 栈的优化器（r7 `capturable_torch`：foreach 范数 + torch fused AdamW + foreach zero_grad + host scheduler，ViT-L 每步 480 个 kernel、GPU 10.7 ms、host 27.8 ms）换成 **两个 kernel**：
  1. 确定性的全局平方和（顺带在设备上算出 clip 系数、step 计数、分组 lr 表）；
  2. AdamW 更新（clip、decoupled weight decay、bias correction、梯度清零）。
- lr / weight decay 来自设备上的分组表（覆盖 layer-wise LR decay 与 no-WD 组），TwoStageWarmupPoly 学习率调度和 step 计数都在设备上，整个优化器步可被 CUDA Graph 捕获（r16 用）。
- 假设：优化器是 DRAM 带宽受限（每元素 36 B：K1 读 g 4 B；K2 读 p/g/m/v 16 B、写 p/m/v/g 16 B），实测可达带宽约 1.5–1.65 TB/s，所以 ViT-L（3.15 亿 float）下界约 7.2 ms，目标 6–7 ms 只有在去掉梯度清零写（4 B）或把 Σg² 融进 wgrad 才可能达到（见 §7）。eager 整步的收益主要来自 host 时间（r7 的 Python 路径 27.8 ms host）。

## 2. 改动

| 文件 | 内容 |
|---|---|
| `impl/flat_adamw.cu` | K1 `sumsq_prep_kernel`（固定网格 8×SM，每块按固定顺序累加，最后一个到达的块用 ticket 做 double 定序归约；同一个 kernel 的尾部计算 norm、clip 系数、step/scheduler 计数、每组 lr/decay/bias-correction 表）；K2 `update_kernel`（每块一个 4096 float 的 chunk，chunk→组号表；torch AdamW 的数学：`p·(1-lr·wd) − lr/bc1·m/(√v/√bc2+eps)`，lerp 形式的 m，读完立即把 g 清零）。`--fmad=false`。 |
| `impl/flat.py` | `FlatParams`（flat `master/grad/m/v`；每个 Parameter 的 `.data` 与 `.grad` 是 master/grad 的视图，按 4096 对齐，非连续（channels_last）参数用 as_strided 保持布局；`register_weight_writer(fn)` 接口）和 `FlatAdamW`（分组表、设备计数器、`step()`、首个 eager 步里接管梯度并识别无梯度参数）。 |
| `impl/patch.py` | `round15_optimizer_flat`（变体 `flat`）：覆盖 `ops.opt_step`；包装 `optimizer.state_dict/load_state_dict/zero_grad` 与 `scheduler.state_dict/load_state_dict`，导出/加载 torch AdamW 的格式。要求/冲突都为空，排在 r7 之后即可（P2 栈字符串：`P2,r15`）。 |
| `run.py` | 与 round16 共用一份：`--action opt_bench`（优化器单独计时，eager 与 graph）、`bench`、`gate_b` 等；栈写法 `P2,r15`。 |
| `tests/test_flat_synthetic.py` | 合成数据 100 步门槛（见 §5）。`tests/check_real_flat.py`：真实模型兼容检查；`tests/real_step_lr.py`：真实梯度、lr>0 的一步对 torch fused。`tests/bench_kernels.py`、`bench_bw.py`：kernel 变体与 DRAM 上限。`tests/queue_*.sh`：队列。 |

## 3. nsys 分析过程

**命令**（持双锁；`tests/nsys_run.sh` 是 `common/profile_nsys.sh` 的副本，区别只是持锁后等 GPU 独占而不是立即退出，因为同机有别的代理的进程在交替使用 GPU）：

```bash
cd round16_full_graph && tests/nsys_queue.sh v1 vitl16:P2:eager vitl16:P2,r15:eager vitl16:P2,r15:graph vitb16:P2:eager vitb16:P2,r15:eager vitb16:P2,r15:graph
#  = nsys profile ... python run.py --model M --stack S --runner eager|graph --action profile --phases mid --warmup 20  (5 步窗口)
#  python common/nsys_analyze.py <prefix> --iters 5 --top 25 ;  python round17_mxfp8_sm120/tests/nsys_step_categories.py <prefix>
```

**优化器 kernel 前后（ViT-L mid，ms/步，5 步平均，`*_cuda_gpu_kern_sum.csv`）**

| | r7 `capturable_torch`（P2 现状） | r15 flat |
|---|---|---|
| AdamW（torch fused `multi_tensor_apply`，235 次） | 6.97 | – |
| 全局范数（foreach_norm，3 次）+ clip 乘（foreach_mul，6 次）+ zero_grad foreach（235 次）+ 其他 | 1.66 + 0.78 + 0.75 + 0.28 + 0.29 | – |
| K1 `sumsq_prep`（1 次） | – | 0.78 |
| K2 `update`（1 次，含梯度清零） | – | 6.65 |
| **合计（`optimizer / grad clip` 类）** | **10.73 ms，480 个 kernel** | **7.43 ms，2 个 kernel**（类别表里的 0.29 是 r13 的 bf16 影子刷新，不属于优化器） |
| NVTX `clip_opt_sched`：GPU 投影 / CPU 区间 | 17.4 / 27.8 ms | 8.4 / 6.1 ms |

ViT-B：GPU 投影 9.1 → 2.7 ms，CPU 区间 34.3 → 3.6 ms。

**整步（ViT-L mid，eager）**：kernel 总时间 120.17 → 117.53 ms，kernel 数 3294 → 2809，GPU 忙碌率 0.796 → 0.872，>20 µs 间隙 323 个/17.3 ms → 304 个/15.7 ms，同步 8 → 7 次/步。ViT-B：114.48 → 112.88 ms，2150 → 1892 个，忙碌率 0.860 → 0.916。

**结论**：eager 下优化器的 host 路径（27.8 ms/步，host 受限）比 GPU 时间更要命，换成两个 kernel 后整步 −10.5 ms（见 §4），其中只有约 3.3 ms 是 GPU kernel 时间的减少。K2 的 kernel 级带宽 1515 GB/s（32 B/元素），和同机 `copy_`（1526 GB/s）、`add_`（1575 GB/s）一样，已在可达上限；K1 读 1654 GB/s（torch 的 `vector_norm` 为 1683 GB/s）。

## 4. 实测结果

**4.1 优化器单独计时（真实参数集，随机梯度，ms；eager = CUDA event 夹住 `ctx.ops.opt_step`（含 host 停顿）、graph = 仅优化器的 CUDA Graph 回放，中位数，40×3 次）**

| 模型 | torch fused（foreach clip + fused AdamW） | r7 `capturable_torch` | r7 `cuda_multi_tensor` | **r15 flat** |
|---|---|---|---|---|
| ViT-L（361 个张量，3.148 亿参数）eager | 10.22 | 13.10 | 8.46 | **7.43** |
| ViT-L graph 回放 | n/a | **捕获失败**（见下） | 7.51 | **7.42** |
| ViT-B（193 个张量，9226 万参数）eager | 4.60 | 6.58 | 2.70 | **2.16** |
| ViT-B graph 回放 | n/a | **捕获失败** | 2.19 | **2.18** |

- r7 `capturable_torch` 捕获失败的根因：`round7_optimizer/impl/fused_optimizer.py:25` 的 `torch.as_tensor(max_norm, device=cuda)` 在 capture 里做了一次 pageable 内存的 H2D 拷贝（报错 "Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned"）；这也是 round10 `test_full_graph.py` 失败的原因（见 round16）。r15 不再有 host 标量到设备的拷贝。
- **目标 ViT-L ≤ 6–7 ms 没有达到：7.43 ms（eager）/ 7.42 ms（graph）**。受限于 36 B/元素的 DRAM 流量与可达带宽（K2 6.66 ms + K1 0.76 ms），不是 launch。在 graph 里 r15 与 r7 `cuda_multi_tensor`（7.51）基本相同，同样受带宽限制；r15 的差别在 eager（−1.0 ms）、设备上的 lr 调度/step 计数（无需 host 准备元数据）和 flat 缓冲区本身。
- K2 去掉梯度清零写的变体（`tests/bench_kernels.py` 的 `kv4_nozero`）K2 为 5.90 ms（总 6.68 ms），说明达到 6.7 ms 需要“backward 不累加到清零的梯度”（见 §7）。

**4.2 整步（三相加权，eager，B 固定，50×3，warmup 20，img/s）**

| 栈 | ViT-L | ViT-B |
|---|---|---|
| P2（r7 优化器，现状） | 51.68（mid 135.6 ms） | 109.92（mid 127.6 ms） |
| **P2 + r15** | **56.00**（mid 125.1 ms，**1.084×**） | **116.57**（mid 120.3 ms，**1.060×**） |

`results.json` 的 `vitl16/P2r15_eager`、`vitb16/P2r15_eager`、`opt_bench` 下有对应字段。

## 5. 正确性

- **合成数据 100 步门槛**（`tests/test_flat_synthetic.py`，12 个形状含 1/3/4096/12345 元素、4D channels_last、6 个 lr/wd 组、尖峰 scale-8 梯度、一个从不收到梯度的参数；对照 `round7_optimizer/impl/optimizer_ref.py` 的 fp32 oracle 与 `torch.optim.AdamW(fused=True)` + `clip_grad_norm_`；调度用真实的 `TwoStageWarmupPolySchedule`）：

| 指标（100 步最大值） | 对 oracle | 对 torch fused |
|---|---|---|
| 参数最大绝对差 | **2.4e-7**（≤1e-6，1 ulp） | 6.9e-6（见下） |
| m / v 最大绝对差 | – | 1.5e-11 / 8e-15（m、v 量级 1e-3 / 1e-6） |
| grad norm 相对差 | 1.4e-7 | 1.1e-7 |
| lr（设备调度 vs torch 调度器）相对差 | 5.7e-8（每组每步） | |

  参数对 torch fused 的 6.9e-6 差异来自 torch fused kernel 自身（`max|p|=4.66`，权重衰减因子 `1−lr·wd≈1−1e-5` 在 fp32 里取整：fused 与 oracle 之间同样是 6.9e-6，`fused_vs_ref=6.9e-6`）；r15 按 torch 非 fused 路径的 double 标量公式，与 oracle 相差 1 ulp。判据写成 `param_vs_ref ≤ 1e-6` 且 `param_vs_fused ≤ 1.25×fused_vs_ref + 1e-6`，通过。
  - 无梯度参数不变（与 torch 一样不进 state）；梯度缓冲在每步后全零；`.data/.grad` 是 flat 缓冲的视图；两次相同运行 **逐位相同**（确定性）。
  - checkpoint 往返：flat 的 `state_dict()` 与 torch AdamW 同键同形（`state: {idx: {step, exp_avg, exp_avg_sq}}`，`param_groups` 键集合相同），加载到 torch AdamW(fused) 后状态逐位相同；torch 的 `state_dict` 加载到 flat 后 m/v 逐位相同、计数器正确；各自续跑 10 步与原轨迹相差 2.4e-7 / 1.9e-6（后者含 torch fused 的 ulp 级差异）。日志：`profile/synthetic_gate.json`。
- **真实一步（门 B，`check.py` 口径）**：`run.py --stack P1,r15 --ref P1 --action gate_b --seeds 0 1 2`，同权重同 batch，ViT-B、ViT-L 各 3 个 seed：loss_rel = 0，grad_norm_rel ≤ 3.5e-6，全局 grad 余弦 ≥ 0.9999997，逐参数 grad 余弦最小 0.9986–0.9999（最差的是 `backbone.storage_tokens` / `blocks.1.norm*.weight`，来自 backward 自身的非确定性，与优化器无关），全部通过 `check.py` 阈值。**但这个门对优化器本身是空的**：训练第 0 步所有组的 warmup lr = 0（`TwoStageWarmupPoly`，非 ViT 组 `base·0/500`，ViT 组 0），两个栈的更新都恰好为 0（`update_rel_l2 = update_maxabs_diff = 0.0`，update 余弦“1.0”是 0/0 的退化值），所以它只验证了前向/反向梯度没变。
- **真实梯度 + lr>0 的一步（`tests/real_step_lr.py`，补上述空洞）**：把 flat 优化器的设备计数器拨到调度器第 1500 步（两段 warmup 之后，每组 lr>0：ViT-B 8.0e-6–9.4e-5，ViT-L 3.9e-7–6.6e-5），用真实 batch 跑一个 eager 步（P1,r15），在 `opt_step` 处截取真实梯度，再用 torch `AdamW(fused=True)` + `clip_grad_norm_(0.01)` 在同一权重、同一梯度、零 Adam 状态上做同样一步（每组 lr 取真实调度器）：

| 模型 | grad norm（flat / torch） | 更新范数（flat / torch） | 更新相对 L2 差 | 更新余弦（全局 / 逐参数最小） | 参数最大绝对差 |
|---|---|---|---|---|---|
| ViT-B | 188.2521 / 188.2521（相对差 0） | 0.365063 / 0.365063 | 6.3e-5 | 0.9999999980 / 0.999998 | 1.2e-7 |
| ViT-L | 173.7520 / 173.7520（相对差 8.8e-8） | 0.319175 / 0.319175 | 1.4e-4 | 0.9999999907 / 0.99931 | 1.2e-7 |

  判据 `update_cos_global ≥ 0.9999 且 update_rel_l2 ≤ 1e-3 且 grad_norm_rel ≤ 1e-5` 均通过（`profile/real_step_lr_{vitb16,vitl16}.json`）。第一个 Adam 步的更新≈ ±lr·sign(g)，对梯度中接近 eps 的小元素的符号敏感，所以逐参数最小余弦（ViT-L 0.9993）比全局低；差异来自 fp32 的 `lr/bc1` 与 torch fused 的算术顺序，不是系统性误差。
- **真实模型兼容检查**（`tests/check_real_flat.py`，P2,r15，4 个 eager 步）：ViT-B 193 个参数全部有梯度（无冻结参数）；`.data/.grad` 视图正确；设备计数器 (4,4) 与 host scheduler `last_epoch=4` 一致；设备 lr 与调度器 `get_lr()` 最大相对差 7e-9；`optimizer.state_dict()` 193 条状态、127 个 param_group，可被 plain torch AdamW 加载；`load_state_dict` 往返 m 逐位相同。ViT-L 同一检查（`profile/check_real_flat_vitl16.json`）：361 个参数全部有梯度（无冻结），视图正确，设备计数器 (4,4) 与 host `last_epoch=4` 一致，设备 lr 与 `get_lr()` 最大相对差 1.5e-8，`state_dict()` 361 条状态 / 235 个 param_group，可被 plain torch AdamW 加载（361 条），`load_state_dict` 往返 m 逐位相同。ViT-B 的检查输出只打印到了控制台（当时脚本还没写 json）；结果如上。

## 6. 静态分析

`cuobjdump --dump-resource-usage`（`profile/static/resource_usage.txt`）：`update_kernel` 89 个寄存器、无 spill、无 smem（每 SM 2 个 256 线程块，每线程 16 条 128 位 load 在飞）；`sumsq_prep_kernel` 73 个寄存器、1121 B smem。SASS 里 `LDG.E.128` 20 条、`STG.E.128` 16 条，全部向量化（`profile/static/flat_adamw.sass`）。`ld/st.cs` 流式提示没有收益（K2 1516 → 1522 GB/s，`tests/bench_kernels.py`）。DRAM 上限实测（`tests/bench_bw.py`）：copy 1526 GB/s、写 1671 GB/s、读归约 1683 GB/s、`add_`（2 读 1 写）1575 GB/s。

## 7. 结论、失败的尝试、下一个瓶颈

**结论**
- 做成了：flat 缓冲 + 两个 kernel 的 clip+AdamW，设备上的 lr 调度和计数器，torch AdamW 格式的 checkpoint，确定性；合成 100 步门、真实梯度 lr>0 的一步（§5）和 checkpoint 往返全过（`check.py` 口径的第 0 步门 B 因 lr=0 而对更新为空，见 §5）。整步（eager）ViT-L 51.68 → 56.00 img/s（1.084×），ViT-B 109.92 → 116.57（1.060×）。同时是 r16 整步 graph 的前提（捕获时没有 host 标量拷贝）。
- 优化器 GPU kernel 时间 ViT-L 10.7 → 7.4 ms、ViT-B 3.8 → 2.2 ms（nsys，kernel 数 258 → 2）；host 时间（NVTX CPU 区间）ViT-L 27.8 → 6.1 ms、ViT-B 34.3 → 3.6 ms。目标 ≤ 6–7 ms（ViT-L）**未达到（7.4 ms）**，已在实测可达带宽上。

**没做 / 失败**
- bf16 / MXFP8 权重写出器：只提供了 `FlatParams.register_weight_writer(fn)` 接口（每次更新后在当前 stream 调用，graph 内也会被捕获），接口未被任何栈使用，也没有单独测试；r13 的 bf16 影子仍在 block 0 刷新，r17 的权重量化仍是每步 ~1 ms 的独立 kernel。
- 流式 load/store 提示无效；没有尝试 2-chunk 流水线（带宽已与 `copy_` 持平）。

**限制**
- 第一个 eager 步接管梯度并识别没有梯度的参数（冻结）；之后若有参数在某步才第一次收到梯度，会被当作冻结。`load_state_dict(..., assign=True)` 会替换 Parameter 对象并破坏视图；不支持 AMSGrad / maximize / 非 TwoStageWarmupPoly 的调度器（后者需要 `set_external_lr`，不可捕获）；必须通过 `ops.opt_step` 驱动（直接调 `optimizer.step()` 不走 flat）。

**下一瓶颈 / 可继续的优化**
1. 梯度累加：`CUDAFunctor_add<float>` 418 次/步 1.52 ms（autograd 把 r13/r17 的 fp32 wgrad 加到已存在的 `.grad` 视图上）+ K2 里的清零写 0.96 ms ≈ 2.5 ms/步。若 wgrad 直接写入 flat 梯度（GEMM epilogue beta=1 / autograd 窃取 `p.grad=None` 的张量并改成指针表的优化器），K2 降到 28 B/元素，ViT-L 优化器可到约 6.7 ms 并省下累加 kernel（预计合计 −2.5 ms，约 2%，未实现）。
2. Σg² 融进 wgrad epilogue（r13 的原计划）可省 K1 的 0.76 ms 读。
3. 剩下的步时主要是 head 卷积 22 ms、MXFP8/bf16 GEMM、attention 16 ms（r14 / r12），见 round16 的 nsys 表。
