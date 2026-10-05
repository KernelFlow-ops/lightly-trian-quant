# Round3：Loss 与 Hungarian matcher（负责：Agent C，2026-10-02）

## 1. 目标与假设

- 对应 PLAN H1：原版每层每张图把 cost 拷回 CPU，经 SciPy 匹配；4 层 batch B 时会产生约 `4B` 次 GPU→CPU 同步。round0 mid nsys 已测 ViT-B14 loss CPU 74.881 ms/step、GPU busy 8.727 ms/step、3647 GPU ops/step；ViT-L7 为 45.168 / 4.664 ms、2051 ops/step。来源：`/root/autodl-tmp/train-opt-tmp/A/nsys/r0_vit{b14,l7}_mid_summary.md`；nsys 带 backtrace 的 CPU 时间不与无 profiler 的 step 直接比较，本轮可节省量仍待受控测量。
- 保留原版随机调用次序、双线性 target 点采样、逐层归一化和 loss 键名。

## 2. 改动

- `impl/ref_criterion.py`：在不改变原版计算顺序的前提下注入 matcher、过采样和随机点坐标。
- `impl/matcher.py`：T≤4 时按每列 top-T 候选批量枚举精确 assignment；T>4 一次批量回退 SciPy。
- `impl/fast_criterion.py`：4 层批量 matcher cost、CE、mask/dice loss，`batched_torch` 和 `fused` 变体。按 SciPy 的 query 升序组织 mask loss 的匹配对；另外支持 r10 的 `masks/labels/valid` 静态补齐输入。
- `impl/point_loss.py`：Triton 双线性 logits/target 点采样，BCE/dice 的 fp32 局部规约与 logits 反向散写。
- `impl/patch.py`：只覆盖 `ops.loss_all_layers`，不增加模型参数或 state_dict 项；REQUIRES/CONFLICTS 均为空。
- `impl/cuda_ops.cpp` + `impl/cuda_ops.cu`：真实 CUDA C++ point loss（bilinear BCE+dice fwd/bwd，atomicAdd backward）和 T≤4 exact matcher；`impl/cuda_extension.py` 使用 `torch.utils.cpp_extension.load` 懒构建到 `/root/autodl-tmp/train-opt-tmp/C/cuda_build`。`variant="cuda"` 接入 `FastMaskClassificationLoss`，matcher 在 CUDA float32 cost 上自动使用 device kernel。
- `run.py`、`check_step.py`：双锁内的合成 microbench/nsys 入口和真实 batch 的 B 级入口。

## 3. nsys 分析过程

- **本轮 nsys 待测**：M0 尚未发布，且 2026-10-02 00:24 Asia/Shanghai 检查时外部 infer-opt PID112295 占用 GPU；首次 vitb14 B 级在模型构建前主动取消。round0 nsys 的 mid baseline 见上一节，不能当成本轮 variant 的分析结论。
- **round0 时间线观察**：从 `/root/autodl-tmp/train-opt-tmp/A/nsys/r0_vitb14_mid_analysis.json` 的 `sqlite.syncs` 按 NVTX `loss` 统计，ViT-B14 每步 196 次同步相关调用（140 次 `cudaStreamSynchronize`、56 次 `cudaMemcpyAsync`）；ViT-L7 的对应分析为每步 112 次（84+28）。56/28 正好是各模型的 `4B` 个 matcher 子问题数；ncu 不可用且该分析的 Python backtrace 为 null，具体调用行以源码匹配为推断。待本轮 `batched_torch` nsys 受控对照确认同步减少量和空泡归因。
- 待 GPU 空闲且 M0 发布后，用 `common/profile_nsys.sh` 对原版、`batched_torch`、`fused` 各采 early/mid/late；从同步 backtrace、loss NVTX、kernel 数和忙碌率定位差值。

## 4. 结果

| model | variant | early/mid/late step ms | weighted img/s | vs R0 | loss GPU fwd/bwd ms | syncs/step |
|---|---|---|---|---|---|---|
| ViT-L | batched_torch | null | null | null | null | null |
| ViT-L | fused | null | null | null | null | null |
| ViT-B | batched_torch | null | null | null | null | null |
| ViT-B | fused | null | null | null | null | null |

## 5. 正确性

- `tests/test_ref_criterion.py`：固定 seed 下与原 `MaskClassificationLoss` 的 loss、assignment、mask/class logits 梯度逐位相同。
- `tests/test_matcher.py`：10,000 个随机/含 tie 的 Q100、T0~4 cost 子问题与 SciPy 最优总 cost 一致；T>4 回退下标一致。
- `tests/test_point_loss.py`：fp32/bf16、257 点尾块、12,544 点 scale-8 输入的前向和 logits 梯度通过阈值。
- `tests/test_patch.py`：默认 ops 的 16 个命名随机调用点 record/replay 完整；4 层小形状的 loss 和梯度通过，patch 幂等、state_dict 键/形状不变；T=0/T=5 与全空 targets 边界通过；静态补齐输入在注入相同坐标后与 list 格式的 loss/全部 logits 梯度一致。命令：`PYTHONPATH=train-opt /root/miniconda3/bin/python -m pytest -q train-opt/round3_loss_matcher/tests/test_patch.py`；日志：`/root/autodl-tmp/train-opt-tmp/C/round3_cpu_patch.log`（5 passed）。
- B14 `batched_torch` quick step 诊断（预热/迭代少于正式 50×3 口径）测得 early/mid/late 331.86/333.23/333.45 ms，loss 约 5.4 ms，约 42.04 img/s；结果写在 `profile/step_vitb16_batched_torch.json`，只能作为诊断，未写入正式性能主表。`fused` 标准 step 在当前显存下反向 OOM，已如实记录。
- `tests/test_fast_criterion.py`：B2/Q100/128² logits/512² target/P12544、4 层 bf16 对照通过；12 个匹配 mask 的最终采样点与参考逐位相同，8 组 logits 梯度 cosine 为 0.99999994~1.0。原训练实际 logits 为 fp32，已补 fp32 测试，待空闲 GPU 复测。
- **B 级首轮 fail**：vitb14、default fp32、mid、`batched_torch`，`loss_rel=0.000203786`、`grad_cos_min=0.876278`、`grad_norm_rel=0.000781724`、`update_cos_min=1.0`；逐参数最差 `backbone.blocks.1.norm1.weight`。命令：`/root/miniconda3/bin/python train-opt/round3_loss_matcher/check_step.py --model vitb16 --variant batched_torch --phase mid --output train-opt/round3_loss_matcher/profile/check_vitb16_batched_torch_mid.json`；详细日志/结果：`profile/check_vitb16_batched_torch_mid.json`。尚未进入主线；C/D 级为 null。
- **同 logits 诊断**：命令 `/root/miniconda3/bin/python train-opt/round3_loss_matcher/diag_assignment.py --model vitb16 --output train-opt/round3_loss_matcher/profile/diag_assignment_vitb16.json`；真实 vitb14 的 assignment 和最终采样点全相同，12 个 loss key 相对误差 ≤9.34e-8，四层 mask logits grad cosine 全 1.0、class logits grad cosine ≥0.99999994。结果：`profile/diag_assignment_vitb16.json`。
- **default/default 控制组**：命令 `/root/miniconda3/bin/python train-opt/round3_loss_matcher/check_step.py --model vitb16 --variant baseline_control --phase mid --output train-opt/round3_loss_matcher/profile/check_vitb16_baseline_control_mid.json`；`loss_rel=0.000376270`、`grad_cos_min=0.992581`、`grad_norm_rel=0.000168`、`update_cos_min=1.0`，pass 但非逐位一致。结果：`profile/check_vitb16_baseline_control_mid.json`。
- **RngTape 修正后正式 B 级 pass**：命令 `/root/miniconda3/bin/python train-opt/round3_loss_matcher/check_step.py --model vitb16 --variant batched_torch --phase mid --output train-opt/round3_loss_matcher/profile/check_vitb16_batched_torch_mid_replayfixed.json`；vitb14、default fp32、mid 得 `loss_rel=0`、`grad_cos_min=0.999999344`、`grad_norm_rel=9.22e-8`、`update_cos_min=1.0`。结果：`profile/check_vitb16_batched_torch_mid_replayfixed.json`；终端日志：`/root/autodl-tmp/train-opt-tmp/C/r3_vitb14_mid_replayfixed.log`；`results.json` 已记录。
- **阶段补测**：命令 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/miniconda3/bin/python train-opt/round3_loss_matcher/check_batch.py --variant batched_torch --cases vitb16:early vitb16:late vitl16:mid`。ViT-B14 early/late、ViT-L7 mid 均 `loss_rel=0`、`update_cos_min=1`，梯度 cosine 最小值依次为 `0.999999046 / 0.999999702 / 0.999999523`；详细 JSON：`profile/check_vitb16_batched_torch_early.json`、`profile/check_vitb16_batched_torch_late.json`、`profile/check_vitl16_batched_torch_mid.json`；终端日志：`/root/autodl-tmp/train-opt-tmp/C/r3_<model>_batched_torch_<phase>.log`。`results.json` 按已测阶段最差值合并，未测性能字段保持 null。

## 6. 静态分析

- Triton point-loss kernel 的寄存器、spill、smem、SASS 和 occupancy 待空闲 GPU 编译后导出；未测字段为 null。
- CUDA C++ extension 于 2026-10-02 使用 `/usr/local/cuda/bin/nvcc`、`TORCH_CUDA_ARCH_LIST=12.0`、`MAX_JOBS=6` 编译成功；构建缓存：`/root/autodl-tmp/train-opt-tmp/C/cuda_build_v2/train_opt_round3_cuda.so`。本次 kernel 仍未做 nsys/ncu 静态寄存器报告，相关字段保持 null。

## 7. CUDA C++ A 级实测

- 命令：`PYTHONPATH=train-opt TRAIN_OPT_CUDA_BUILD_DIR=/root/autodl-tmp/train-opt-tmp/C/cuda_build_v2 /root/miniconda3/bin/python -m pytest -q train-opt/round3_loss_matcher/tests/test_cuda_ops.py`
- 结果：`5 passed in 2.11s`。point loss 覆盖 fp32、scale=8、257 点、尾部越界坐标，前向误差 ≤`2.4e-7`，梯度最大绝对误差 ≤`4.7e-9`，cosine=`1.0`；matcher T=1/2/3/4 各 32 个 Q=100 矩阵与 SciPy 总 cost 相等。
- 真实 ViT-B/L 训练步、300 步曲线、microbench、nsys 及 C/D 级尚未测，不能据此宣称性能收益或训练收敛通过。

## 8. SM120 高性能 CUDA 版本（2026-10-02）

- `impl/cuda_ops.cu` 已升级为 SM120 专用路径：point-loss forward 使用 `float2` 坐标向量加载、`__restrict__`、`__launch_bounds__(256,2)`、`__shfl_down_sync` warp 规约，移除每点对 shared accumulator 的 atomicAdd；backward 对生产 128x128 map 使用每 block shared gradient accumulation，最后一次性 coalesced 写回，超出 96 KiB shared carveout 时回退 global atomic kernel。
- matcher 使用 warp-per-problem：32 lanes 各自扫描 query，寄存器保留每列 top-T，再以 4 KiB shared candidate 合并并在 lane 0 执行 T^T 精确枚举；T≤4 语义、无效 target 和 tie 已与 SciPy 检查。
- 编译命令：`/usr/local/cuda/bin/nvcc -std=c++20 -O3 --use_fast_math -arch=sm_120 --ptxas-options=-v -lineinfo ... -c impl/cuda_ops.cu`；sm_120 ptxas 资源：forward fp32 50 regs/128 B smem，bf16 51 regs/128 B，shared-backward 31 regs + dynamic smem，global-backward 38 regs，matcher 54 regs/4096 B smem；无 spill。完整输出：`/root/autodl-tmp/train-opt-tmp/C/static_perf/ptxas.log`。环境未提供 `cuobjdump`/`ncu`，因此没有伪造 SASS/occupancy 字段。
- 真实形状 point-loss microbench（P=12,544,H=W=128，warmup=20,iters=100，CUDA events）：B14 新 CUDA C++ `0.1891 ms` vs 旧 CUDA C++ `0.5359 ms`（2.83x，+183.3%）；L7 新 `0.3830 ms` vs 旧 `0.5296 ms`（1.38x，+38.2%）。同一新版本 vs Triton fused：B14 `0.1891` vs `0.2879 ms`（1.52x），L7 `0.3830` vs `0.6457 ms`（1.69x）。该数字只代表 point-loss fwd+bwd microbench，不代表整步吞吐。
- A 级回归：`PYTHONPATH=train-opt TRAIN_OPT_CUDA_BUILD_DIR=/root/autodl-tmp/train-opt-tmp/C/cuda_build_perf2 /root/miniconda3/bin/python -m pytest -q train-opt/round3_loss_matcher/tests/test_cuda_ops.py`，`5 passed in 1.97s`；另测 T=1..4、Q=5/31/100、随机 valid（17 problems×20 seeds）与 SciPy，`invalid matcher ok`。
- 真实 ViT-B14 整步 CUDA variant 在本次双锁窗口因 31.3 GiB/31.36 GiB 显存余量不足申请 672 MiB 而退出；ViT-L7 同样因共享窗口显存不足退出。整步加速、300 步/C/D 与 nsys 仍为 null，不能由 microbench 外推。

## 7. 结论、失败的尝试、下一个瓶颈

- 首版批量 loss 按 target 列顺序分配随机点，原版 SciPy 按 query 行升序；小形状偶然通过，但真实尺寸 mask grad cosine 约 0.494。修正匹配对排序后，真实尺寸 bf16 对照通过。
- 同 logits 的 round3 路径已排除；整步差异来自两次前向的随机流。RngTape 原 replay 分支不执行 `draw()`，而 record 消耗随机数；mid 的 `mask/j1` 发生在后续 RoPE 之前。A 修正 replay 的 RNG 消费后，ViT-B 三阶段、ViT-L mid 均通过。下一步测 ViT-L 其余阶段与 fused，然后做正式性能分析。
- 下一关是 fp32 正式 batch 的 B 级、原版与两变体的双锁 microbench/nsys，以及 C/D 级排队。低收益或失败变体按计划如实收尾。
