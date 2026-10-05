# Round7：优化器与梯度裁剪（负责：Agent C，2026-10-02）

## 1. 目标与假设

- 对应 PLAN H10：减少 optimizer/clip 的 launch 数，并为 round10 CUDA Graph 提供 capturable opt_step。round0 mid nsys 已测 ViT-B14 opt CPU 180.081 ms/step、GPU busy 6.036 ms、946 GPU ops/step；ViT-L7 为 168.026 ms、12.866 ms、1750 ops/step。来源：`/root/autodl-tmp/train-opt-tmp/A/nsys/r0_vit{b14,l7}_mid_summary.md`。本轮收益仍需受控实测。

## 2. 改动

- `impl/optimizer_ref.py`：按 torch AdamW（decoupled weight decay、bias correction）和 `clip_grad_norm_(0.01)` 写 fp32 oracle。
- `impl/fused_optimizer.py`：foreach global-norm clipping，norm/scale 均保留在 device，避免 host scalar sync。`impl/patch.py`：提供 capturable_torch/capturable_shadow；每组 lr 保持 device scalar 且 data_ptr 稳定，图内跳过 scheduler，图外可调用保存的 scheduler hook；shadow writer 使用 no_grad foreach copy。checkpoint state_dict 导出 float lr、CPU step，并接受原版状态。
- `impl/cuda_adamw.cu` + `impl/cuda_adamw.py`：真实 CUDA C++ correctness-first kernel，device global norm、clip、AdamW update、device step/lr/wd 和可选 bf16 shadow 写回；编译缓存位于 `/root/autodl-tmp/train-opt-tmp/C/cuda_adamw_build`。当前为每参数张量一组 launch，后续再做 flat/multi-tensor 融合。

## 3. nsys 分析过程

- null：未进行 GPU 计时或 nsys；需要 round0 opt/clip 段的 launch、同步、显存数据。

## 4. 结果

| model | variant | opt ms | launch 数 | grad norm | 参数/状态误差 |
|---|---|---|---|---|---|
| fp32 synthetic | reference | null | null | 与 torch 一致 | 1/100 步 ≤1e-6 |

## 5. 正确性

- CPU oracle 与 `torch.optim.AdamW(foreach=False)` + `clip_grad_norm_` 100 步对照通过：参数、step、exp_avg、exp_avg_sq 和 grad norm 均在 `1e-6` 内。命令：`PYTHONPATH=train-opt /root/miniconda3/bin/python -m pytest -q train-opt/round7_optimizer/tests/test_optimizer_ref.py`；日志：`/root/autodl-tmp/train-opt-tmp/C/round7_cpu_optimizer.log`（1 passed）。
- CPU capturable_torch/capturable_shadow 协议、foreach clip 与常规 AdamW state_dict 加载测试通过（5 tests）：device scalar lr、scheduler 顺序、shadow 无 autograd、state_dict 导出格式；GPU fused kernel 与真实大模型 ckpt 仍待 A 的 GPU 窗口。
- CUDA extension 编译成功；257 元素 fp32 一步 smoke 与 Torch AdamW+clip 的参数最大绝对差 `2.3842e-7`，step=1，输出 finite。正式大模型 B/C、microbench、shadow kernel 仍待 GPU 窗口。

## 6. 静态分析

- CUDA C++：资源/SASS 静态分析、正式真实形状 microbench 尚未完成；当前仅记录编译与小形状 A smoke。

## 7. 结论、失败的尝试、下一个瓶颈

- capturable fallback 已交付并通过 CPU 协议门槛；未测 GPU 速度与完整 checkpoint roundtrip，相关字段保持 null。
