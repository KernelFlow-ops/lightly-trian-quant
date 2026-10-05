# Round6：EoMT head 候选（负责：Agent C，2026-10-02）

## 1. 目标与假设

- round0 mid nsys 已测 `_predict` 四次：ViT-B14 每次 GPU busy 约 8.69 ms、合计约 34.8 ms；ViT-L7 每次约 6.48 ms、合计约 25.9 ms。来源：`/root/autodl-tmp/train-opt-tmp/A/nsys/r0_vit{b14,l7}_mid_summary.md`。DWConv backward 与布局拷贝的独立占比尚待细分，当前不估算节省 ms/step。

## 2. 改动

- `impl/fused_head.py`：保留原参数对象，提供训练可反传的 `bmm`、`channels_last_bmm`、`convt_gemm`、`dwln`、`all` 候选。`convt_gemm` 将 k2s2 ConvTranspose2d 改成 GEMM + 像素重排；`dwln/all` 使用 channels-last 和显式 depthwise + LayerNorm 路径。
- `impl/dwln.py`：训练安全的 depthwise + LayerNorm 组件，支持 CUDA channels-last 与 CPU 正确性回退。
- `impl/patch.py`：只覆盖 `ops.predict`；输出 mask logits 为连续 `(B,Q,128,128)`、class logits 为 `(B,Q,K+1)`；REQUIRES/CONFLICTS 均为空。

## 3. nsys 分析过程

- null：round0 尚未发布 head 分段数据，且本窗口 GPU 被其他轮次占用，未对候选计时或采集 nsys。进入 M1 后按双锁规范采 early/mid/late 与 head 单算子 fwd+bwd。

## 4. 结果

| model | variant | early/mid/late step ms | weighted img/s | head fwd/bwd ms | peak mem |
|---|---|---|---|---|---|
| ViT-L | bmm / channels_last_bmm / convt_gemm / dwln / all | null | null | null | null |
| ViT-B | bmm / channels_last_bmm / convt_gemm / dwln / all | null | null | null | null |

## 5. 正确性

- CPU 小形状 ConvT GEMM、完整 head 的输入/全参数梯度，以及 `dwln/all` 变体均与原版对照通过；patch 幂等，state_dict 键/形状不变。命令：`PYTHONPATH=train-opt /root/miniconda3/bin/python -m pytest -q train-opt/round6_head_fusion/tests`；结果：8 passed。
- 实际 ViT-L/B 形状、B/C/D 级：null。

## 6. 静态分析

- 当前仅 PyTorch 算子，无自定义 CUDA/Triton kernel；寄存器、spill、SASS 暂不适用。自定义 C6.2 kernel 未在本轮实现，因此不声称静态或速度收益。

## 7. 结论、失败的尝试、下一个瓶颈

- 当前为训练正确性候选，性能字段保持 null。下一步由 A 在 GPU 队列空闲后决定是否进行正式形状 A/B 检查。
