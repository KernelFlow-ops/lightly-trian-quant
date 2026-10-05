# Round9：低精度可行性（负责：Agent B，2026-10-02）

## 1. 目标
验证 sm_120 `_scaled_mm` 的 FP8 tensorwise/rowwise、MXFP8、NVFP4格式，并提供需要 C/D 门槛的训练 Linear 实验分支。

## 2. 改动
`impl/fp8.py`：E4M3 per-row激活/权重和 E4M3梯度，fwd/dgrad/wgrad 使用 `_scaled_mm`，wgrad bf16 accumulator 后转 fp32 master；current/delayed scale。`impl/patch.py` variants `fp8_mlp_cur`、`fp8_mlp_delayed`、`fp8_all`。`tests/probe_formats.py` 为注明来源的 layout 辅助。

后续补充 `impl/mxfp8.py` / `mxfp8_mlp`：Triton 1x32 block amax/二次幂量化，直接写 cuBLAS 要求的 UE8M0 swizzled scales；所有Linear矩阵及grad路径复用 `_scaled_mm`。小型137×128→256已证明scale布局、FP8字节、dequant参考输出逐位相同，backward finite。真实MLP形状和全部grad数值验证待窗口，未入主线。delayed history以Python index循环，已在ctx.meta标记noncapturable_reason。

## 3. 结果
`profile/formats.json`：tensorwise、rowwise、MXFP8 (E8M0 1x32)、NVFP4 均 supported/finite；前三者 dequant GEMM relative L2 约 0.166%。`correctness_real_shapes.json`：真实行数和四种 MLP K/N 的 fwd/全部 grad finite，E4M3 gradient cosine >=0.99929、output cosine约0.99965；尚未取得 C/D，故未进入主线。

初始 E5M2 grad cosine约0.99826，已记录失败并改 E4M3。sm_120 rowwise `_scaled_mm` 不接受 fp32 output，wgrad 使用 bf16 输出再 cast 到 fp32 master。NVFP4/MXFP8 尚未训练 autograd。

## 4. nsys / D 级
M0未完成，未运行 nsys；所有整步性能、C/D字段 null。任何FP8训练需 A 调度 300步和6 epoch×2 seeds。

## 5. 结论
FP8 保留实验分支，默认 bf16；低精度格式可行不等于训练收敛通过。
