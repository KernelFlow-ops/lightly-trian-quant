# Round5：Block 融合（负责：Agent B，2026-10-02）

## 1. 目标
完成 fp32 residual + LayerScale + LN 的 fwd/bwd、exact-erf GELU，以及不改变 Parameter 的 bf16 shadow Linear。

## 2. 改动
`impl/fused_norm.py`、`impl/gelu.py`、`impl/shadow.py`、`impl/patch.py`；variants 为 `ln_fuse`、`gelu`、`shadow`、`all`。shadow 按 handoff regex 填充 `ctx.shadow`，由 post-opt hook 刷新并尊重 `meta["shadow_writer"]`。

## 3. nsys
M0 未完成，未运行 nsys；独立 CUDA event 使用双锁、warmup 20、50 iter×3。整步字段 null。

## 4. 独立 microbench（B=16 开发形状）
| kernel | ViT-L N=1029/1129 fwd+bwd us | ViT-B N=1029/1129 fwd+bwd us |
|---|---:|---:|
| torch residual+LN | 929.7 / 1075.2 | 660.7 / 738.6 |
| fused residual+LN | 494.9 / 608.4 | 566.8 / 569.7 |
| torch exact GELU | 433.7 / 476.4 | 325.1 / 357.2 |
| Triton exact GELU | 434.6 / 477.0 | 327.3 / 359.1 |

详见 `profile/microbench_B16.json`；GELU kernel略慢，默认不推荐。

## 5. 正确性
`profile/correctness_B16.json` 覆盖真实 M=B*N、ViT-L/B、N=1029/1129、尖峰残差，fwd/dx/db/dgamma/dweight/dbias cosine >=0.999998。`interface_acceptance.json` 真实 SelfAttention/LayerScale/Mlp block 与 r4 组合：key、shape、Parameter identity 保持，ln_fuse grad cos min >=0.9999998，all >=0.9999985；FP8 local block >=0.998369（不作为 full B）。官方 B_AUTO ViT-L7（r4+ln_fuse）全部阶段通过：`loss_rel=8.60546e-6`、`grad_cos_min=0.999977171`、`grad_norm_rel=4.36933e-5`、update cosine 1。阶段证据在 `profile/check_vitl16_B7_ln_fuse_{early,mid,late}.json`，日志在 `/root/autodl-tmp/train-opt-tmp/B/r5_vitl7_check.log`。ViT-B14 mid 保留此前通过结果，early/late 等待队列。

## 6. 结论
推荐 `r4 sdpa_rope + r5 ln_fuse` 进入 M0 后 B 级；GELU 收尾，不伪造 GEMM epilogue 收益。静态 summary 显示 fwd 48 regs/4 KiB，bwd 166 regs/8 KiB。

Fabric源路径更新：A核实原脚本实际前向为FP32；已改LN在无CUDA autocast时输出FP32，shadow在无autocast时回退master F.linear。CPU协议测试 `tests/test_shadow_protocol_cpu.py` 验证fp32 fallback与原Linear逐位相同、post-opt foreach更新、C优化器接管后B跳过写入、hook幂等和Parameter身份。GPU的FP32分支尚待正式A/B，不用旧bf16 microbench冒充FP32结果。
