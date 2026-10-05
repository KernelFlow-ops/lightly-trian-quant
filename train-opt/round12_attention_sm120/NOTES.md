# Round12：sm_120 Attention（Agent B，2026-10-02/03）— 进行中版本（随实测更新）

## 1. 目标与假设
- 目标：DINOv3-EoMT 训练 attention（fwd+bwd）在 RTX 5090 上最快实现，对手是 Agent A 的 T-BF16 栈 TB1 的 attention 路径
  （`rope_triton + plain_flash + joint_qsplit + joint_cudnn`，见 `round11_bf16_mixed/impl/attn.py`）。
- 形状：ViT-L B=7 H=16、ViT-B B=14 H=12，D=64，plain N=1029（5 个前缀 token + 1024 patch），joint N=1129（100 query + cls + 4 reg + 1024 patch）。
- 假设（已被实测支持）：
  1. BF16 HMMA 在 5090 上被限速（约 225 TFLOPS 实测上限），flash/cuDNN/Spark 在**填充后的工作量**上都已接近上限；
     N=1029 在 128 行 tile 下浪费约 25% FLOPs（1152 vs 1029），所以收益来自：尾块精确处理、融合 RoPE、packed 布局、mask 不走 dense bool。
  2. joint 块的 query mask 只影响 q-tile 0（行 0..99），用 `QueryBitmask u32[B,100,32]` 在 S 累加器上按位应用，比 dense `[B,N,N]` bool + mem-efficient 快数倍。

## 2. 改动
- `impl/attn_fwd.cu`：前向。8 个 consumer warp x 16 行 = 128 行 Q tile，64-key K/V tile 经 TMA（4 级 mbarrier 环），
  独立的 TMA producer warp 和 RoPE rotator warp（K tile 落地后在 smem 里原地旋转，fp32 数学，位级等同 `rope_apply`），Q 在寄存器中旋转；
  key 循环精确覆盖 N（尾 tile 只做 `ceil(valid/8)` 个 n8 / k16 步），行全在 N 之外的 warp 直接退出；
  joint：q-tile 0 的 warp 按 32-key 窗口用 funnel shift 从 bitmask 取位。保存 lse（log2 域）以及旋转后的 `qk_rot [B,N,2,H,64]`（供反向）。
- `impl/attn_bwd.cu`：反向（FA2 式，Spark attention_bwd v3 改写）。`bwd_pre`（Dv=dO·O，lse 转换，dQ 累加器清零）、
  `bwd_kv`（128 key/CTA，8 warp、无辅助 warp、255 寄存器、无溢出，3 级 TMA Q|dO|lse 环，V 在寄存器，dS^T 双缓冲 + mbarrier 交接，
  dQ 用 fp32 向量原子累加，dK 在寄存器里做逆 RoPE，dK/dV 直接写 packed `dqkv`）、`bwd_dq`（缩放 + 逆 RoPE + bf16 写入 q 槽）。
  query 循环同样精确覆盖 N（尾 tile 5/41 行只做需要的 n8/k16 步，用模板特化避免热循环里的分支）。
- `impl/attn_misc.cu`：mask 构建 kernel（128→32 双线性即 2x2 平均，>0，随机行禁用；fp32/bf16 logits）。
- `impl/bitmask.py`：`QueryBitmask`；`ops.build_mask` 覆盖：host 模式（RNG 消耗与源码/r1 一致：仅当 host prob<1 才抽 `mask/j{j}` (B,100)；host prob==0 时丢弃 mask 如 r1 mask_skip）、`devprob` 模式（相位无关，始终抽随机数、设备端读 prob）。
- `impl/fused.py`：`autograd.Function` + `ops.attn_core`；dense bool / fp32 / 不支持形状回退到 `dense_attn_core`（或之前安装的 attn_core）。
- `impl/patch.py`：variants `fused`、`devprob`、`maskdense`。`impl/third_party/spark_common.cuh`：改编自 Spark（MIT，commit a401bfa9e363）。
- 失败/弃用的设计：FA4-sm120 反向不支持 mask_mod / block sparsity（interface.py 断言），cuDNN DSL 反向的 bias 必须是稠密 `[1|B,H,N,N]`（joint 约 285 MB + dBias fp32 570 MB），都不适合紧凑 query mask。

## 3. nsys 分析过程
（待填：命令 `common/profile_nsys.sh`，before = TB1，after = R12，attention 在整步里的占比。）

## 4. 结果
（待填：候选矩阵、整步表。）

## 5. 正确性
- A 门（fp64 参考）：fwd cos 0.999994（普通）/ 0.99998（尖峰 scale-8，对 fp64；对"rope 取整到 bf16 的参考"是 0.9999993）；
  bwd cos 0.999996（普通）/ 0.9997（尖峰 scale-8 对纯 fp64，rope 的 bf16 取整是主因）。mask：all / none / random / rowmasked 全部通过。
- 位掩码 vs 源码 dense mask：fp32/bf16 logits、含精确抵消和 1e-3 量级，全部逐位相等（`profile/test_mask.json`）。RNG 消耗契约已测。
- B / S 门：待填。

## 6. 静态分析
（待填：ptxas 寄存器/溢出/smem，SASS 指令计数。）

## 7. 结论
（待填）
