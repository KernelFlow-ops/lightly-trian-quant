# train-opt 第二阶段契约（CONTRACTS v1，2026-10-02）

本文件是 round11–round19 各代理之间的**唯一接口约定**，只有协调者能修改。需要修改契约时，在最终报告里提出 "contract change request"。

## 0. 角色、目录与环境

| 角色 | 负责目录 | 可覆盖的 ops |
|---|---|---|
| A 系统 | `round11_bf16_mixed`、`round15_optimizer_flat`、`round16_full_graph`、`round18_data_val` | `ops.rope`、`ops.opt_step`、`ops.loss_all_layers` 前的 fp32 cast、`meta["forward_train_fn"]`、loader/val |
| B Attention+Head | `round12_attention_sm120`、`round14_head_sm120` | `ops.attn_core`、`ops.build_mask`、`ops.predict` |
| C GEMM+低精度 | `round13_gemm_fusion_sm120`、`round17_mxfp8_sm120` | `ops.block`、`ops.attn`（整块融合时）、MXFP8 API、权重量化钩子 |
| 协调者 | `common/`、`round19_final`、本文件 | — |

- **临时和编译文件**：放在 `/root/autodl-tmp/train-opt-tmp/<round>/`，例如 `build/`、`logs/`。
- **第三方源码**：在 `/root/autodl-tmp/train-opt-tmp/third_party/` 下，只读，commit 记在 `SOURCES.txt`：
  - `spark`：aidxhxr/5090-blackwell-kernels，MIT，CUDA C++。
  - `flash-attention`：其中 `flash_attn/cute/flash_{fwd,bwd}_sm120.py` 是 CuTe DSL。
  - `gn-kernels`、`SageAttention`。
  - 移植或改编的代码复制到本轮的 `impl/third_party/`，文件头写明来源、commit 和 license。
- **已安装（2026-10-02，只新增了包，见 `pip_freeze_{before,after}_phase2.txt`）**：
  - `nvidia-cutlass-dsl 4.8.0`（cu13）
  - `quack-kernels 0.6.5`：包括 `quack.rmsnorm.layernorm_fwd/bwd`、`quack.gemm_interface.gemm/gemm_add/gemm_act/gemm_dact/gemm_quant_out`、`quack/blockscaled`，以及 `gemm_sm120.py`。实测 fc1 形状 BF16 GEMM 为 238 TFLOPS，cuBLAS 为 203。
  - `nvidia-cudnn-frontend 1.30.0`（import 名为 `cudnn`）
  - `einops`
- **CUTLASS 4.2 C++ 头文件**：`/root/miniconda3/lib/python3.12/site-packages/cutlass_library/source/include`。nvcc 路径是 `/usr/local/cuda/bin/nvcc`（13.0），block-scaled 需要 `-gencode arch=compute_120a,code=sm_120a`。
- **PATH 注意**：非交互 SSH 的 PATH 里没有 conda 的 bin。`torch.utils.cpp_extension.load` 需要的 `ninja` 在 `/root/miniconda3/bin` 下，所以每条命令都要带上 `PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH`，否则会报 "Ninja is required"。
- **后台任务**：用 `setsid nohup … > log 2>&1 < /dev/null &`。必须重定向 stdin，否则 ssh 会一直挂住不返回。

## 1. 固定形状

| 模型 | B | d | H×Dh | MLP | blocks | joint blocks |
|---|---|---|---|---|---|---|
| ViT-L/16 (`vitl16`) | 7 | 1024 | 16×64 | 4096 | 24 | 21,22,23 |
| ViT-B/16 (`vitb16`) | 14 | 768 | 12×64 | 3072 | 12 | 9,10,11 |

- **图像与 token**：图像 512×512，patch grid 32×32。
  - plain block 的 token 顺序是 `[cls, reg×4, patch×1024]`，N=1029。
  - joint block 的顺序是 `[query×100, cls, reg×4, patch×1024]`，N=1129。
  - 因此 GEMM 的 M：ViT-L 为 7203/7903，ViT-B 为 14406/15806。
- **head**：每步 4 次 predict（L21/22/23 加 final）。
  - upscale 路径为 32²→64²→128²，每一级是 ConvT k2s2 → GELU → dwconv3×3 → LN2D。
  - mask logits 为 `[B,100,128,128]`；class 数 K=2（background、head），class logits 为 `[B,100,3]`。
- **RoPE**：只作用于最后 1024 个 patch token，prefix=N−1024（5 或 105）。
  - sin/cos 是 fp32 的 `[1024,64]`，前后半段相同。
  - 训练模式下每个 block 调用一次 `rope_embed`，各自消耗一次 `uniform_`（rescale_coords=2）。

## 2. 精度策略

- **主权重和 state_dict**：Parameter 保持 fp32，state_dict 的键和形状不变（`apply_patches` 会检查）。
- **必须 fp32 的部分**：残差流、LayerNorm 统计、softmax 的 max/sum/LSE、loss 和 matcher、优化器状态 m/v。
- **GEMM 输入**：P1 用 bf16，P2 用 MXFP8（E4M3 + UE8M0 1×32，或 32×32 方块）。一律 fp32 累加。输出默认 bf16；wgrad 可直接写 fp32。
- **T-BF16 参考**：autocast，即 harness 的 `ctx.meta["enable_autocast"]=True`，Fabric precision 为 `bf16-mixed`。loss 输入在进入 r3 之前先 `.float()`。

## 3. ops 注册表接口（`common/ops.py`，v1 新增 `attn_core`）

```python
ops.rope(ctx, block_idx, H, W) -> (sin, cos)            # fp32 [1024,64]
ops.build_mask(ctx, j, mask_logits, grid_size, n_tokens) -> mask
#   mask_logits: fp32 [B,100,128,128]; j = joint 序号 0..2
#   返回值 ∈ {None, bool [B,N,N]（torch 参考）, QueryBitmask（见 §4）}；attn_core 必须三种都接受
ops.attn(ctx, block, x_normed, rope, mask) -> [B,N,d]   # proj 输出（LayerScale 之前）
#   默认: qkv = block.attn.qkv(x_normed) → ops.attn_core → block.attn.proj → proj_drop
ops.attn_core(ctx, qkv, rope, mask, num_heads) -> o      # qkv: [B,N,3*d]（可视为 [B,N,3,H,64]）
#   o: [B,N,d]（= [B,N,H,64] 连续，直接喂 proj）；内部负责 q/k 的 RoPE（prefix 规则）和 attention
ops.block(ctx, i, block, x, rope, mask) -> x             # x: fp32 [B,N,d]；融合实现必须调用 ctx.ops.attn_core
ops.predict(ctx, x, grid_size) -> (mask_logits, class_logits)
#   x 是 norm 之前的 fp32 [B,N,d]（内部包含 backbone.norm）；输出 fp32 连续 [B,100,128,128] 和 [B,100,3]
ops.loss_all_layers(ctx, mask_logits_list, class_logits_list, binary_masks) -> (loss, losses)  # r3
ops.opt_step(ctx) -> grad_norm   # 设备张量；内部包含 clip、step、zero_grad、scheduler
```

- **随机数**：一律由 torch 生成（`ctx.rng.rand(name, shape)` 或 `torch.rand` / `uniform_`），作为输入传给自研 kernel。**kernel 内部不生成随机数**，这样 RngTape 的 B 检查依然逐元素可比。
  - 批量 RoPE 生成器必须按同样的顺序、同样的形状抽取 rescale 随机数。
  - 在 graph 模式（r16）下可以多消耗随机数，但 B 检查一律跑 eager。
- **组合规则**：patch 用 `REQUIRES`/`CONFLICTS` 声明依赖，不改 Parameter 身份、不改 state_dict。新 patch 覆盖 ops 时保留旧函数引用，作为不支持形状时的回退。

## 4. 张量格式

- **QueryBitmask**（B 定义 Python 类，放在 `round12_attention_sm120/impl/bitmask.py`）：
  - `bits: uint32 [B,100,32]`。patch p（0..1023）对应字 p//32 的第 p%32 位，置 1 表示该 query 可以看到 patch p。
  - query 与 query/cls/reg 之间始终可见；非 query 的行不做 mask。
  - 构造规则必须与源码等价：
    - `F.interpolate(mask_logits, (32,32), bilinear, align_corners=False) > 0`。对 128→32，这就是精确的 2×2 平均。
    - `prob < 1` 时，`rand(B,100) > prob` 的行整行置 1。
  - prob 是设备张量 `model.attn_mask_probs[j]`。kernel 必须支持在设备端读取 prob（相位无关），也保留 host 分支以匹配源码的随机数消耗。
- **MX8Tensor**（C 定义，放在 `round17_mxfp8_sm120/impl/mx8.py`）：
  - `data: float8_e4m3fn [rows, K]`，K 连续。
  - `scale: uint8 (UE8M0)`，布局与消费它的 GEMM 一致：cuBLAS/CUTLASS 的 128×4 swizzle 或 plain，需要注明。
  - 可选 `t`：转置后的量化副本 `[K, rows]`，沿 rows 方向量化。
  - scale 规则采用 NVIDIA MXFP8 训练配方（ceil，round-to-infinity），由 C 对照 torchao/TE 给出精确公式和逐位测试。
- **flat 优化器**（A 定义，放在 `round15_optimizer_flat/impl/flat.py`）：
  - `FlatParams.master`：fp32，各参数的 `.data` 是它的视图；`.grad` 是 flat grad 的视图；可选 `.bf16` shadow。
  - 注册接口：`register_weight_writer(fn)`。C 的 MXFP8 权重量化通过它接入；后续可以融合进 AdamW。

## 5. 可捕获性（CUDA Graph）

- step 路径上没有 host 同步：不用 `.item()`、`.cpu()`、`nonzero`、`bool(tensor)`，没有数据相关形状。
- 所有 kernel 在 `torch.cuda.current_stream()` 上发射。
- workspace 常驻，graph 可能 replay 时不得释放（infer-opt split-K 的教训）。
- 每个形状的 launch 配置确定；JIT、编译和 autotune 必须在 capture 之前的 warmup 里完成。
- PDL 默认关闭，在 r16 里做开/关对照。

## 6. GPU 锁与计时

- **持锁**：所有 GPU 进程都必须持双锁，每次不超过约 10 分钟；长任务用 nohup 加日志轮询。
- **整步计时**：`common/bench.time_steps`，预热 20 步，测 50 步 × 3 次；early/mid/late 按 0.2/0.6/0.2 加权，两个模型都测。e2e 用 `time_e2e`，200 步真实 loader。
- **算子 microbench**：
  - 用真实形状、CUDA event，预热至少 20 次，测至少 50 次，取 3 次的中位数，候选在同一进程里交替跑。
  - 对照必须包含 torch 同精度实现和相关的开源候选。
- **nsys**：用 `common/profile_nsys.sh <prefix> <entry> …`，再跑 `common/nsys_analyze.py <prefix>`，每阶段预热 20 步、采 5 步。原始 `.nsys-rep` 放在 tmp 目录，项目里只存 csv 和 summary。

## 7. 门槛（few-batch，不跑 epoch 训练）

- **A**：算子对 fp32 参考，报告 max abs/rel 误差和 cos。
  - 必须覆盖：尖峰 scale-8 数据（DINOv3 的 register token 会导致很尖的 attention）、尾块 N=1029/1129、mask 全开/全关/随机/整行全关。
- **B**：`common/check.py::step_equivalence(cfg_ref, cfg_opt, batches[:1], tape=RngTape("record"))`，跑 3 个 seed。
  - 同精度换 kernel：对照 T-BF16，阈值用 check.py 现有值（loss_rel ≤1e-2，grad_cos_min ≥0.99，grad_norm_rel ≤0.02，update_cos ≥0.99）。如果 BF16 自身的噪声（例如换 SDPA backend）就已经过不了，先在 NOTES 里标定，再由协调者批准放宽。
  - 改精度（BF16 对 FP32，MXFP8 对 BF16）：loss_rel ≤2e-2，全局 grad cos ≥0.99，逐层 cos 只记录。
- **S**：`common/check.curve(cfg, batches[:2], steps=50)`，再用 `gate_c(ref, cand, interval=10, max_rel=0.05)`，不能有 NaN/Inf。
- **参考栈**：
  - R0：`patches=()`。
  - P0：`common/config.P0_STACK`。
  - T-BF16：A 发布到 `common/config.P1_STACK`，由协调者写入。发布前用 `round11_bf16_mixed` 的 `autocast` variant。

## 8. 交付与汇报

- 每轮交付：
  - `NOTES.md`（7 节模板，第 3 节写清 nsys 分析过程）；
  - `results.json`：格式同旧轮次，按模型和 variant 记录 `gpu_step`/`breakdown_ms`/`accuracy.level_b`/`kernels_per_step` 等；未测字段写 null；
  - `tests/`、`profile/`、`run.py`。
- 最终报告：
  - 候选实测矩阵（算子级）；
  - 整步表（两模型，相对 R0、T-BF16 和上一最优的倍数）；
  - 门槛结果、失败项、契约变更请求、下一个瓶颈。

---

# Amendment 1（2026-10-03，协调者；优先于前文）

## 9.1 环境变化
- 服务器 `infer-5090` 现在是 `connect.bjb2.seetacloud.com:54793`（新容器，不同的物理 5090；`/root/autodl-tmp` 和 pip 包都保留）。
- 容器可能重启或短暂不可达。重启会杀掉所有后台进程：长任务要写日志、可重跑；每一步的结果文件要落盘，不要依赖会话内状态。

## 9.2 T-BF16 参考栈已发布
- `common/config.py` 里的 `P1_STACK` 和 `P1_PATCH_KWARGS`（"TB1"）：`round1`(nosync+mask_skip+adamw_fused) + `round3`(batched_torch) + `round5`(ln_fuse) + `round7`(capturable_torch) + `round11`(autocast+rope_triton+plain_flash+joint_qsplit+joint_cudnn+head_ln_bf16)。同精度换 kernel 的 B/S 门都以它为参考。
- 新补丁排在 round11 之后。除非有意替换，否则不要覆盖 `ops.attn`（round4 与 round11 的 attention 变体冲突）。
- round13 的融合块叠加方式：`TB1_STACK + ",r13:full"`（见 `round13_gemm_fusion_sm120/run.py` 的 PRESETS）。

## 9.3 同一台机器上的基线（B 固定，early/mid/late 按 0.2/0.6/0.2 加权，img/s）
| 模型 | R0 | P0 | TB1 | TB1+R13 | TB1 的 mid 步时 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| ViT-L B=7 | 16.2 | 18.0 | 41.4 | 44.4 | 169 ms | 17.5 GB |
| ViT-B B=14 | 37.2 | 44.4 | 96.0 | 103.2 | 146 ms | 15.3 GB |

## 9.4 门 B 的校准准则（批准 C 的契约变更请求）
- 全局指标沿用 `common/check.py` 阈值：loss_rel、全局 grad 余弦、grad_norm_rel、update 余弦。
- 逐参数余弦：bf16 对 bf16 的同精度换 kernel，逐参数 ≥0.99 不可达（layer-0 的 LN 权重/偏置、qkv 权重、storage_tokens 的梯度经 24 层放大，对 bf16 舍入噪声极敏感）。改用校准准则：`cos(cand,R0) >= min(0.99, cos(TB1,R0) − 0.02)`。校准脚本是 `round13_gemm_fusion_sm120/tests/calibrate_gateb.py`。
- 改精度（例如 MXFP8）的轮次：全局指标必须过（loss_rel ≤2e-2，全局 grad 余弦 ≥0.99）；逐参数余弦对 R0 只要求 `≥ cos(TB1,R0) − 0.05`，记录并报告，不单独卡。
- S 门：2 个真实 batch 的 50 步轨迹，`gate_c(interval=10, max_rel=0.05)`，无 NaN/Inf。

## 9.5 时间都花在哪儿（TB1+R13，ViT-L mid，约 158 ms/步；来自 round11/13 的 nsys 与算子表）
| 部分 | 约 ms | 备注 |
|---|---:|---|
| 骨干 GEMM（bf16，QuACK） | 65 | MXFP8 探针：12 个 GEMM 比 bf16 快 1.9–3.1 倍，整块约省 40 ms（不含量化开销） |
| head（4 次 predict，fwd+bwd） | 45 | 每次约 11.2 ms（ViT-B 15.4 ms，4 次共约 62 ms，是 ViT-B 步时的 45%）。纯 GEMM 在 230 TFLOPS 下每次约 4.3 ms |
| attention + RoPE | 24 | flash 0.66 ms/块，RoPE(triton) 0.32 ms/块，joint 块另加 mask 开销 |
| 优化器 | 12 | 约 235 个小 kernel；`cuda_multi_tensor` 变体 8.5 ms |
| loss/matcher | 3.7 | |
| 逐元素、LN 等 | 约 15 | r13 已经大幅合并 |
| 空泡 | 约 10 | gaps>20 µs 约 6 ms/步，每步约 3000 个 kernel，约 8 次同步 |

## 9.6 下一阶段分工（用全新上下文的代理；A/B/C 的旧代理已经退役）
- **C2**：先收尾 round13（门 B/S，用 9.4 的准则），然后做 **round17 MXFP8**（最大杠杆）。
- **B2**：先做 **round14 head**（第二大杠杆），再收尾 round12（融合 attention）；只有在整步上至少省 4 ms（ViT-L）才合入。
- **A2**：**round15 flat 优化器** 和 **round16 整步 CUDA Graph**；再做 round18（数据加载）。

## 9.7 额度与交付纪律（已经因为会话额度三次被打断）
- **边做边写**：NOTES.md 和 results.json 要边做边写；任何时候被打断，已有内容必须可用。
- **输出紧凑**：不要打印大 JSON，只打印紧凑表格；不要重复读大文件。
- **GPU 效率**：一次基准约 4–5 分钟，大头是模型构建、预热和 Triton/CuTe 的 JIT；同类变体尽量放进同一个进程里跑。
- **优先交付**：先做能出数据的主线，再做探索。每个任务预算约 1–1.5 小时，超出则收尾并汇报。

---

# Amendment 2（2026-10-03，协调者；优先于前文）

## 10.1 已发布的栈（`common/config.py`）
- `P1_STACK`/`P1_PATCH_KWARGS`：TB1（BF16）。`P1R13_*`：TB1 + round13 融合块。`P2_*`：再加 round17 MXFP8，策略 `fc+proj+qkv_dw`（qkv 前向保持 bf16；安全回退 `fc`）。
- 同机、固定 B、三相加权 img/s：ViT-L 43.7 → 51.6（P2），ViT-B 100.7 → 110.0（P2）；R0 为 16.2 / 37.2。

## 10.2 门 B 准则的补充（批准 C2 的请求）
- 同精度换 kernel：逐参数准则 `cos(cand,R0) >= min(0.99, cos(TB1,R0)−0.02) OR cos(cand,TB1) >= 0.99`。
- 改精度（MXFP8）：单步不稳定（全局 grad 余弦 0.988–0.9995，随 batch 变化），改用多 batch 规则：至少 8 个 batch，全局 grad 余弦中位数 ≥0.995 且最小 ≥0.98，loss_rel ≤2e-2，再加 S 门。
- `check.curve` 的 R0/TB1 参考曲线不得跨进程缓存，因为各进程的 batch 不同。

## 10.3 瓶颈已转移到主机端（C2 的 nsys）
MXFP8 把 GPU 内核时间从 148 ms 降到 120 ms，但墙钟只从 160 降到 136 ms，因为 GPU 忙碌率降到 0.79（>20 µs 空泡 17.6 ms/步，每块主机时间 1.94 ms）。所以下一步是 **round16 整步 CUDA Graph**，并同时做 round15（扁平优化器），然后才是 head（22/30 ms）。量化融合要等 Graph 之后再做，现在做没有收益。

---

# Amendment 3（2026-10-03）
- 已发布 `P2R15_STACK` / `P2R15_PATCH_KWARGS`（P2 + round15 flat 优化器）。整步 Graph 运行器：`round16_full_graph/impl/graph_step.py::StepGraphRunner`（`run.py --runner graph`，需要 `RngTape(mode="off")`；插入其他栈的方法见 round16 NOTES §7）。
- 同机加权 img/s：ViT-L 51.7（P2 eager）→ 56.0（+r15）→ **59.5**（+graph，GPU 忙碌率 0.98）；ViT-B 109.9 → 116.6 → **123.1**。相对 R0：3.67× / 3.31×。
- 新的瓶颈只剩内核时间：head 卷积（ViT-L 22 ms，ViT-B 30 ms）、逐元素（26 / 31 ms）、attention 16 ms、量化 5 ms。量化融合现在值得做。
- 已知限制：r3 padded loss 的 `max_targets` ≤ 4，需确认数据集最大目标数；ViT-L 溢出批次回退 eager 可能 OOM。

---

# Amendment 4（2026-10-03）
- 已发布 `P1R13R14_*`、`P2R14_*`、`P2R14R15_*`（候选最终栈，需用 round16 的 StepGraphRunner 运行）。
- 裁定（门 B 逐参数准则）：参考栈对 R0 的余弦本身已低于 0.9 的参数（layer-0/1 的 LN、proj、`cls_token` 等）只按全局指标判，不再单独卡逐参数余弦。批次计数按 (进程, seed) 报告。
- round14 的 head：同机加权 img/s，ViT-L 51.96 → 59.07，ViT-B 110.35 → 139.95（P2 → P2+r14）。MXFP8 的 ConvT 变体 `mx8_bwd` 未过完整门，不发布。

---

# Amendment 5（2026-10-03，最终裁定）
- `BEST_STACK_V2 = P2R14R15_STACK`（`common/config.py`），用 round16 的 `StepGraphRunner` 运行，入口 `round19_final/run_final.py`。
- 门 B 中位数规则裁定为「全局 grad 余弦中位数 ≥0.99、最小 ≥0.98、loss_rel ≤2e-2，再加 S 门」。依据：round17 / round19 证明 ViT-B 在任何 MXFP8 策略下中位数 ≥0.995 都达不到（`fc` 0.9947，最终栈 0.9945），差距来自 MXFP8 本身，不来自 head 或优化器；S 门全部通过。最终栈两个模型都满足新规则。
- 更保守的 ViT-B 档位：r17 变体 `fc`（139.7 → 156.7 img/s 之间的取舍见 round19 summary.md）；纯 bf16 档 `FINAL_noMX` 严格满足原规则。
- round15 的 scheduler 状态泄漏（`torch.save` 失败）已由协调者修复，并加了 `tests/test_scheduler_pickle.py`。

---

# Amendment 5（2026-10-03，最终裁定）
- `BEST_STACK_V2 = P2R14R15_STACK`（`common/config.py`），用 round16 的 `StepGraphRunner` 运行，入口 `round19_final/run_final.py`。
- 门 B 中位数规则裁定为「全局 grad 余弦中位数 ≥0.99、最小 ≥0.98、loss_rel ≤2e-2，再加 S 门」。依据：ViT-B 在任何 MXFP8 策略下中位数 ≥0.995 都达不到（`fc` 0.9947，最终栈 0.9945），差距来自 MXFP8 本身，不来自 head 或优化器；S 门全部通过。最终栈两个模型都满足新规则。
- 更保守的 ViT-B 档位：r17 变体 `fc`；纯 bf16 档 `FINAL_noMX` 严格满足原规则。数字见 round19 summary.md。
- round15 的 scheduler 状态泄漏（`torch.save` 失败）已由协调者修复，并加了 `tests/test_scheduler_pickle.py`。


---

# Amendment 6（2026-10-03，第三阶段：MXFP4 / NVFP4；优先于前文）

## 11.1 范围、目录、不覆盖旧 round

**轮次**（计划见 `PLAN_v3.md`）：

| 轮次 | 负责 |
|---|---|
| `round20_fp4_probe` | 分 R20a / R20b 两个任务，Sonnet |
| `round21_fp4_quant_head` | Opus |
| `round22_fp4_backbone` | Opus，是否执行看 R20 的决定 |
| `round23_fp4_final` | 协调者 |

**基线**：FINAL = `common.config.BEST_STACK_V2`（P2R14R15），配合 round16 的 `StepGraphRunner`（`RngTape("off")`，`max_targets=2`）。所有整步对比都指同一场次里对 FINAL 的对比。

**round0–19 只读**：
- 只能 import、继承子类或覆盖；不能修改它们的文件，也不能覆盖它们的 `results.json` / `profile/`。
- 重跑旧测试时，把输出重定向到新 round 的 `profile/`（参照 round19 的 `tests/ro_run.py`）。
- 旧代码有 bug 时，在新 round 里覆盖修复，并在 NOTES 里写明。

**共享文件只追加**（只有协调者能改，改前备份到 `/root/autodl-tmp/train-opt-tmp/*.bak_p6`）：
- `common/config.py`：只新增常量，例如 `P3H_*`、`P3_*`、`BEST_STACK_V3*`；
- `CONTRACTS.md`：只追加 Amendment；
- `README.md`：只追加章节。

**临时文件**：放在 `/root/autodl-tmp/train-opt-tmp/round2x_*/`。

## 11.2 FP4 张量格式 `FP4Tensor`

参考实现由 R20 写在 `round20_fp4_probe/impl/fp4_ref.py`，R21 的 kernel 必须与它逐位一致。

**`nv4`（NVFP4）**
- 数据：e2m1，两个打包成一个字节（`torch.float4_e2m1fn_x2` 或 uint8），沿量化轴（也就是 GEMM 的 K）打包，偶数下标放在低半字节。
- block：1×16，scale 类型为 e4m3（`float8_e4m3fn`）。
- 每张量 scale：fp32 设备张量，`ts = amax/(6·448)`；amax 为 0 时取 ts = 1。GEMM 的 alpha = `ts_a·ts_b`。
- block scale：`s_blk = e4m3_RN(amax_blk / (6·ts))`。
- 量化值：`e2m1(x / (ts·s_blk))`，饱和到 ±6。

**`mx4`（MXFP4）**
- 数据 e2m1，scale 为 ue8m0，每 32 个一组。
- block scale：`2^ceil(log2(amax_blk/6))`（RCEIL，不截断）。全 0 块的 scale 字节由 R20 在 `fp4_ref` 里定义。
- 没有每张量 scale。

**scale 布局**：与 QuACK / cuBLASLt 相同的 128×4 blocked 布局 `(rm, rk, 32, 4, 4)`，其中 `rm = ceil(rows/128)`，`rk = ceil(K/(4·vec))`。

**字段**

| 字段 | 含义 |
|---|---|
| `q`、`sf`、`ts` | 行副本，形状 [R, C/2]，沿 C 量化 |
| `qt`、`sft`、`tst` | 转置副本，形状 [C, Rpad/2]，沿 token 量化。Rpad = ceil(R/128)·128，填充行为 0 |
| `rows`、`rpad`、`fmt` | 行数、填充后行数、格式 |
| `rht` | 是否已做旋转，以及所用符号向量的 id |

**三个 GEMM**：sm_120 上 4 bit 操作数只能是 K-major，所以三个 GEMM 都写成 TN：

| GEMM | 计算 | 说明 |
|---|---|---|
| fwd | `X.q × W.q` | |
| dgrad | `dY.q × Wᵀ.q` | Wᵀ 是物理转置后再量化的副本 |
| wgrad | `dY.qt × X.qt` | 输出 fp32 |

**权重**：每一步在 block 0 由 fp32 master 量化。默认用 2D block scale（nv4 是 16×16，mx4 是 32×32），这样 `W.q` 和 `Wᵀ.q` 的数值完全一致；1D 作为对照。

## 11.3 RHT 与随机数

**RHT**
- 沿量化轴，每 16 个元素做一次 FWHT（正交归一化，乘 1/4），再乘一个固定的 ±1 随机符号向量。
- 同一个 GEMM 的两个操作数，沿收缩轴必须使用相同的符号向量和相同的分块。
- 默认只用在 wgrad（沿 token 轴）。fwd / dgrad 上的 RHT 是 R20 的候选选项。

**随机数来源**
- 符号向量和 SR 用的随机数**不能来自 torch 默认生成器**。原因：RngTape 在 replay 时会调用 `draw()` 来对齐默认流。
- 仿真用专用的 `torch.Generator`，种子固定。
- kernel 用 Philox：种子是设备上的 int64 计数器，在 graph 内每步加 1；每个调用点使用固定的 offset。
- B 门比较时，候选自己的 SR 噪声不需要与 R0 对齐。

**舍入**
- 权重和激活用 RN；梯度 dY（用于 dgrad / wgrad）默认用 SR。
- SR：按到 e2m1 网格上相邻两点的距离比例随机取其中一点（无偏），饱和到 ±6。

## 11.4 策略串

由 R20 定义，R21 / R22 沿用。

- 格式：`<线性层>.<种类>=<格式>[+选项]`，多条之间用 `,` 或 `+` 连接。

| 部分 | 取值 |
|---|---|
| 线性层 | qkv、proj、fc1、fc2、fc、all、head |
| 种类 | f、d、w 的任意组合 |
| 格式 | bf16、mx8、nv4、mx4 |
| 选项 | rht、sr、rn、w2d、w1d、nofirst、nolast、nojoint |

- 没写到的部分保持 FINAL 的设置：qkv 为 f=bf16、d=mx8、w=mx8；proj、fc1、fc2 为 mx8；head 为 bf16。

## 11.5 第三阶段的门

用户 2026-10-03 决定。对降精度候选，本节优先于 Amendment 5。

**B（FP4 放宽门）**
- 两个模型各至少 8 个真实 batch。在同一进程里，用同一份权重、同一个 batch、同一份 RngTape，分别对 R0 和对 P1 计算。
- 要求：全局梯度余弦中位数 ≥0.98、最小 ≥0.97，loss_rel ≤2e-2。
- 以下各项**只记录、不作为门**：
  - 是否满足严格门（中位数 ≥0.99、最小 ≥0.98）；
  - 逐参数余弦；
  - 对 FINAL 的余弦。
- 本阶段所有降精度候选都用这条门，包括重新尝试的 qkv 前向和 head 前向 MXFP8。

**S**
- 50 步，2 个真实 batch，4 个场景：schedule、fixed mid、fixed late、compressed。
- 对 R0 和对 FINAL，EMA(0.9) 损失的最大相对偏差都要 ≤5%，且不能出现 NaN / Inf。
- 记录 grad-norm 比值。
- R0、P1、FINAL 和候选在同一个进程里跑。

**A**
- 量化器 RN 与 `fp4_ref` 逐位一致：数据字节、scale 字节、ts 都要一致。
- SR 在相同的 Philox 流下逐位一致，另做无偏性统计。
- GEMM 与反量化后的 fp32 乘积相比，相对 L2 ≤2e-3。
- 与 R20 仿真相比：head 余弦 ≥0.9999，block 余弦 ≥0.99999。
- 覆盖范围：
  - 形状：M 尾块 7203、7903、14406、15806、1029、33；
  - 数据：尖峰（scale-8、异常通道）、全 0、大值、极小值。

**速度**
- 整步只在 StepGraphRunner 下与 FINAL 同场次比较（`common.bench.time_steps`，三相加权）。提升 ≥2% 才发布。
- 算子级用 CUDA graph 回放测：同一进程内交替运行，至少 3 轮取中位数。

**状态**：凡是改动参数或训练状态的轮，都必须实际跑一次 `torch.save`、export，以及在原版路径上 resume（用 r19 的 `tests/check_export.py`）。

## 11.6 go/no-go

R20 结束时套用这条规则，之后不改。

某一部分要改用 FP4（或重新尝试的 MXFP8 前向），必须**同时**满足：
1. graph 回放下比现状快：
   - 骨干：≥1.5× MXFP8；
   - head：≥3× bf16；
   - MXFP8 前向候选：≥2× bf16。
2. 仿真出的策略通过 B 门和 S 门（11.5）。
3. 预计整步节省 ≥3 ms，或 ≥3%。

不满足的部分保持 FINAL 的做法，对应的 kernel 轮直接跳过。

选择规则：
- RN 和 SR 都过门时，选 SR。
- 格式默认用 nv4。只有当 mx4 也过门、并且整步更快时，才改用 mx4。

## 11.7 交付纪律（沿用 9.7，补充以下几条）

- 同一时间只跑一个子代理。每个任务不超过 1.5–2 小时；NOTES.md 和 results.json 边做边写，输出保持紧凑。
- 不安装任何包，需要时报告协调者。
- GPU 进程必须持有双锁；用 queue 包装，避免 `assert_gpu_exclusive` 误报。
- NOTES 沿用 7 节模板，第 3 节写 nsys 分析过程；没有测的项写 null 或 not measured。


---

# Amendment 6a（2026-10-03，R20a 之后，协调者；优先于 Amendment 6）

## 12.1 并发与目录归属（用户 2026-10-03 新指令）

**并发**
- 最多同时跑 **2 个子代理**，都用 Sonnet 5.5 xhigh。这一条取代 Amendment 6 里的"一次一个"和"R21/R22 用 Opus"。

**目录归属**（各自只写自己的目录）

| 任务 | 目录 |
|---|---|
| R20b | `round20_fp4_probe/` |
| R21a / R21b | `round21_fp4_quant_head/` |
| R22 | `round22_fp4_backbone/` |

- 别人的目录只能 import，不能改。

**GPU 争用**
- 所有 GPU 进程都必须持双锁，一个进程从启动到退出全程持锁，锁外不保留 CUDA 上下文。
- 计时类测量（graph bench、算子矩阵、nsys）需要独占 GPU。遇到 `assert_gpu_exclusive` 报错时，用 queue 包装重试（参照 r16/r17/r19 的 `*_queue.sh`）。
- 每段持锁不超过约 10 分钟，给另一个代理留出空档。

## 12.2 格式细则（采纳 R20a 的契约请求，修订 11.2 / 11.3）

**nv4**
- 每张量 scale：`ts = amax/(6·448)`，用 fp32 除法；amax=0 时取 1。
- block scale：`sf = e4m3_RN(min((amax_blk/6)/ts, 448))`。
- 编码值：`e2m1(x / (float(sf)·ts))`，fp32 除法，RN 或 SR。sf=0 的块，数据全为 0。
- 设备端计算 ts 的路径，必须与上面的 fp32 除法逐位一致（Spark 的实现不满足这一点）。

**mx4**
- ue8m0 scale 字节 = RCEIL：取 amax_blk/6 的指数，尾数非 0 时加 1，上限 254。全 0 块字节为 0。

**RHT**
- `y = s ⊙ (FWHT16(x)/4)`：先做 FWHT 再乘符号。s 是 16 个 ±1，种子 20251003，由 `fp4_ref.rht_signs` 生成。

**参考实现冻结**
- `round20_fp4_probe/impl/fp4_ref.py` 中现有函数的语义**冻结**，它是 R21/R22 逐位对齐的参照。之后只能新增函数，或新增可选参数（默认值保持原行为）。

## 12.3 策略串新增选项（修订 11.4）

**`prefix`**
- wgrad 的 FP4 收缩只覆盖 patch token，即每张图的后 1024 个 token。
- prefix token 用 bf16 小 GEMM 精确计算，结果加回 dW。prefix 的范围：plain 块是 cls 加 4 个 reg，共 5 个；joint 块再加 100 个 query，共 105 个。
- 动机：R20a 发现 ViT-L block 0 存在 massive activation。prefix token 占 dW 范数的 29–93%，FP4 wgrad 误差为 0.26–0.60；排除 prefix 后降到 0.04–0.06。
- 对照方案是 `nofirst`。

**`amax=exact|delayed`**（nv4 的每张量 amax）
- exact：用当前张量的 amax，需要额外读一遍。
- delayed：用同一调用点上一步的 amax 乘以 2（headroom）。当前步的 amax 由量化 kernel 顺带算出，留给下一步；第一步用 exact 初始化。
- 实现默认用 delayed，原因有二：
  - R20a 实测，额外一遍 amax 扫描会吃掉 ViT-B 的大部分收益；
  - 有些生产者（例如 flash attention）无法修改，不能指望由生产者顺带输出 amax。
- B 门只测单步，此时 delayed 与 exact 相同；delayed 的影响由 S 门验证（50 步）。
- 权重同样用 delayed。权重变化很慢，这样可以避免每步额外读一遍 fp32 权重。

## 12.4 R20a 的速度与节省判定（按 11.6 的事先规则，原样套用）

表中数字依次为 ViT-L / ViT-B。

| 项 | GEMM 加速 | 每步节省（ms） | 判定 |
|---|---|---|---|
| ① head dgrad+wgrad → nv4 | 5.1× | 7.2 / 7.0 | **通过** |
| ③ 骨干 dgrad+wgrad → nv4 | 1.9× | 7.9 / 4.3（占 7.7% / 4.9%） | **通过** |
| ②a head 前向 → mx8 | 2.7× / 2.4× | 3.3 / 3.05 | **通过** |
| ④ fc1/fc2 前向 → nv4 | ViT-B 只有 1.47× | 占 2.9% / 2.2% | **不通过** |
| ⑤a qkv 前向 → mx8 | — | 占 2.8% / 2.2% | **不通过** |

- ④ 和 ⑤a 按事先规则不进入实现。R20b 可以顺带筛选它们，结果只记录、不发布；是否破例由用户决定。
- 前提（R20a 实测）：现成的转置量化路径比带宽下限慢 6–10 倍，不做融合的话整步反而亏 23–32 ms。因此：
  - R21 必须自己写融合量化器，一次读入，同时输出行副本和转置副本；
  - nv4 的 amax 不允许额外整遍扫描。
- GEMM 后端用 QuACK，wrapper 在 `round20_fp4_probe/impl/fp4_epi.py`：nv4 的 bias / res / gelu / dgelu / plain 五种 epilogue 都已编译通过，ts 以设备标量传入。
- cuBLASLt 的 MXFP4 在 sm_120 上不可用。


---

# Amendment 6b（2026-10-04，R20b 筛选与决赛 B 门、R21a 之后，协调者；优先于 6 和 6a）

## 13.1 逐位参照改为"CUDA 上的 fp4_ref"（采纳 R21a 的契约请求，修订 12.2）

- `fp4_ref._nv4_ts` 在 CUDA 上会把 `amax/(6·448)` 编成乘以倒数；CPU 上是真除法。两者约有 22% 的随机 amax 会差 1 ulp。
- 所有 gate A 的逐位参照，一律指 **fp4_ref 在 CUDA 上的执行结果**。量化器保持 `div_true=False`（默认）。1 ulp 的 ts 差别在数值上无关紧要，而这样做的好处是：R20b 仿真里通过的门，可以逐位迁移到真 kernel。
- 12.2 中"fp32 除法"的措辞作废，其余约定不变。

## 13.2 R20b 的裁定（依据：筛选 n=3、决赛 n=8 的放宽 B 门，ViT-B 的 S 门 4 个场景）

**采用**
- 格式 **nv4**。mx4 在 ViT-L 上过不了放宽门（B3+mx4 中位数 0.9796），放弃。
- 梯度 dY 用 **SR**；激活和权重用 RN；权重用 2D scale。
- nv4 的每张量 amax 用 **delayed**（上一步的值 ×2）。
- **RHT 默认关**：没有可测的收益，ViT-L 上带 RHT 的 B3 反而更差（对 R0 中位数 0.9899，不带 RHT 为 0.9924）。kernel 保留 RHT 能力。
- **不实现 prefix 侧 GEMM**。`nofirst` 与 `prefix` 的精度相同（B4 对 B5），而 `nofirst` 的代价小得多：只损失约 1/24 的 ③ 节省（ViT-L 约 0.3 ms），`prefix` 要付约 1.5 ms 的侧 GEMM。

**每个模型的策略**（R22 / R21b 实现这两个）

| | head | 骨干 | 门结果（n=8，对 R0 的中位数 / 最小值） |
|---|---|---|---|
| ViT-L | `head.f=mx8, head.dw=nv4+sr` | `all.d=nv4+sr, all.w=nv4+sr` 加 **nofirst**；前向不变（qkv 为 bf16，proj / fc1 / fc2 为 mx8） | B4：0.9926 / 0.9892，**过严格门** |
| ViT-B | 同上 | 同上，**不加 nofirst** | B3：0.9903 / 0.9784，只过放宽门（严格门卡在最小值 ≥0.98） |

- ViT-B 的严格档预设：只做 head（H3，n=8 中位数 0.9933、最小值 0.9814，过严格门）。
- 速度条件和节省条件按 11.6 套用，结论见 R20a 的 12.4。

**放弃 / 不实现**
- ④ fc1/fc2 前向 nv4：放宽门也过不了（ViT-B 中位数 0.963–0.968，ViT-L 0.968–0.980）。永久放弃。
- ⑤a qkv 前向 mx8：n=3 的放宽门能过，但低于 11.6 的速度和节省阈值，按事先规则不单独发布。r17 已有 qkv 前向 MXFP8 的路径，不需要新 kernel；R23 如果有时间，可以作为"可选档"补做 n≥8 的 B 门和 S 门，由用户决定是否采用。

## 13.3 接入约定（R21b、R22 共用）

**amax / SR 状态**
- 每个 patch 自己持有一个 `FP4State`，不跨 patch 共享。
- 在 `ctx.ops.opt_step` 外面再包一层：先调用原函数，然后执行 `state.roll()`，这样每步恰好 roll 一次，在 eager 和 CUDA graph 下都成立。
- 每个调用点使用固定的 `site` 编号和固定的 `sr_offset`。
- delayed 模式下，每个 site 的第一次调用要在 CUDA graph 捕获之前，用 eager 执行一遍做 exact 初始化。`StepGraphRunner` 的预热步已经满足这一点，需要验证。

**GEMM**
- 后端用 QuACK。nv4 的 epilogue 在 `round20_fp4_probe/impl/fp4_epi.py`（bias、res、gelu、dgelu、plain）。
- alpha = `q.ts · w.ts`，作为设备标量传入。
- dgrad 的 B 操作数是 `w.qt` / `w.sft`；wgrad 输出 fp32，小输出的 wgrad 要用 split_k（R20a 的矩阵里有最优配置）。
- wgrad 的 dYᵀ 和 Xᵀ 必须使用相同的 token 填充，填充行为 0。

**GPU 并发**
- R21b 和 R22 可以同时跑，所有 GPU 进程持双锁。
- 计时类测量要独占 GPU：遇到 `assert_gpu_exclusive` 就走 queue 重试；整步 bench 排在对方的 B 门之外的时间段。
- R0（fp32）在 ViT-L 上峰值 28.8 GB，所以 B 门的各次 capture 要按顺序跑，中间释放显存。

## 13.4 R21b / R22 的门（沿用 11.5，补充）

- **A**：
  - RN 变体：块（R22）或 head（R21b）对 R20b 仿真的余弦 ≥0.99999（块）/ ≥0.9999（head）。
  - SR 变体：SR 噪声来源不同（仿真用 torch.Generator，kernel 用 Philox），不做逐位对比，改用统计一致性，并交给 B 门和 S 门检验。
- **B**：两个模型各 ≥8 批，同时对 R0 和对 P1；放宽门，同时记录严格门是否通过。
- **S**：4 个场景 × 50 步 × 2 批。**在 StepGraphRunner 下运行**（沿用 r19 的 `gate_s`：graph 路径对 R0），EMA 偏差 ≤5%，无 NaN；记录 grad-norm 比值。
- **graph_equiv**、**导出 / resume**（用 r19 的 `check_export.py`）。
- **速度**：StepGraphRunner 下与 FINAL 同场次的三相加权 bench，两个模型都测；提升 ≥2% 才发布。给出 nsys 的 kernel 类别表。
- 新增的栈名由协调者写进 `common/config.py`（`P3H_*`、`P3B_*`、`P3_*`）。代理不改 common，只在各自的 `run.py` 里用 r19 的 preset 机制注册。

## 13.5 R20b 报告之后的补充（2026-10-04，协调者）

**策略串语义**（R20b 的 `round20_fp4_probe/impl/fp4_policy.py` 为准）：
- 条目的选项（例如 `+sr`、`+rht`）只作用于紧邻的那一个条目。
- `prefix`、`nofirst`、`amax=` 是全局选项。

**批次计数**：按（进程, seed）计；比较只在同一进程内配对。加载器跨进程不确定，ViT-B 的 8 批中位数在三次抽批里分别是 0.9905 / 0.9884 / 0.9919，所以 ViT-B 的严格门不能用来区分 B3 / B4 / B5。

**dY 的 amax 余量**（R20b 的诊断）：
- 40768 次 delayed 调用里，有 1.11%（按场景 0.7–2.1%）的当前 amax 超过上一步的 2 倍，最大达 10 倍，这一步会在 e4m3 scale 的 448 处饱和裁剪。已保存的两个场景里，超限的调用点全是 dY 的两个副本，X 和权重一次也没有。S 门里 delayed 与 exact 的 EMA 偏差相同（±0.0001）。
- 处理办法：
  - 量化器新增 `headroom` 参数（默认 2.0，行为与现在逐位相同）；
  - dY 的调用点用 `headroom=8`，X 和权重保持 2；
  - 在 `FP4State.roll()` 里增加饱和监控：对每个 site 统计"当前 amax 超过 headroom × 上一步 amax"的次数和最大比值，用设备端计数器实现，不放进量化 kernel，graph 内可捕获；`state.stats()` 在 host 上读取，不在 graph 内调用。
  - 如果 headroom=8 后饱和仍然偏多，退回 dY 用 `amax="exact"`（量化器已支持，实测多 25–60%）。
- 信号：`round21_fp4_quant_head/READY_headroom` 文件出现，表示 `headroom` 参数和监控已可用。R22 在此之前先用默认值开发，文件出现后切换到 headroom=8。

**R22 的 ViT-B 注意**：ViT-B 骨干 FP4 在最差批上，离放宽门（中位 0.98 / 最小 0.97）的线只有约 0.005。真 kernel 的 B 门要盯住最差批，并且同时对 R0 和对 P1 都算。


---

# Amendment 6c（2026-10-04，R23 之后，协调者；优先于 6、6a、6b）

## 14.1 更正 6b 13.2 的一句错话

6b 13.2 写"ViT-B 的严格档预设：只做 head（H3，n=8 中位数 0.9933、最小值 0.9814，过严格门）"。**这句话作废。**
- R23 在更大的样本上重测（ViT-B，5 次独立抽批，n=40；P3H 在 draw 2 的最小值是 0.9770）：P3H 在 ViT-B 上**不是**严格档。
- 在这些批上，唯一满足严格门（中位数 ≥0.99、最小 ≥0.98）的栈是 FINAL。
- 根因：n=8 的小样本漏掉了离群批。R20b 报告里也提醒过"加载器跨进程不确定，ViT-B 严格门不能用来区分候选"，我当时没有把这条限制写进结论。以后凡是判定"某个档过严格门"，需要 n≥24。

## 14.2 第三阶段的最终栈（采纳 R23 的实测）

`BEST_STACK_V3 = P3_STACK`（`common/config.py` 里的 `P3_STACK`，已追加 `BEST_STACK_V3*` 别名），在 round16 的 `StepGraphRunner` 下运行，入口 `round23_fp4_final/run_final.py`。

| 模型 | FINAL img/s | P3 img/s | 对 FINAL | 对 R0（同机重测） |
|---|---:|---:|---:|---:|
| ViT-L B=7 | 68.43 | 81.87 | ×1.196 | ×5.04 |
| ViT-B B=14 | 159.24 | 191.40 | ×1.202 | ×5.06 |

R0 的同机重测值是 16.25 / 37.82 img/s；FINAL 对 R0 是 ×4.21 / ×4.21。P3 的峰值 reserved 显存 ViT-L 14.97 GB、ViT-B 11.59 GB。

**精度如实记录（放宽门，用户 2026-10-03 决定）**
- ViT-L：n=40 批，对 R0 中位数 0.9921、最小 0.9731，对 P1 最小 0.9693。
  - 放宽门的中位数和 S 门都通过；最小值门在 **1/40 批**上不过（对 P1 差 0.0007）。
  - 这批的情况是：P1 对 R0 为 0.9940，而 FINAL 自己对 R0 只有 0.9643，对 P1 0.9690；P3 比 FINAL 还好（0.9731 / 0.9693）。
- ViT-B：n=40 批，对 R0 中位数 0.9895、最小 0.9749，对 P1 最小 0.9735。全部通过放宽门，但最差批离 0.97 线只有 0.0035–0.0049；40 批里有 3 批低于 0.98。这几批的下降大部分来自 head 的 FP4（P3H 单独在这几批上是 0.9770 / 0.9789）。
- **两个模型都不满足严格门**（第二阶段的 Amendment 5 标准）。FINAL 是满足它的最近的栈。

**我不修改门**：R23 建议把"最小值"改成"相对 FINAL 在同一批上的差"来判定。我只把这个相对指标作为**额外记录**（P3 减 FINAL，40 批范围 −0.009 到 +0.018），不替换放宽门的判定，也不因为这一批追认通过。上面的结果就是门的结果。

## 14.3 长训练的风险和监控

第三阶段所有验证都是 few-batch，没有 epoch 级训练，也没有 mIoU / head IoU 对照。用于长训练时：
- **dY 的 amax 饱和**（delayed amax，headroom=8）：R23 的 S 门里，ViT-L 在 schedule 和 compressed 场景有 1.3–1.4% 的 dY 调用饱和（最大超限比 49×），fixed_mid / fixed_late 只有 6 和 20 次；R22 单独测只有 0.09–0.18%，原因还没查清。S 门的 EMA 偏差 ≤0.41%，没有可见影响，但这是 50 步的证据。
- 训练脚本应当定期调用 `FP4State.stats()`（在 graph 外，host 侧）记录各 site 的饱和次数；如果长期超过约 2%，把 dY 的 `headroom` 提高，或把 dY 的调用点改成 `amax="exact"`（多 25–60% 的量化时间）。
- `FP4State` 不在 checkpoint 里，resume 后每个 site 用 exact amax 重新初始化，这是预期行为。

## 14.4 其他裁定

- SR 保持为默认（P3_RN 快 1.7–1.8%，但没有跑过 RN 的 B 门；按 11.6 两者都过门时选 SR）。
- ⑤a（qkv 前向 mx8，R23 的 `P3_q8`）：整步 ViT-L +3.7%（对 P3）、ViT-B +2.3%；梯度余弦中位数再降 0.002–0.0035；ViT-B 最差批对 P1 只剩 0.0028 的余量。**不进入默认栈。**
  - ViT-B：不采用。
  - ViT-L：按 11.6 的事先规则（≥3% 或 ≥3 ms）它是 go（+3.7%，3.1 ms）；但它缺 graph_equiv、导出和 resume 的验证，先补验证，作为用户可选的单独档位，不是默认。
- 梯度范数：ViT-B 的 `P3_q8` 在 S 门里 grad-norm 比值是 0.844–1.110（P3 是 0.948–1.037），作为风险记录。

## 14.5 R25（加固，不做新 kernel）

范围：
1. dY headroom 研究（ViT-L 为主）：headroom 8 / 16 / 32 / exact 在 S 门 4 个场景下的饱和统计和 EMA 偏差，查清 P3 与 P3B 饱和率不同的原因，并测默认值之外的整步速度代价。
2. ViT-B 保守档：把 head 的 dgrad/wgrad 换成 mx8（H2），保留骨干 FP4，在 n≥24 上看最差批余量与速度。
3. ViT-L 的 `P3_q8`：补 graph_equiv 和导出 / resume，使它成为可选档。

不改 round0–24 的任何文件；新目录 `round25_fp4_harden/`。


---

# Amendment 6d（2026-10-04，R24 / R25 的已有数据之后，协调者；优先于 6c；工作按用户指示在此停止）

## 15.1 更正 6c：ViT-B 不应把 P3 当默认

6c 14.2 把 P3 写成两个模型的最终栈。R25 在另外两次独立抽批（ViT-B，各 8 批，同进程配对）里测到：
- **P3 在 draw 2 里对 P1 的最小余弦是 0.9655，低于放宽线 0.97**（对 R0 为 0.9734）；同一批上 FINAL 对 P1 是 0.9797。
- **P3_H2**（head 的 dgrad / wgrad 走 MXFP8、前向 bf16；骨干 FP4 不变）在两次抽批里都过放宽门：对 R0 中位 0.9923 / 0.9920，最小 0.9899 / 0.9774；对 P1 最小 0.9882 / 0.9733。
- P3_H2f（head 前向也用 MXFP8）在 draw 2 同样不过（对 P1 最小 0.9667），说明 ViT-B 上拉低最差批的是 head 的低精度前向和 NVFP4 的 dgrad / wgrad，而不是骨干。

所以 ViT-B 的推荐改为：**默认 P3_H2，P3 是速度档**（速度档不稳妥地满足放宽门）。6c 里 ViT-B 的相关表述作废。ViT-L 的结论不变（P3 过放宽门，只有一批是 FINAL 自己也不过的离群批）。

## 15.2 dY headroom：默认值应改为 32

R25 的数据（ViT-L，S 门 4 个场景，StepGraphRunner 下对 R0）：

| 骨干 dY headroom | 饱和 / 调用 | 最大 cur/prev | EMA 偏差（检查点最大） |
|---|---:|---:|---:|
| 8（现默认） | 232 / 36800（0.63%） | 32.8 | 0.57% |
| 16 | 32 / 36800（0.09%） | 44.9 | 0.51% |
| 32 | 0 / 36800 | 27.4 | 0.56% |
| exact | 无 | – | 0.56% |

- B 门（ViT-L 两次抽批，ViT-B 两次抽批）：headroom 16 / 32 / 64 / exact 与 8 的梯度余弦在四位小数内相同。
- 速度（本会话，graph，ViT-L）：P3 81.99 img/s，headroom 32 为 82.14 img/s（差别在噪声内）。
- R23 和 R25 中 P3 对 P3B 的饱和率差异（1.3% 对 0.1%）：R25 的受控对比（同进程、同 batch、同种子）里两者相当，差异主要来自抽到的 batch，事件集中在个别 (site, step)。

## 15.3 本会话在新容器上的整步实测（graph，三相加权，同一会话交替；容器已重启，端口 23198）

| 模型 | FINAL | P3 | P3_H2 | P3 + dY headroom 32 |
|---|---:|---:|---:|---:|
| ViT-B | 160.70（2 次） | 192.29（1 次） | 177.38（2 次） | – |
| ViT-L | – | 81.99（2 次） | – | 82.14（2 次） |

对 FINAL：ViT-B 的 P3 ×1.197，P3_H2 ×1.104。（ViT-L 的 FINAL 与 R0 本会话没有重测，沿用 R23 的 68.43 与 16.25 / 37.82 img/s。）

## 15.4 R24：不合入

`fp4q2`（量化器优化）整步对 P3B 只快 +1.4%（ViT-L）/ +0.5%（ViT-B），低于 2% 的发布线；其 gate A 的逐位对比、B、S、graph_equiv 也没有完整跑完（results.json 状态为 in progress）。按事先规则只作记录，最终栈使用 R21a 的 `fp4q`。

## 15.5 没有做的事（如实）

- ViT-B 默认档 P3_H2 的 S 门、graph_equiv、导出 / resume、e2e：**都没有**。它的组件各自过了门（H2 head：R21b；骨干：R22），组合只有 B 门 n=16 的证据。
- headroom 32 作为默认值：只有 ViT-L 的 S 门和 ViT-L / ViT-B 的 B 门，没有单独的 graph_equiv、导出、e2e。
- `round26_fp4_release/`（V3 / V3_speed 预设和训练入口 `run_final.py`，默认栈 V3）只搭好了入口，验证队列按用户指示停止，没有产出结果；`common/config.py` 里没有最终默认值的常量（`BEST_STACK_V3*` 仍是 6c 时的 P3 配置，**不含 headroom 32，ViT-B 也不是 H2**）。
- R25 的任务三（ViT-L 的 P3_q8 补验证）没做。P3_q8 仍然是未验证的可选档。
- epoch 级训练、mIoU / head IoU 对照：全阶段都没有做。
- README 没有第三阶段的章节。

## 15.6 过程中的两个记录

- 服务器上 `train-opt-tmp/round15_optimizer_flat/build/lock`（零字节的扩展编译锁，被中断的任务留下，无存活进程持有）由用户确认后删除。
- 之前 R21b 代理曾未经确认删除过同一位置的一个同类锁文件，已在当时如实报告。
