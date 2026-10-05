# Round 6：低精度（FP8 e4m3）—— 精度先行

环境：bf16-autocast 基线（fp32 权重 + autocast），真实视频 8 帧 gate（`mf.gate`）+ 全部 126 帧的 held-out 验证（对 fp32 eager 现算参考）。
前后处理用 Round 7 的 fused kernel（与 fp32 参考的差异远小于 FP8 误差）。基线 gate：ViT-L pooled IoU 0.9933 / conf 0.45%；ViT-B 0.9923 / 0.09%。
（126 帧上基线 bf16 对 fp32：ViT-L IoU 0.9929、conf 最大 1.5%；ViT-B 0.9914、0.46%。）

## 1. 阶段 1：逐层敏感度

### 方法
- `impl/fp8sim.py`：把选定的 nn.Linear 用 e4m3 模拟替换（monkeypatch 实例 forward）。量化值做 fp32 GEMM（等价于 fp32 累加的 FP8 tensor core，实测 `_scaled_mm` 累加误差 ~1e-9，与 fp32 一致），再套用基线的 bf16 输出舍入。
- 粒度：`tensor`（动态 per-tensor 激活 / 静态 per-tensor 权重）、`row`（per-token 动态激活 × per-output-channel 静态权重，即 `_scaled_mm` rowwise）、`b32`（1×32 块、E8M0 二次幂缩放，MXFP8 风格）、`b128`（1×128 块，fp32 缩放）。
- 扫描（`sens.py`，日志 `sens_*.jsonl`，共 ~470 个配置）：A = 层类型 × 全部层 × 4 种粒度；B = 层范围（前半/后半/前 k/后 k/joint 3 层/除 joint 外）× {all4, mlp} × {row, b32}；C = 每个 (层, 类型) 单独量化。指标：pooled head IoU、最差单帧 IoU、conf 最大相对误差、logit MAE（对 fp32 参考，连续指标）。
- 贪心（`greedy.py`）：按单 unit 的 logit-MAE 增量从小到大排序，逐步加 unit（步长 2），选出满足 `gate 通过 ∧ conf≤3% ∧ pooled IoU≥0.975` 的最大前缀，再在 126 帧上对 fp32 现算参考做 held-out 验证。

### 主要发现
1. **ViT-L 对 FP8 非常敏感，ViT-B 稍好**。head 只有约 2000 个像素，IoU 和 confidence（head 像素上的均值）对微小扰动非常敏感；conf 相对误差对量化噪声是“混沌”的（同一配置换一种舍入，2.6% 可以变成 10%）。logit MAE 才是平滑的排序指标。
2. **全部层全部类型**（all4/all layers）：所有粒度都不过 gate（ViT-L：tensor IoU 0.892/conf 47%，row 0.938/0.8%(IoU 不过)，b32 0.879，b128 0.953/20%；ViT-B：row 0.941、b32 0.939，仅 b128 0.950/0.8% 过线）。
3. **分类型（全部层）**：ViT-L 只有 `row: attn`(qkv+proj)、`b32: mlp`、`b128: proj/fc2` 通过；qkv 最敏感（ViT-L b128 也 11%），proj 最不敏感。ViT-B 上 proj/fc1/fc2（任意粒度）通过，qkv 在 tensor/row 失败（b32/b128 勉强通过）。
4. **分层范围**：ViT-L joint 层（后 3 层）的 mlp 单独 FP8（row/b32）几乎无损（IoU 0.992/0.991，conf 3.3%/0.8%）；“除 joint 外全部量化”、“前半/后半 all4”都失败。ViT-B 类似。
5. **单层**（row）：ViT-L 最敏感的是 L0.fc2（dMAE +2.0e-5，且 L10/L17 的 qkv、L12/L17 的 fc1…）；平均敏感度 qkv(0.29) > fc2(0.23) > fc1(0.18) > proj(0.10) e-5，各层段无明显差别。ViT-B 中 qkv 最敏感（平均 1.04e-5，L2~L9 的 qkv 全在 top10）。
6. **粒度**：细粒度并没有稳定地更好（ViT-L 全层 all4：b128 IoU 0.953 好于 row 0.938，但 b32 最差 0.879）；贪心可覆盖 unit 数（8 帧 gate，含 proj，gate+conf≤3%+IoU≥0.975）：

| 模型 | row | b32 | b128 | 总 unit |
|---|---|---|---|---|
| ViT-L | 52（qkv9/proj18/fc1 12/fc2 13） | 10 | 40 | 96 |
| ViT-B | 22 | 12 | 20 | 48 |
  126 帧 held-out（对 fp32，sim）：ViT-L row-52：IoU 0.977 / conf 最大 2.8%；b128-40：0.978 / 2.1%；b32-10：0.991 / 3.7%。ViT-B row-22：0.973 / 1.5%；b128-20：0.978 / 1.8%；b32-12：0.980 / 0.4%。
   （proj 的 FP8 在性能上几乎没有收益，见下，因此性能方案排除 proj 后再次贪心：`*_row_noproj.json`。）

### 为什么用“real FP8”再验证
sim 与 real 的 logit MAE 一致（`test_fp8.log`，例如 ViT-L all-proj 2e-5 = 2e-5），但 IoU/conf 这类离散指标每次实现的舍入细节（LN 输出先舍入 bf16 再量化等）会造成漂移，因此**所有推荐方案都用真实 FP8 kernel 在 8 帧 gate + 126 帧上重新验证**（`heldout_real.py`）。

### 真实 FP8（含融合 head）在 126 帧上的精度（对 fp32；贪心前缀，proj 排除，row）
ViT-L（基线 IoU 0.9929，conf max 1.5%）

| unit 数 | 8 帧 gate（IoU / conf max） | 126 帧 pooled IoU | 126 帧 conf max / p99 | >3% 的帧数 |
|---|---|---|---|---|
| 0 | 0.9930 / 0.44% | 0.9929 | 1.5% / 1.2% | 0 |
| 6（fc1×1 fc2×5） | 0.9888 / 1.6% | 0.9897 | **2.7%** / 1.6% | 0 |
| 10 | 0.9845 / **9.3%（fail）** | 0.9884 | 9.3% / 2.7% | 2 |
| 14 | 0.9840 / 5.0% | 0.9865 | 5.0% / 3.6% | 2 |
| 18（qkv3 fc1 8 fc2 7） | 0.9841 / 2.2% | 0.9855 | 4.5% / 2.3% | 1 |
| 24 | 0.9796 / **10.1%（fail）** | 0.9831 | 10.1% / 6.5% | 3 |

ViT-B（基线 0.9914 / conf max 0.46%）

| unit 数 | 8 帧 gate | 126 帧 pooled IoU | 126 帧 conf max | 
|---|---|---|---|
| 0 | 0.9921 / 0.11% | 0.9914 | 0.46% |
| 4 | 0.9894 / 0.18% | 0.9891 | 0.75% |
| 8 | 0.9852 / 0.26% | 0.9843 | 0.90% |
| 14（qkv3 fc1 5 fc2 6） | 0.9825 / 0.53% | 0.9802 | 1.35% |
| 20（qkv4 fc1 8 fc2 8） | 0.9643 / 0.74% | 0.9732 | 0.99% |
| 28 | 0.9525 / 0.57%（IoU 贴线） | 0.9609 | 2.0% |

### 推荐方案
- **ViT-B：14 个 unit（qkv3 / fc1 5 / fc2 6，row 粒度）**，IoU 0.980，conf ≤1.4%，有余量。想要更多收益可到 20 个（IoU 0.973，仍过 gate，conf 1%）；不建议 28（IoU 0.952 贴线）。
- **ViT-L：保守 6 个 unit（fc1 ×1、fc2 ×5，多数在 joint/尾部层），126 帧 conf 最大 2.7%（≤3%）**；更激进的 18 个 unit（IoU 0.985，8 帧 conf 2.2%，但 126 帧里有 1 帧 4.5%）能过 8 帧 gate，但**不满足“≤3% 留余量”**；24 个 unit 会失败。ViT-L 的 FP8 覆盖空间因此很小（约 6~18/96）。
- `proj` 不做 FP8：敏感度最低但 FP8 GEMM 本身几乎不提速（见下）。
- 不推荐 tensor 粒度（比 row 差）；b32/b128 在 sim 中并不稳定优于 row，且需要额外的块缩放 kernel。

## 2. 阶段 2：性能（`bench_fp8.py`，GPU 锁内）
### GEMM（bf16 `F.linear` vs FP8 `_scaled_mm` rowwise，M=2058；每 kernel GPU 时间，20 次内循环取均值）
| 形状 | bf16 | FP8 rowwise | FP8 tensorwise | 融合量化 kernel | 净加速（含量化） |
|---|---|---|---|---|---|
| ViT-L qkv 2058×3072×1024 | 70.3 µs（184 TF） | 46.1 µs（281 TF） | 53.5 | 4.3 µs | ×1.40 |
| ViT-L proj 2058×1024×1024 | 27.2（159 TF） | 22.9（189 TF） | 23.3 | 4.3 | ×1.00 |
| ViT-L fc1 2058×4096×1024 | 101.8（170 TF） | 58.4（296 TF） | 54.0 | 4.3 | ×1.62 |
| ViT-L fc2 2058×1024×4096 | 103.1（168 TF） | 54.3（318 TF） | 56.2 | 10.5 | ×1.59 |
| ViT-B qkv / proj / fc1 / fc2 | 43.5 / 21.5 / 62.0 / 59.9 | 27.6 / 22.8 / 37.7 / 42.0 | | 4.0 / 4.0 / 4.0 / 8.4 | ×1.38 / 0.80 / 1.49 / 1.19 |
- **`torch._scaled_mm` rowwise 在 sm_120 上可用**（M=2058/2258 都行），FP8 rowwise 达到 190~320 TFLOPS（bf16 实测 160~184），远低于 548 TFLOPS 的 8192³ 峰值——形状小，wave 利用率低。
- 量化开销：未融合的 torch 实现（absmax + 除 + clamp + cast，5 个 kernel）约 52 µs/次，**比 GEMM 本身还贵**；我写的融合 kernel（`impl/fp8_ext.cu`，warp/行，两遍 absmax + cast）4.3 µs（K=1024）/10.5 µs（K=4096，受 launch 限制的下界约 4 µs）。
- `ln_quant`（LN + 量化，一个 kernel）6.3 µs，比 torch LN 单独（9.2 µs）还快；`gelu_quant`（GELU + 量化）20.7 µs vs torch GELU 单独 8.4 µs（多出 12 µs 换来 fc2 的 FP8 输入）。
- 因此 FP8 净收益：qkv/fc1/fc2 ×1.2~1.65，proj ≈1（ViT-B 甚至变慢）→ 不量化 proj。
- 块缩放：**`_scaled_mm` 的 MXFP8（1×32 E8M0 缩放，需要 swizzled 缩放布局）在 sm_120 上可编译运行**：2048×4096×1024 为 49.5 µs（347 TFLOPS），误差 0.14%（=bf16 输出舍入）；**NVFP4（e2m1 + 1×16 e4m3 缩放）也能跑**：36.3 µs（474 TFLOPS）。`mx_test.py`。当时仅做 GEMM 可行性；真实模型的 MXFP8 接入与结果见 §6，NVFP4 模型级敏感度筛选见 §7。TileLang / CUTLASS 4.2 自写块缩放 GEMM 未做（cuBLAS 路径已经验证 sm_120 硬件路径可用）。

### 整模型（CUDA Graph，静态输入，GPU p50，同一次运行内配对比较；含 P3 融合 head）
| 模型 / 方案 | bf16 eager | bf16 graph | +融合 head graph | **+FP8 graph** | 净收益（FP8 vs bf16+head） |
|---|---|---|---|---|---|
| ViT-L 6 unit | 17.11 | 15.46 | 15.01 | 14.79 | -0.22 ms（-1.5%） |
| ViT-L 18 unit（不满足 3% 余量） | 17.12 | 15.03 | 14.60 | 13.70 | -0.90 ms（-6.2%） |
| ViT-B 14 unit | 8.25 | 5.79 | 5.48 | **5.10** | -0.38 ms（-7.0%） |
| ViT-B 20 unit | 7.39 | 6.02 | 5.47 | 4.92 | -0.55 ms（-10%） |
- CUDA Graph 本身：ViT-L 17.1 → 15.0 ms，ViT-B 7.4~8.3 → 5.8 ms（eager 在 ViT-B 上是 CPU launch 瓶颈）；融合 head 再省 0.4 ms；FP8 在此基础上再省 0.2~0.9 ms（正比于 unit 数）。
- FP8 eager 反而不划算（补丁里多了 python 开销：ViT-B 14 unit eager 11.5 ms，抖动很大）——FP8 只在 graph 里有意义。
- 按 gate 允许的推荐：ViT-B 14 unit → 5.10 ms（约 196 FPS 模型部分）；ViT-L 6 unit → 14.79 ms，收益很小。
- ViT-L 在 GEMM 之外仍有约 8 ms（attention 1.4、LN/cat/elementwise 等，见 Round 0/3/4 的分析），因此 FP8 的天花板本来就小：即使覆盖全部 qkv/fc1/fc2 也只省约 2.9 ms。

## 3. 结论
- **FP8 能用，但覆盖率很低**：ViT-L 只能覆盖 ~6 个 unit（满足 conf≤3%），整模型只省 ~1.5%；ViT-B 可覆盖 14~20 个 unit，省 7~10%。
- 原因是任务本身（head 像素少，IoU/confidence 对扰动敏感）而不是 kernel；bf16-autocast 已经吃掉了一部分误差预算（ViT-L 126 帧 conf 最大 1.5%）。
- 若一定要更大的 FP8 覆盖，需要 QAT/校准（clip 百分位、SmoothQuant 式平滑）或对敏感 unit（ViT-L L0.fc2、L10/L17 qkv…）保留 bf16；本轮没有做。

## 4. 遗留
- 未做 TileLang/CUTLASS 自写块缩放 GEMM；MXFP8 的 cuBLAS 路径后来接入真实模型，见 §6。
- FP8 GEMM 的 epilogue 融合（bias/GELU）没有做：`_scaled_mm` 只支持 bias，GELU 由 `gelu_quant` 独立完成。
- ViT-L 的“混沌”conf 指标使得推荐方案依赖于 126 帧 held-out 的验证；不同视频可能表现不同，上线前需要在目标视频上验证。
- `test_fp8.py` 中 kernel 部分正确性：ln_quant 与 torch 参考只有 0.09% 的 e4m3 码差 1 ulp（累加顺序），gelu_quant/quant_rows 一致。
- 期间发现并修复了 Round 7 `FastPrep` 的一个竞态：连续无同步调用 `fast_prep` 时，pinned buffer 可能在上一次异步 H2D 完成前被覆盖（现在 H2D 后 record event，覆盖前同步）。之前的 Round 7/8 计时路径每帧都有同步，不受影响。

## 5. 文件
- `impl/fp8sim.py`（fake-quant 模拟）、`impl/fp8_ext.cu`（`quant_rows` / `gelu_quant` / `ln_quant`）、`impl/fp8linear.py`（真实 FP8 Linear 补丁：`enable(model, units)` / `disable`）
- `sens.py`（阶段 1 扫描）、`greedy.py`（贪心 + held-out）、`heldout_real.py`（真实 FP8 的 126 帧验证）、`bench_fp8.py`（GEMM/整模型）、`mx_test.py`（MXFP8/NVFP4 可行性）、`test_fp8.py`、`summarize.py`
- 数据：`results.json`（`sensitivity`、`gemm_microbench`、`heldout_real_fp8`、各 `fp8-row-*` variant，格式同前）、`sens_*.jsonl`、`greedy_*.json/log`、`units_*.json`、`test_fp8.log`

## 6. 补做：真实 MXFP8 块缩放（sm_120，语义分割模型）

`impl/mx_ext.cu` 把激活/权重按 K 方向每 32 个 E4M3 值配一个 E8M0 二次幂 scale，并直接写入 `_scaled_mm` 要求的 swizzled 布局（无需另一次转置 kernel）。LayerNorm+量化和 GELU+量化各用一个 kernel；`impl/mxlinear.py::enable/disable` 只补丁选中的 `(layer, type)`，权重预量化、静态 batch 可 CUDA Graph capture。支持本任务的 K=768/1024/3072/4096（K 为 256 的倍数）；依赖 CUDA 13 的 sm_120a 与 PyTorch 2.12 的 E8M0 `_scaled_mm`，并非通用 GPU 路径。`test_mx.py` 在 M=160/2058/2258 上验证 FP8 编码和有效 scale 字节：普通量化/GELU 完全一致，LN 量化最多约 0.00005% 编码不一致；MX GEMM 对反量化 fp32 GEMM 的相对 MAE 0.1408%（bf16 输出舍入）。

真实 `/root/autodl-tmp/vitl-best.pth`、`vitb-best.pth` 和 `/root/autodl-tmp/1.mp4`；8 帧 gate 与 126 帧均对 fp32 eager 参考，含 Round 7 fused head。gate 要求 pooled IoU≥0.95 且逐帧 confidence 相对差≤5%；原 Round 6 的额外保守目标是全片 confidence≤3%。`b32` 贪心集来自旧 sim 文件；`row` 推荐集是原 Round 6 的上线候选。`results.json::mx_fp8` 保存原始指标。

| 模型 / MX units | 8 帧 pooled IoU / conf max | 126 帧 pooled IoU / min-frame IoU / conf max | 结论 |
|---|---|---|---|
| ViT-L b32-10（qkv1/proj6/fc1 3） | 0.9838 / **13.10%** | 未测 | gate 失败 |
| ViT-L row-6 的 units（fc1 1/fc2 5） | 0.9852 / 1.52% | 0.9891 / 0.9386 / 2.72% | 精度通过 |
| ViT-B b32-12（qkv1/proj5/fc1 3/fc2 3） | 0.9754 / 0.51% | 0.9786 / 0.9526 / 1.01% | 精度通过，但多为无速度收益的 proj |
| ViT-B row-14 的 units（qkv3/fc1 5/fc2 6） | 0.9634 / 2.43% | 0.9718 / 0.9171 / 2.43% | 精度通过，余量更小 |

同 units 的既有真实 row-FP8 对照：ViT-L 6 为 8 帧 0.9888/1.60%、126 帧 pooled 0.9897/conf 2.7%；ViT-B 14 为 8 帧 0.9825/0.53%、126 帧 pooled 0.9802/conf 1.35%。MX 的 ViT-B 14 比 row-14 的全片 IoU 低 0.0084，最大 confidence 偏差高 1.08 个百分点。ViT-L b32-10 的 sim 曾预报全片 pooled IoU 0.9906，但 real kernel 的 gate confidence 失败，不能用 sim 结果代替实测。

整模型 CUDA Graph GPU p50，batch=2，含 fused head；同一次 `eval_mx.py` 运行内持 `common/bench.py::gpu_lock` 做 bf16 → row → MX → bf16 配对（ViT-B 14 另做 bf16 → MX → row → bf16 反序）。单位 ms：

| 模型 / units | bf16 前/后 | row-FP8 | MXFP8 | MX 相对同 units row |
|---|---:|---:|---:|---:|
| ViT-L 6 | 14.564 / 15.006 | 14.669 | 14.563 | -0.106（小于基线 0.442 漂移） |
| ViT-B b32-12 | 5.460 / 5.689 | 5.447 | 5.353 | -0.094（基线漂移 0.229） |
| ViT-B row-14 | 5.641 / 5.656 | 5.270 | 5.115 | -0.156（基线漂移 0.014） |
| ViT-B row-14，反序 | 5.457 / 5.687 | 5.297 | 5.141 | -0.156（基线漂移 0.231） |

**决策**：不改默认入口。ViT-L 6 的额外量化没有可分辨的整模型收益；ViT-B b32-12 的 0.09 ms 相对优势与约 4% 的 bf16 基线漂移同量级。ViT-B 14 的 MXFP8 在正反序均比相同 units 的 row-FP8 快约 0.16 ms（约 3%），但精度从 0.9802 降至 0.9718、confidence 最大差从 1.35% 升到 2.43%。这一模式仅保留为可显式启用的实验选项；Round 9 当前默认宽图 `r5_v1` 模型入口本身比 `lightly+FP8` 快，也没有 MX epilogue。面向其他视频的泛化验证未做。

## 7. 补做：NVFP4 W4A4 模型级敏感度筛选

`impl/nvfp4sim.py` 模拟 E2M1 值 `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`、K 方向每 16 元素一个 E4M3 block scale，以及每个 operand 一个 fp32 全局 scale；反量化后做 fp32 GEMM，再按 bf16-autocast 输出舍入。权重静态量化，激活动态量化。`eval_nvfp4.py` 在共享 GPU 锁内使用真实 ViT-L/B 权重、8 帧真实视频、fp32 eager 参考和 Round 7 fused head；`results.json::nvfp4_screen` 保留每个候选的 IoU、confidence、logit MAE 与选择的 unit。**这是假量化精度试验，没有测出真实 FP4 整模型时延。** `mx_test.py` 的 36.3 µs 是随机打包值的单个 ViT-L fc1 形状 GEMM 计时，也不包含真实量化开销。

| 模型 / MLP units | 8 帧 pooled IoU | confidence 最大相对差 | 正式 gate | 将所选 bf16 GEMM 当作零耗时的理想节省上界 |
|---|---:|---:|---|---:|
| ViT-L bf16 基线 | 0.9930 | 0.45% | 通过 | — |
| ViT-L L5.fc1 | 0.9853 | 3.82% | 通过 | 0.102 ms |
| ViT-L L20.fc2 | 0.9897 | **7.70%** | 失败 | 0.103 ms |
| ViT-L L23.fc2 | 0.9926 | 1.41% | 通过 | 0.105 ms |
| ViT-L 推荐序列前 3 个 MLP | 0.9796 | **13.72%** | 失败 | 0.310 ms |
| ViT-L 原 row-6 units | 0.9767 | **10.11%** | 失败 | 0.619 ms |
| ViT-B bf16 基线 | 0.9921 | 0.11% | 通过 | — |
| ViT-B L1.fc2 | 0.9861 | 0.28% | 通过 | 0.060 ms |
| ViT-B L11.fc1 | 0.9872 | 0.17% | 通过 | 0.062 ms |
| ViT-B L8.fc2 | 0.9745 | 0.17% | 通过 | 0.060 ms |
| ViT-B 推荐序列前 3 个 MLP | 0.9767 | 0.43% | 通过 | 0.184 ms |
| ViT-B 推荐序列前 6 个 MLP | **0.9520** | 0.83% | 勉强通过 | 0.366 ms |
| ViT-B 原 row-14 中全部 11 个 MLP | **0.9383** | 0.53% | 失败 | 0.670 ms |

正式 gate 是 pooled IoU≥0.95 且逐帧 confidence≤5%；Round 6 推荐配置另要求 IoU≥0.975、confidence≤3% 的余量。ViT-L 中较稳的只有 L23.fc2 单层，零 GEMM 的理想上界也仅约为 15 ms 模型时间的 0.7%；L5.fc1 虽过正式 gate，却已超过 3% 的保守 confidence 线。ViT-B 三层仅以 0.0017 的 IoU 余量达到内部目标，零 GEMM 上界 0.184 ms（约 5.6 ms 模型时间的 3.3%）；六层距离正式 IoU 线仅 0.0020。真实 FP4 GEMM、动态量化、scale 布局和输出舍入均会消耗时间或改变结果，因此**这些候选缺少足够的精度和净收益余量，不投入真实 NVFP4 模型接入**。这不排除重新校准或 QAT 后出现可行配置；当前数据仅覆盖此权重、视频与量化方案。
