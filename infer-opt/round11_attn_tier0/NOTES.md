# Round 11：不改精度的小项（r11）+ attention 候选矩阵

> 状态：进行中（边做边写；被打断后续做请先读 `results.json` 与本文件）。日期 2026-10-04。
> 环境：RTX 5090 (sm_120a)，torch 2.12.1+cu130，所有 GPU 计时/精度任务持 `common/lock2.py::gpu_lock2()`，编译在锁外。
> 进度（23:00）：Part 1 完成（kernel、门 M/H/G、同场配对、nsys 都已测）；Part 2：候选矩阵速度/算子精度/整模型门基本完成，最后一批（Sage2 修流后的速度与门、a16pol、attention 对整段视频 FPS）在跑，结果见 6、7 节与 `results.json`。

## 1. 目标与假设

- 第一部分：在 `r5_v1` 上叠 4 个 bf16 等价小项，做成 `r11` model_fn（+ 一个 `VideoPipeline` 子类，因为 post_a 换了），过门 M+H+G，同场 A-B-A-B 整段视频配对，FPS ≥ +2% 才发布。
  1. patch-embed：cuDNN `implicit_convolve_sgemm`(121-131 µs) + cat + cast + 每次调用的权重 cast → 一个隐式 im2col GEMM，直接写 fp32 残差流 x0。
  2. head `dw_ln_kernel`：2 次共 ~270 µs，目标 2×。
  3. `post_a`：106-138 µs。
  4. 尾部：`bb.norm` + 切片/cast + class_head + 每次调用的 autocast 权重 cast。
- 第二部分：attention 候选矩阵（速度 + 精度），给出 A0/A1/A2/A3 的推荐和 R13 生产内核需求。

## 2. 基线 nsys（r5_v1，ViT-L，30 帧平均，`profile/vitl16_r5v1_cuda_gpu_kern_sum.csv`，本机本时段）

总 GPU kernel 时间 9.742 ms/帧，204 kernel/帧。

| 项 | µs/帧 | 备注 |
|---|---:|---|
| R5 GEMM（gemm_kernel 48 + main_kernel 48） | 4380 + 2422 | R10/R12 的范围 |
| flash_fwd | 1450 (24×60.4) | |
| head ConvT GEMM（cutlass 256x128，2 次） | 455 | 范围外（R10/R12） |
| `dw_ln_kernel`（2 次） | 274 | 项 b |
| add_ln（48 次） | 225 | |
| `post_a_kernel` | 138 | 项 c |
| 隐式卷积 `implicit_convolve_sgemm` | 131 | 项 a |
| mask bmm（cutlass 64x256） | 104 | |
| class_head（cutlass_75 align1 GEMM） | 25 | 项 d |
| mask_head 3×wmma GEMM | 23 | |
| 权重 cast 10×bfloat16_copy | 21 | 项 a/d |
| `.to(bf16)` direct_copy（head 输入） | 20 | 项 d |
| prep / post_b | 20 / 16 | |
| cat（prepare_tokens） | 9 | 项 a |
| 其余小 kernel（LN 7.7、bf16 add 4.7、copy …） | ~25 | |

## 3. 改动（文件都在 `impl/`，旧轮次只读 import 或复制后改名注明来源）

| 项 | 文件 | 做法 |
|---|---|---|
| a patch-embed | `patch_embed.cu/.py` | 隐式 im2col GEMM（mma.sync m16n8k16 bf16，cp.async 多级流水，ldmatrix 异或 swizzle，结构抄 R5 `mg_core.cu`）。A 操作数直接从 `[B,3,512,512]` 取：K 顺序 (ch,ky,kx)，一个 16 像素行 = 32 连续字节 = 2 个 16B cp.async；W = conv 权重 `[C,768]`。epilogue：`bf16(bf16(acc)+bf16_bias)`（与 torch 的 "bf16 conv + 单独 bf16 加 bias" 位级一致），以 fp32 写 `x0[B,1029,C]`，cls/register 行（`cls_token+0*mask_token`、`storage_tokens`）由每张图第一个 M tile 的 CTA 一并写入。选 (a) 隐式 im2col，**保持 R9 model_fn 接口**，没有加 `model_fn_patches`。 |
| b head dw_ln | `head_ext11.cu`（`dw_ln_launch`） | 复制 R7 `head_ext.cu::dw_ln_kernel` 改：通道按 256 一遍（smem 只放 `(TH+2)(TW+2)×256` 通道的 halo，2-4 block/SM 互相重叠 load 和 compute 阶段）；每个 warp 对 PPW 个相邻像素滑窗算 3×3（权重解包一次摊到 PPW 像素）；dwconv 结果用 bf16x2 打包放寄存器（参考本来就在 LN 前取整到 bf16）；halo 4 个向量一批发出再算 GELU。**与 R7 输出逐位相同**（精确 erff，tap 与求和顺序不变）。 |
| c post_a | `postfuse_ext11.cu`（`post_a2_launch`） | 新 kernel（语义同 R7 `post_a`）：thread = 一个 4×4 crop 像素块；x4 双线性上采样时 16 个像素只用 3×3 源 logits（行/列 by-1,by,by+1 边界 clamp，权重固定 {.625,.875,.125,.375}），每 (块,query) 9 次 load + 可分离 lerp；sigmoid = `rcp.approx(1+ex2.approx(-v*log2e))`；q 拆到 NG=8 个 warp（warp=q 组，lane=块，合并访存），smem 归约；两个 crop 用两次 launch（crop0 存，crop1 对重叠区做 `(s0+s1)*0.5`），确定性。只支持 2 crop、crop==4h==4w、Q≤128、3 类；`Pipeline11` 子类 `_post_launch` 不满足时回退 R7 post_a。可选 `skip<eps>`：softmax 后 `p0+p1<eps` 的 query 不参与（确定性的活跃表）。 |
| d 尾部 | `head_ext11.cu`（`tail_ln_launch`）+ `head11.py` | 一个 kernel 做 `bb.norm`（fp32 LN）+ bf16 cast + 只取 head 需要的行（q 行→`[B*Q,C]`，patch 行→连续 `[B*1024,C]`，cls/register 行跳过）+ class_head（fp32 点积，bf16 输出）；`mask_head` 3 个 Linear 用预先 cast 好的 bf16 权重（无逐次 autocast 权重 cast，结果与 autocast 相同）。省掉：LN 7.7 + `.to(bf16)` direct_copy 20 + cutlass_75 align1 class-head GEMM 25 + 3 个 q cast 9 + 8 个权重 cast ~17 ≈ 79 µs，换成 1 个 kernel。 |
| model_fn / 流水线 | `r11_model.py`、`r11_common.py` | `make_variant(name, model, "r11[+patch|head|tail|post|skip<eps>|none|attn=<name>]")`；`r11` = patch+head+tail+post。block = r5 `fused_block` 的拷贝，attention 可插拔（Part 2）。 |

## 4. 分析过程与已完成的单项测量（本机，GPU 锁内，CUDA graph 回放）

**a. patch-embed**（`tests/test_patch.py`，ViT-L，B=2）
- 数值：与 `prepare_tokens_with_masks`(bf16 autocast)+`.float()` 比，2107392 个元素中 344 个差 1 个 bf16 ulp（fp32 累加顺序，0.016%），max_abs 0.0156，前缀行逐位相同，cos 1.00000000。（第一版只取整一次，26.8% 元素不同，原因是 torch 先把卷积结果取整成 bf16 再单独加 bias 并再取整；已按此改。）
- 时间：参考路径（卷积+bias add+cat+float+权重 cast，graph 内合计）**161.9 µs → 28.2 µs**；6 个 tile 配置扫描都在 27.3–31.0 µs（`logs/tune_patch_vitl16.log`），默认 BM=64,BN=64,BK=32,ST=3 已足够，不再调。

**b. dw_ln**（`tests/test_dwln.py`，随机但量级合理的输入，与 R7 kernel 逐元素比；`logs/dwln_sweep2.log`）

| 配置 TH×TW, PPW, MINB | C=1024 stage1 (Hs=32) µs | stage2 (Hs=64) µs | C=768 stage1 | stage2 | 与 R7 输出差异 |
|---|---:|---:|---:|---:|---|
| R7 原 kernel | 57.3 | 201.2 | 40.1 | 136.7 | — |
| 4×8, 4, 2 | 36.9 | 148.8 | 29.4 | 112.6 | 0 |
| **4×8, 2, 2（默认）** | **34.8** | **139.2** | **27.3** | **107.2** | 0（逐位相同） |
| 2×8, 2, 3 | 43.8 | 193.1 | 36.1 | 156.3 | 0 |
| 2×16, 4, 2 | 40.6 | 159.0 | 32.3 | 121.4 | 0 |
| 4×8, 4, 2 + FAST_GELU（A&S erf） | 42.5 | 169.8 | 33.5 | 129.3 | 6e-5 的元素差 1 ulp |

- dw_ln 两次合计 ViT-L 258.5 → 174.0 µs（-84.5 µs，1.49×），ViT-B 176.8 → 134.5 µs（-42 µs，1.31×）；**没有达到计划的 2×**。FAST_GELU 反而更慢，说明瓶颈不是 erf 的 ALU 指令数；stage 2 要读写各 67 MB（`G` 与输出 `U`），DRAM 下限约 90-100 µs，当前 139 µs ≈ 1.4× 下限。再往下需要把 `U` 的写回和 mask bmm 融合（见"未做"）。

**c. post_a2**（`tests/test_post.py`，真实模型输出，3 个评测帧，ViT-L，与 R7 `post_a` 的 lr 比；`logs/test_post_vitl16.log`）
- 时间：R7 post_a 134–143 µs → **31.6 µs**（4.4×），lr max_abs 3.0e-7–6.0e-7（fp32 求和顺序 / approx 指令）。
- query 跳过（时间 µs / lr 最大绝对误差，eps = 1e-4, 1e-3, 1e-2）：帧0 8.4/5.6e-4, 8.2/9.1e-4, 8.3/9.1e-4；帧3 19.9/4.2e-4, 9.1/1.7e-3, 8.2/8.6e-3；帧7 21.0/3.4e-3, 8.6/5.8e-3, 8.1/1.0e-2。活跃 query（p0+p1≥1e-3）每 crop 只有 2–9 个 / 100。误差上界 = 被跳过 query 的 (p0+p1) 之和（帧7 eps=1e-3：crop0 0.024、crop1 0.0058），实测都在上界之内。默认**关**，作为单独标志 `skip<eps>`，需单独过门。

**d. r11 与 r5_v1 的 logits 直接对比**（`tests/test_r11_model.py`，ViT-L，评测帧 0 和 7，`logs/test_r11_model_vitl16.log`；ml = mask logits [2,100,128,128]，|ml| 均值约 75）

| variant | ml max_abs | ml mean_abs | cos | class logits max_abs |
|---|---:|---:|---:|---:|
| r11+none（r5_v1 的重实现） | 0 | 0 | 1 | 0 |
| r11+head（只换 dw_ln） | 0 | 0 | 1（逐位相同） | 0 |
| r11+post（只换 post，model_fn 不变） | 0 | 0 | 1 | 0 |
| r11+tail | 0.5 | 5e-4 / 2.8e-4 | 0.99999998 | 0 |
| r11+patch | 1.5 | 0.081 / 0.085 | 0.9999969 | 0.0625 |
| r11（全开） | 1.5 | 0.081 / 0.085 | 0.9999969 | 0.0625 |

patch 项的 0.016% 元素差 1 个 bf16 ulp 经 24 层放大为 mean_abs 0.08（相对 |ml|≈75 为 1e-3，量级等同 bf16 噪声）；最终是否过门由 gate 决定。

## 5. Part 2 已完成的前置工作
- 编译（tmp 目录，不 pip install）：SageAttention2（`sage2/`，`TORCH_CUDA_ARCH_LIST=12.0`，sm_120 走 sm89 内核）、SageAttention3（`sage3/`，CUTLASS 4.2 取自 `cutlass_library/source`）、Spark `attention_fp8.cu`（D=64、128 都支持；`impl/spark_fp8_bind.cpp`）均已编译成功。
- 自写 **FP16 累加 FA2 原型** `impl/attn16.cu`（基于 train-opt r12 `attn_fwd.cu`：TMA + mbarrier，8 个 consumer warp × 16 行，64-key tile，尾块精确；输入布局直接用 `[3,B,N,H,64]`，无 RoPE/mask/lse）。5 种 KIND：0 bf16/f32acc，1 fp16/f32acc，2 S 用 f16 累加，3 PV 用 f16 累加（每 64-key tile 提升到 fp32 O），4 两者。首轮结果（`logs/test_attn16_a.log`，B=2，随机输入）：

| 形状 | flash | k0 | k1 | k2 (S f16) | k3 (PV f16) | k4 (both) |
|---|---:|---:|---:|---:|---:|---:|
| H16 N1029 µs（TFLOPS） | 60.4 (144) | 55.4 | 55.3 | 43.8 (198) | 46.4 | **34.6 (251)** |
| H16 N1129 | 61.2 (171) | 60.2 | 60.3 | 48.0 | 50.3 | **37.5 (279)** |
| H12 N1029 | 60.5 (108) | 55.1 | 55.1 | 42.4 | 45.7 | **34.2 (190)** |
| H12 N1129 | 60.9 (129) | 59.7 | 59.6 | 46.1 | 49.6 | **37.0 (211)** |

  精度（对 fp64，rel_l2）：随机输入 flash 2.2e-3，k0 2.2e-3，k1 1.7e-3，k2 1.8e-3，k3 1.7e-3，k4 1.8e-3（fp16 的 P 比 bf16 的 P 更准）；**尖峰 scale-8 输入：flash/k0/k1/k3 8.9e-4，k2/k4（S 用 f16 累加）7.1e-3（max_abs 0.3，对比 0.0155）**。→ PV 用 f16 累加没有精度代价；S 用 f16 累加在极端尖峰下代价明显，在真实数据（`qkv_real_*.pt`）上的结果决定能否用（待跑）。

## 6. Part 1 结果（r11 = patch + head + tail + post；`r11+head+tail+post` = 去掉 patch-embed 的版本）

### 6.1 每帧 GPU 时间（nsys，30 帧平均，`profile/*_cuda_gpu_kern_sum.csv`，`nsys_cmp.py` → `profile/nsys_cmp_vitl16.json / vitb16.json`）

| 项（µs/帧） | ViT-L r5_v1 → r11 | 节省 | ViT-B r5_v1 → r11 | 节省 | 计划估计 L/B |
|---|---|---:|---|---:|---|
| a patch-embed（cuDNN conv → GEMM；含掉 cat/cast 小 kernel 中的一部分） | 131.3 → 27.8 | **-103.5** | 95.7 → 25.3 | **-70.4** | 130 / 100 |
| b dw_ln（2 次） | 274.4 → 189.8 | **-84.7** | 193.6 → 145.6 | **-47.9** | 130 / 90 |
| c post_a | 138.0 → 33.0 | **-105.0** | 127.4 → 33.0 | **-94.4** | 50 / 50 |
| d 尾部 + 其余小 kernel（LN、`.to(bf16)`、class_head、权重 cast、q cast …） | 117.1 → 28.0 + tail_ln 6.4 | **-82.7** | 149.4 → 74.1 + tail_ln 5.0 | **-70.3** | 20 / 20 |
| R5 GEMM / flash / ConvT GEMM / mask bmm / add_ln（未动，run-to-run 噪声） | 8835 → 8872 | +37 | 3195 → 3188 | -7 | — |
| **合计 GPU kernel 时间** | **9742 → 9405** | **-336 µs（-3.5%）** | **3804 → 3512** | **-292 µs（-7.7%）** | |
| kernel 数/帧 | 204 → 185 | | 120 → 101 | | |

dw_ln 没有到 2×（1.49× / 1.31×）；post_a 和尾部比估计多省（post_a 4.2×；尾部顺带去掉了每次调用的 autocast 权重 cast 和 class_head 的 cutlass_75 align1 GEMM）。

### 6.2 同场 A-B-A-B 整段视频配对（`bench11.py`，126 帧 × 3 loop/轮，inflight=2，稳态 FPS；`other_gpu_procs()` 开头与结尾均为空）

| 模型 | 实验 | r5_v1 | `r11+head+tail+post` | `r11`（全开） | 
|---|---|---:|---:|---:|
| ViT-L | paired1（2 轮，r5_v1 vs r11） | 97.62（98.67 / 96.57） | — | **101.97（101.62 / 102.32）= +4.46%** |
| ViT-L | paired_nopatch（3 轮，三者交替） | 96.25（96.49/96.25/96.24） | 98.51 = **+2.35%** | 99.83 = **+3.72%** |
| ViT-B | paired1（2 轮） | 226.13（228.1 / 224.2） | — | **239.64 = +5.97%** |
| ViT-B | paired_b3（3 轮，三者交替） | 225.23 | 234.68 = **+4.20%** | 239.87 = **+6.50%** |

（各轮原始数字在 `results.json["bench11"]`、`logs/bench11_*.log`。ViT-L 的 FPS 增益 3.7–4.5% 略高于 nsys 的 -3.5% kernel 时间，host 侧小 kernel 少了 19 个也有一点贡献。）

### 6.3 门（`results.json["bench11"]`；M = 8 帧 vs fp32；H = 126 帧 vs fp32，`ref126.py` 缓存；G = model_fn graph==eager 逐位 + 整条 pipeline graph vs 非 graph 126 帧 mask 逐位）

| 模型 / variant | M：pooled IoU / min 帧 IoU / conf max | H：pooled IoU / min 帧 IoU / conf max / conf>3% 帧数 / 5% 帧数 | mask-score MAE | G |
|---|---|---|---|---|
| ViT-L r5_v1 | 0.9946 / 0.9897 / 0.13% | 0.9947 / 0.9846 / 1.04% / 0 / 0 | 6.4e-6 | 通过 |
| ViT-L `r11+head+tail+post` | 0.9946 / 0.9897 / 0.13% | 0.9947 / 0.9846 / 1.04% / 0 / 0 | 与 r5_v1 同量级 | 通过 |
| ViT-L `r11`（全开） | 0.9922 / 0.9653 / **4.47%** | 0.9946 / 0.9653 / **4.48%** / **1** / 0 | 6.5e-6 | 通过 |
| ViT-B r5_v1 | 0.9943 / 0.9875 / 0.21% | 0.9931 / 0.9803 / 0.43% / 0 / 0 | 1.45e-5 | 通过 |
| ViT-B `r11` | 0.9935 / 0.9890 / 0.31% | 0.9925 / 0.9735 / 0.62% / 0 / 0 | 1.47e-5 | 通过 |

按 PLAN_v2 的门 M+H+G 全部通过（safe 档要求），但 **ViT-L 全开 r11 的 conf 余量只剩 4.48% / 5%**。单项隔离（`logs/test_r11_model_vitl16.log`、`logs/bench11_sens1.log`）：
- `head`、`post` 单独：mask logits 与 r5_v1 逐位相同（dw_ln 与 post_a2 在 fp32 求和顺序之外保持语义；post_a2 的 lr 与 R7 差 3e-7）。
- `tail`（LN 求和顺序不同）：M conf 0.13%、H 1.04%，与 r5_v1 同。
- `patch`：x0 中 344/2107392 个元素（0.016%）与 cuDNN 参考差 1 个 bf16 ulp（fp32 累加顺序），经 24 层放大使 ViT-L 那个已知的混沌帧（README：第 7 帧）conf 误差跳到 4.47%。
- 对照探针（`r11+none+jit<k>`：在 r5_v1 等价路径的 x0 上加种子固定的 0.017% 元素 ±1 ulp 抖动，**不含 patch kernel**）：jit1 M 2.08%/H 2.51%；jit2 2.09%/3.21%；jit3 M 4.45%（最小帧 IoU 0.962）；jit4 2.10%。`patch+jit1/jit2`：M 0.15%/0.53%。→ 4.5% 落在"对 ulp 级扰动的混沌帧"的正常分布内，不是 patch kernel 的 bug（patch kernel 对 ViT-B 完全无事）。patch kernel 的另外 3 个 tile 配置（`pcfg1-3`）k 方向累加顺序相同，结果与默认配置逐位相同，不构成独立探针。

### 6.4 发布决定（FPS ≥ +2% 才发布）
- **ViT-B：发布 `r11`（全开）**：+5.97% / +6.50%，门 M/H/G 余量充足。
- **ViT-L：发布两个条目**：`r11+head+tail+post`（+2.35%，门指标与 r5_v1 完全相同，建议作 safe 档默认）和 `r11`（+3.72% ~ +4.46%，门形式上通过但 conf 余量薄，建议作 fast 档或在 R14 视余量选择）。选择理由：ViT-L 对 ulp 级扰动的混沌，patch-embed 只多 +1.4% FPS，却把门余量从 1.0% 降到 4.5%。
- 可选 `skip<eps>`（post_a2 跳过低概率 query）：kernel 31.6 → 8–20 µs（多省 12–23 µs/帧，约 0.15–0.25%）。**低于 +2%，且需要单独过门，未发布、未做整段视频门**；误差界见 4.c。

