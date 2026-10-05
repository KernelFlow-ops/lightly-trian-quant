# Round 12：FP16 整模型栈（r12）+ 统一测速脚本

> 日期 2026-10-05。本轮把 PLAN_v2 的 R12/R13 合并成一个集成轮（用户要求少测试、早出结果）。
> 所有数字都是本机实测，GPU 任务持双锁。R0–R11 没有改动，只读 import。

## 1. 做了什么

`r12` = round11 的 bf16 等价小项（patch-embed GEMM、dw_ln、tail、post_a2）+ FP16 骨干：

| 部分 | 实现 | 来源 |
|---|---|---|
| LN → fp16 | `impl/ln16.cu`：R5 `add_ln` 的拷贝，输出改成 fp16 | 本轮 |
| qkv / proj / fc1 / fc2 | `impl/mg16*.{cu,py}`：fp16 操作数，`mma.sync … f16.f16.f16.f16`，每个 k-tile 提升到 fp32；R5 的 RoPE+布局 / GELU / γ+fp32 残差 epilogue 不变 | round10 `mg_f16acc` 原型的拷贝（只改了 build 目录），tile 表 `tuned16.json` |
| attention | `impl/attn16o.cu`：FP16 累加 FA2，kind 4（S 和 PV 都用 f16 累加，每个 64-key tile 提升到 fp32）；第 0、1 层 Q 预缩放 1/8（round11 的 `a16pol` 策略）；输出直接写 fp16 | round11 `attn16.cu` 的拷贝，只改了输出 dtype |
| head ConvT（`+head16`） | 两个 ConvT GEMM 走 `mg16` 的纯 GEMM epilogue，bf16 输出，`dw_ln` 不变 | 本轮 `stack12.py::FastHead12` |

生产者到消费者之间全部直接传 fp16，没有 cast kernel。残差流、LN 统计量、RoPE、softmax 统计量、LayerScale 仍是 fp32。

入口：`impl/stack12.py::make_variant(name, model, "r12[+nopatch][+head16]")`，返回 `(model_fn, Pipeline11)`。

## 2. 结果

### 2.1 同场 A-B-A-B 整段视频（`bench12.py`，126 帧 × 3，2 轮，inflight=2）

| 模型 | r5_v1（R9） | r11 | r12 | r12+head16 |
|---|---:|---:|---:|---:|
| ViT-L FPS | 96.34 | 99.64（去掉 patch 项） | 122.50（+27.2%） | **124.30（+29.0%）** |
| ViT-B FPS | 226.96 | 242.18 | **273.25（+20.4%）** | 271.82 |

### 2.2 只测模型段（`speed12.py`，CUDA graph 回放，3 轮交替取中位数）

| 模型 | variant | model ms | 整帧 graph ms | GPU 上限 FPS | 视频 FPS |
|---|---|---:|---:|---:|---:|
| ViT-L | r5_v1 | 9.676 | 9.846 | 101.6 | 96.5 |
| ViT-L | r12+head16 | 7.353（1.32×） | 7.498 | 133.4 | 125.7 |
| ViT-B | r5_v1 | 3.672 | not measured | — | 227.0 |
| ViT-B | r12 | 2.719（1.35×） | not measured | — | 273.3 |

ViT-B 的视频 FPS 已经被 host / cv2 解码限制（模型 2.72 ms，每帧实际 3.66 ms）。之后比较模型速度看 `model_ms` / `frame_ms`。

### 2.3 精度（只作记录；2026-10-05 起用户决定不再用精度门卡低精度方案）

| 模型 / variant | 8 帧：IoU / conf | 126 帧：IoU / 最差帧 IoU / conf 最大 / >3% 帧数 | graph==eager |
|---|---|---|---|
| ViT-L r5_v1 | 0.9946 / 0.13% | 0.9947 / 0.9846 / 1.04% / 0 | 是 |
| ViT-L r12 | 0.9965 / 2.52% | 0.9964 / 0.9792 / 2.52% / 0 | 是 |
| ViT-L r12+head16 | 0.9964 / 2.52% | 0.9964 / 0.9796 / 2.52% / 0 | 是 |
| ViT-L r12+nopatch | 0.9959 / 3.70% | 0.9964 / 0.9718 / 3.70% / 1 | 是 |
| ViT-B r5_v1 | 0.9943 / 0.21% | 0.9931 / 0.9803 / 0.43% / 0 | 是 |
| ViT-B r12 | 0.9968 / 0.18% | 0.9954 / 0.9854 / 0.34% / 0 | 是 |

fp16 操作数比 bf16 多 3 位尾数，r12 对 fp32 的 IoU 和 mask-score MAE（ViT-L 4.7e-6 对 6.4e-6，ViT-B 1.05e-5 对 1.45e-5）都比 r5_v1 好。

## 3. 测速脚本（round13 / round14 用它）

```bash
cd /root/lightly-train/infer-opt/round12_lowp_stack
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
python speed12.py --model vitl16 --variants r5_v1,r12+head16,<新 variant> --plugin <round 目录>:<模块名> --tag <名字>
```

- 插件模块要定义 `PREFIXES = ("r13",)` 和 `make_variant(name, model, variant) -> (model_fn, pipeline_class)`。
- 输出：`model_ms`、`frame_ms`、`fps_gpu`、`fps_video`，以及只作记录的 sanity（输出有限、mask logit 对第一个 variant 的 cos）。
- 第一个 variant 是对照。脚本自己持双锁。

## 4. 文件

`impl/stack12.py`、`impl/ln16.cu`、`impl/attn16o.cu`、`impl/mg16*.{cu,py}` + `tuned16*.json`、`bench12.py`（门 + 配对）、`speed12.py`（只测速度）、`tests/smoke12.py`、`results.json`、`results_speed.json`、`logs/`。

## 5. 没做

- `mg16` 原型比 cuBLAS fp16-acc 慢 15–25%（217–260 对 285–344 TFLOPS），主循环还能再调。
- 没有给合批（M=4116）重新调 tile；没做解码 / host 侧优化；没做单次宽图推理。
