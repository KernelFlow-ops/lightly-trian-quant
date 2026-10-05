# Round 2：CUDA Graph + torch.compile

文件：`graphed.py`（`GraphedModel(model_fn, static_shape, dtype)`：side stream warmup，静态输入 buffer，`copy_` + `replay`，返回静态输出）、`run.py`、`bench_paired.py`、`results.json`、`profile/`。被捕获对象是 R1 的 `OptEoMT`（cudnn SDPA，fp32 残差）；capture 约 0.05-0.5 s。

## 可信数据：同进程内 eager/graph 配对（交错 6 轮，取最佳轮 wall p50，ms）
| | eager | CUDA graph | 节省 |
|---|---|---|---|
| ViT-L model | 13.03 | 12.43 | 0.60（4.6%） |
| ViT-L model_lowres | 12.97 | 12.35 | 0.62 |
| ViT-L e2e（基线 prep/post） | 54.3 | 53.5 | 0.8 |
| ViT-B model | 5.13 | 4.79 | 0.34（6.6%） |
| ViT-B model_lowres | 5.03 | 4.69 | 0.34 |
| ViT-B e2e | 46.9 | 46.2 | 0.7 |

graph 版输出与 eager 位一致（test_model_tl 中 torch.equal），gate 与 eager 相同（ViT-L 2.11%，通过）。
**第一次跑的数据（model 24-28 ms，graph 比 eager 慢）作废**：当时 GPU 上有其他 worker 锁外的正确性/调参任务并占满 CPU；已移到 `old_contaminated/`，改用配对 + 多轮取最佳。

## nsys 证据（ViT-L，每帧，`common/nsys_analyze.py`；`profile/*_summary.md`）
| | eager（R1） | CUDA graph | 基线 R0 |
|---|---|---|---|
| model 段 CPU 时长 | 7.75 ms | **0.82 ms**（1 次 cudaGraphLaunch，0.79 ms） | 20.6 ms |
| CPU launch API | 357 cudaLaunchKernel（1.42 ms）+ 104 cuLaunchKernel（0.36 ms） | 21 次 cudaLaunchKernel（0.14 ms）+ 1 次 graph launch | 1220 次，5.05 ms |
| model 段 GPU span | 13.07 ms | 12.56 ms | 21.5 ms |
| model 段 GPU 忙碌率 | 93% | 97% | 71% |
| kernel 总时间 | 12.12 ms | 12.15 ms | 15.35 ms |

graph 消掉约 1.8 ms 的 CPU launch API 时间和约 0.5 ms 的 GPU 空隙；kernel 本身耗时不变。**R1 之后模型已经是 GPU-bound（CPU 7.75 ms < GPU 13.07 ms），所以 graph 只省 0.5-0.6 ms**；基线在 launch-bound 时收益大得多。graph 的价值是把 model 段 CPU 占用降到 0.8 ms，并在后续融合把 kernel 数继续降低后仍保持无空隙。

## torch.compile（对 OptEoMT.forward_backend，fullgraph，cudnn SDPA）
| 模式 | ViT-L model | kernels | conf 误差 / gate | 编译耗时（冷 / 缓存命中） | ViT-B model | kernels | 编译耗时（冷 / 缓存） |
|---|---|---|---|---|---|---|---|
| max-autotune-no-cudagraphs | 10.40 | 241 | 4.86% 通过（勉强） | 80 s / 6 s | 4.05 | 134 | 59 s / 5 s |
| max-autotune（自带 cudagraphs） | 10.25 | 242 | 1.69% 通过 | 43 s / 9 s | 3.92 | 135 | 23 s / 8 s |
| reduce-overhead | 10.66 | 246 | 2.72% 通过 | 16 s / 4 s | 4.10 | 141 | 9 s / 2 s |

对照：手写 eager 12.93 / 5.13，手写 graph 12.43 / 4.79。Inductor 把 LN、GELU、RoPE、残差等 elementwise 融合，kernel 数减半，model 比 graph 版再快约 2.1 ms（ViT-L）/ 0.8 ms（ViT-B）。代价：形状变化要重编译；no-cudagraphs 版 4.86% 很贴近门槛（同一 frame 7 敏感性，见 R1 NOTES）。
ViT-B compile 变体的 e2e 显示 59-62 ms（对比 eager 47 ms）。我没有查明原因（怀疑 guard/重编译或 cudagraph tree 输出处理），model 阶段数字不受影响；未解决，compile 的 e2e 不应引用。
compile 的 autotune 会占用 GPU，本轮全程持 GPU 锁运行。

## 遗留
- compile 的 e2e 异常；compile 与自定义 TileLang kernel 的组合未验证。
