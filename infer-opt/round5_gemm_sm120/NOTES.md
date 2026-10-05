# Round 5: sm_120 GEMM 调优

## 结论

推荐 `r5_v1`：保留 Round 3 的 fp32 残差流、独立 LayerNorm 和融合 GEMM epilogue，对真实视频每帧的两个 crop (`M=2058/2258`) 分形状选 tile。`impl/tuned_cfg.json` 是实际运行配置；`tuned_summary.json` 保留交错微基准候选。Round 9 可以通过 `r5_model_impls.make_model_fn_r5(name, model, "r5_v1")` 使用，输出合同与其现有 `model_fn` 一致。Round 5 入口使用具名 `round5_gemm_sm120.impl.ops`，不会与 Round 4 的顶层 `ops` 冲突。

| 模型 | Round 3 模型 graph p50 | Round 5 v1 模型 graph p50 | 同权重、同输入交错 graph A/B (R3 -> R5) | 8 帧 head IoU / 最大 conf 误差 | 126 帧 head IoU / 最大 conf 误差 |
|---|---:|---:|---:|---:|---:|
| ViT-L/16 | 10.852 ms | 10.513 ms | 10.114 -> 9.676 ms (-4.3%) | 0.9952 / 0.49% | 0.99484 / 2.98% |
| ViT-B/16 | 4.377 ms | 4.087 ms | 3.852 -> 3.594 ms (-6.7%) | 0.9932 / 0.31% | 0.99272 / 0.49% |

前两列来自 `test_model_v2.py --time` 的顺序 CUDA Graph 重放，包含原 `_predict`。交错 A/B 来自 `bench_model_ab.py`，两个实现共用模型权重、输入和 Round 7 融合 head；正反顺序各 27 次 graph 重放，表中是两次 p50 的平均。其余前处理、后处理和视频吞吐不包含在这些模型阶段数字里。8 帧门槛是合并 head IoU >= 0.95 且每帧 conf 相对误差 <= 5%。126 帧是完整 `1.mp4` 对 fp32 eager 的检查；两模型均无超过 5% 的帧。数据均保存在 `results.json`。

Round 9 的同轮视频配对（每变体 3x126 帧，inflight=2，持共享 GPU 锁）也显示收益：ViT-L 原 fused 为 92.45 FPS、服务延迟 p50 10.547 ms，`r5_v1` 为 94.65 FPS、10.243 ms；ViT-B 为 208.30 FPS、4.281 ms -> 214.10 FPS、4.003 ms。吞吐分别提高 2.4% 和 2.8%。原始数据在 [Round 9 results.json](../round9_integration/results.json) 的 `round5-v1-tuned+graph+fused-prepost+fused-head` 条目；其 8 帧 gate、graph/eager logits 一致性也通过。入口现默认宽图使用 `r5_v1`：一张竖图上，Round 3 fused 与 Round 5 v1 输出完全相同，head IoU 均为 0.8752，而 lightly 路径为 0.9821，因此竖图自动回退 lightly。这是共享融合 block 的数值敏感性，尚未解决。

## 形状与选择

下面是交错微基准的最快融合算子候选，单位 µs；TL 为 TileLang，MG 为本目录的 CUDA `mma.sync` kernel。每个格子是 `实现 / BMxBNxBK / split-K / µs`。`cp` 表示 TileLang 的 cp.async 路径，`TMA` 表示启用其 TMA/warp-specialized 路径。完整 stages、threads、warp 几何和 swizzle 见 `impl/tuned_cfg.json`。Torch `F.linear` 只算 bias，不含本轮的 RoPE/GELU/残差 epilogue，因此仅作参考下界，不能直接比较端到端效果。

| 模型 / M | QKV + RoPE | proj + residual | fc1 + GELU | fc2 + residual |
|---|---|---|---|---|
| L / 2058 | TL cp / 64x128x32 / 1 / 68.4 | TL TMA / 256x64x32 / 1 / 28.4 | MG / 160x64x32 / 1 / 83.5 | MG / 80x64x32 / **2** / 89.7 |
| L / 2258 | TL cp / 96x64x32 / 1 / 74.3 | TL TMA / 256x64x32 / 1 / 28.8 | MG / 64x128x32 / 1 / 91.0 | MG / 112x64x64 / 1 / 97.0 |
| B / 2058 | TL cp / 64x128x32 / 1 / 42.1 | TL cp / 64x64x64 / 1 / 17.6 | MG / 64x128x32 / 1 / 51.4 | MG / 64x80x64 / 1 / 55.3 |
| B / 2258 | TL cp / 64x128x32 / 1 / 42.7 | TL cp / 64x64x64 / 1 / 17.9 | MG / 96x64x32 / 1 / 54.2 | TL TMA / 96x128x64 / 1 / 57.0 |

L / 2058 的 fc2 以 split-K=2 从同轮非 split-K 候选的约 97.1 µs 降到 89.7 µs；其他形状并未选 split-K。四组 ViT-L stream-K 扫描中，2058 的最优项 `skm=0`（未启用 stream-K），2258 的 fc2 stream-K 约 98.6 µs，仍慢于选中的 97.0 µs。`v2` 的 LN/stats epilogue 与 v1 不同，因此 `ops._mg_kernel` 只读取专属 `v2_*` 条目；v1 的调参表不会误用于 v2。

## 静态检查与正确性

`ptxas -v` 对选中的 ViT-L / 2058 MG fc2 split-K kernel 报告 118 registers、0 spill stores/loads，动态 shared memory 为 18 KiB。其 cubin 的 SASS 有 `HMMA` 和 `LDGSTS`（静态反汇编分别 60 和 20 条）；选中的 MG fc1 报告 168 registers、0 spills、动态 shared memory 28 KiB，SASS 有 120 条 `HMMA` 和 28 条 `LDGSTS`。这里的条数是静态指令条数，不是运行时计数。容器未开放 ncu 性能计数器。

调参时发现 split-K 工作区在扩容后会释放旧分配，而已捕获的 CUDA Graph 仍持有旧指针，导致 fc2 交错重放非法访存。`impl/mgb.py` 现保留所有已分配工作区到进程退出；`test_workspace_graph.py` 在扩容后重放旧 graph 五次，输出最大绝对误差 `3e-6`，计数器归零。保留旧分配会在大量调参配置共存时增加内存占用，普通运行只有少数工作区大小。

已跑的模型级精度：ViT-L tuned v1 在 8 帧的最大 conf 误差 0.49%，126 帧最大 2.98%；ViT-B 分别 0.31% 和 0.49%。ViT-B 的 126 帧指标与 Round 3 一致。v1 的输出在 batch=2、M=2058/2258 的真实权重与真实视频上验证过；其他输入形状只有回退选择，没有同等验证。

## 未采用的方向

`v2` 把 LayerNorm 折入 QKV/fc1，生产者 GEMM 同时写 row stats，将每层 kernel 数从 7 降到 5；PDL 也可用。但顺序模型 graph p50：ViT-L `v2+PDL` 11.012 ms、`v2` 不带 PDL 11.060 ms，ViT-B 分别 4.377/4.381 ms，都慢于 tuned v1。ViT-L v2 的 8 帧最大 conf 相对误差为 4.93%，离 5% 门槛太近。L2 权重预取在既有四层链测试中也无稳定收益。故 v2、PDL 和预取仅保留实验入口。

没有对 CUTLASS 4.2 example 79 做同一融合 epilogue 的公平对照，当前对照为 Round 3 TileLang、Round 5 TileLang/MG 和 Torch bias-only `F.linear`。视频吞吐依据 Round 9 的 pipeline 配对实测；Round 5 仅作为已验证宽图 2 crop 路径的默认变体，竖图回退 lightly。

## 复现

所有 GPU 计时使用 `common/bench.py` 的同一把锁。完整命令外层也持锁；`INFER_OPT_GPU_LOCK_HELD=1` 避免内层重复加锁。

```bash
cd /root/lightly-train
flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock env INFER_OPT_GPU_LOCK_HELD=1 /root/miniconda3/bin/python infer-opt/round5_gemm_sm120/make_tuned.py vitl16 2058 2258
flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock env INFER_OPT_GPU_LOCK_HELD=1 /root/miniconda3/bin/python infer-opt/round5_gemm_sm120/make_tuned.py vitb16 2058 2258
flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock env INFER_OPT_GPU_LOCK_HELD=1 /root/miniconda3/bin/python infer-opt/round5_gemm_sm120/test_model_v2.py vitl16 --time
flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock env INFER_OPT_GPU_LOCK_HELD=1 /root/miniconda3/bin/python infer-opt/round5_gemm_sm120/test_heldout_v2.py vitl16
flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock env INFER_OPT_GPU_LOCK_HELD=1 /root/miniconda3/bin/python infer-opt/round5_gemm_sm120/bench_model_ab.py vitl16
flock -x /root/autodl-tmp/infer-opt-tmp/gpu.lock env INFER_OPT_GPU_LOCK_HELD=1 /root/miniconda3/bin/python infer-opt/round5_gemm_sm120/test_workspace_graph.py
```

将后三个命令的 `vitl16` 改为 `vitb16` 可复现 ViT-B 结果（工作区回归测试无模型参数）。
