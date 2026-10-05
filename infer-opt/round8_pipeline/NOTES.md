# Round 8：系统级视频流水线（`video_infer.py`）

测试：`/root/autodl-tmp/1.mp4`（1900×1440，126 帧），循环 5 遍，取第 2 遍起的稳态；整个计时放在一次 GPU 锁内。
基线 = 原脚本 `predict_head`（bf16-autocast，未开 overlay）+ 同一个 cv2 解码循环。

## 1. 设计
- **解码线程**：`cv2.VideoCapture.read(pinned_ndarray)` 直接解码进 pinned ring（4~8 个槽，已验证 cv2 原地写入，`data` 指针不变），去掉 6 MB host memcpy。解码 3.5 ms/帧，不是瓶颈。
- **3 个 stream + event**：H2D（copy stream）→ compute stream `wait_event` → prep → `model_fn` → post → `record` → D2H stream（uint8 mask 2.7 MB + 4×int64）→ 结果线程 `event.synchronize()` 后算 confidence。`inflight` 信号量限制在途帧数（默认 2）。
- **CUDA Graph**：prep（1 kernel）与 post（memset + 2 kernel）按槽位各 capture 一张图（静态输入/输出地址随槽位固定）；model 保持 eager。model_fn 的输出复制进静态缓冲（2 个小 kernel）再进 post 图，因此换成 graphopt 的静态输出模型时可以去掉这次拷贝。
- **可插拔模型**：`model_fn(batch) -> (mask_logits_lr, class_logits)`，运行在 compute stream 上。默认 `make_eager_model_fn`：eager backbone + bf16 autocast + P3 融合 head。
- **overlay / 写视频**：独立 CPU 线程（`OverlaySink`，overlay + `VideoWriter` + CSV + best frame）；线程持有槽位直到写完再归还。
- **CLI**（与原脚本参数一致）：`python video_infer.py --model X.pth --video-dir D --out-dir O [--overwrite]`，输出 `*_infer.mp4 / *_frames.csv / *_best_frame.jpg / *_best_mask.png / summary.json`（不输出 `*_best_score.png`，因为整图 head_score 不再回传）。
- 只支持 K=2 类 + void；本轮整段视频流水线只测过宽图。共享的 Round 7 前后处理内核现已通过竖图单帧对照；相同尺寸的视频复用同一 pipeline。

## 2. 结果（`results.json` → `video` 字段）
| 模型 | 基线 FPS（decode+predict_head） | 流水线 FPS（inflight=2，稳态） | 加速 | 单帧服务延迟 p50/p90（inflight=1，提交→host 结果） | 基线 predict_head p50/p90 |
|---|---|---|---|---|---|
| ViT-L | 15.5 | **57.3** | 3.7× | **17.0 / 18.9 ms** | 59.2 / 69.2 ms |
| ViT-B | 12.7 | **118.3**（另两次 125.8 / 128.8） | 9.3× | **7.5 / 10.6 ms** | 64.3 / 71.8 ms |

- inflight=1/2/3 的 FPS：ViT-L 54.1 / 57.3 / 56.2；ViT-B 108.7 / 118.3 / 125.8（ViT-B 有约 ±5% 的 run 间抖动；inflight=1 略低，2~3 差别在噪声内）。
- 不用 CUDA Graph（prep/post 直接 launch）：ViT-L 57.5 / ViT-B 122.4 FPS，差别在噪声内 —— prep+post 只有 3 个 kernel，graph 的收益可忽略；保留它主要是为了以后整条链路进 graph。
- ViT-L 流水线 FPS 17.5 ms/帧 ≈ 单帧 e2e（17.4 ms）：**eager backbone 的 CPU launch 时间（约 16 ms）是上限**，流水线只把 prep/post/H2D/D2H/解码藏掉了；换成 graphopt 的 CUDA Graph 模型后这里会明显下降。ViT-B 的 GPU 时间约 8 ms，流水线达到 7.5~8.5 ms/帧。
- 环形队列口径（frame 进入 ring → 结果）在 inflight=2 时约 25 ms(ViT-B)/64 ms(ViT-L)：解码比 GPU 快，帧在 ring 里排队，这是“灌满流水线”的排队延迟，不是真实相机场景的延迟；实时相机应看服务延迟一列。
- overlay + 写视频（1 个 overlay 线程 + mp4v 编码）：优化版 **33~34 FPS**（ViT-L 33.6 / ViT-B 32.9），原脚本 **10.1~10.3 FPS**。此时瓶颈是 CPU 端 overlay/编码（GPU 不再是瓶颈），ViT-L 和 ViT-B 相同。
- nsys（`profile/vitl16_pipeline*`，`vitb16_pipeline*`，capture 1 遍 126 帧；SQLite 统计）：H2D / compute / D2H 分别在独立 stream（13/17/21）上，H2D 拷贝与 kernel 重叠 **91%（ViT-L）/ 80%（ViT-B）**，D2H 重叠 70% / 67%，kernel 只在 compute stream（17）上；H2D 每帧 0.148 ms，D2H 共 0.05 ms。

## 3. 精度
- 126 帧逐帧对比原脚本（bf16-autocast）：

| 模型 | mask 一致率 min / mean | head IoU min / pooled | confidence 差 max / mean |
|---|---|---|---|
| ViT-L | 0.99999 / 0.999995 | 0.9844 / 0.9942 | 0.0125 / 0.0010 |
| ViT-B | 0.99998 / 0.999994 | 0.9772 / 0.9927 | 0.0033 / 0.0010 |
  （单帧 head IoU 最低值来自 head 像素很少的帧：边界几个像素的差异；两边都是 bf16，各自与 fp32 的差距一样大。）
- 8 个评测帧对 fp32 参考的 gate（`mf.gate`）：ViT-L pooled IoU 0.9930 / 最大 conf 相对误差 0.44% → pass；ViT-B 0.9921 / 0.11% → pass。

## 4. 遗留 / 下一步
- 接入 graphopt 的 CUDA Graph 模型：`model_fn` 直接返回静态输出，去掉 `ml_buf` 拷贝；如果整个 model 也在 graph 里，可把 prep→model→post 合成一张图，CPU 占用降到接近 0，ViT-L 才能逼近 GPU 时间（约 15 ms）。
- overlay 线程是纯 CPU 瓶颈（约 30 ms/帧），可再拆成多线程 overlay + 顺序写盘，或把 overlay 挪到 GPU。
- 只做单路视频；多路可共用模型、每路一个 pipeline。

## 5. 文件
`video_infer.py`（流水线 + CLI）、`bench_video.py`（全部测量；`--profile` 供 nsys）、`results.json`、`profile/`（nsys rep/csv/summary）。依赖 `../round7_head_postprocess/impl`。
