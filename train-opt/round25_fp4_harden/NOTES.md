# Round25（R25）：最终栈 P3 的加固（dY headroom / 饱和研究、ViT-B 保守档 P3_H2、ViT-L P3_q8 补验证；不写新 kernel）

> 数字全部在 `infer-5090`（RTX 5090）上实测，所有 GPU 进程持双锁（与 R24 共用 GPU，进程之间用锁串行）；没测的写 null / not measured。机器可读结果：`results.json`。
> 没改 round0–24、common、PLAN、CONTRACTS、README；只 import 了 round16 / 19 / 21 / 22 / 23 的代码。本文件边做边写，下面的状态行标明哪些已经完成。

## 0. 摘要

> 状态：**进行中**（已完成：搭建 `run.py` / `gate_b_p25.py` / 队列；任务一的饱和研究在跑）。

## 1. 目标与假设

| 假设 | 结果 |
|---|---|
| H-a：P3 与 P3B（R22）的 dY 饱和率差别来自两次不同的真实 batch（加载器跨进程不确定） | 待测 |
| H-b：P3 里 head 的 FP4 梯度改变了回传到骨干的 dY 分布 | 待测 |
| H-c：某些 site 的 amax 统计受相邻步影响（2 个 batch 交替） | 待测 |

## 2. 改动（全部在 `train-opt/round25_fp4_harden/`）

| 文件 | 内容 |
|---|---|
| `impl/presets25.py` | round23 的预设 + `P3_hr16/32/64`（骨干 dY headroom）、`P3_ex`（两个 patch 都 `amax=exact`）、`P3_H2`（head `head.dw=mx8`）、`P3_H2f`（`head.f=mx8,head.dw=mx8`） |
| `run.py` | round23 `run.py` 的拷贝（bench / profile / gate_s / graph_equiv / e2e）+ `gate_s_sweep`（同一进程、同样 2 批、R0 每场景一次、多个栈）+ `sat_study`（逐步逐 site 的 amax 轨迹） |
| `run_final.py` | round23 `run_final.py` 的拷贝（round25 的预设 / 缓存） |
| `tests/gate_b_p25.py` | round23 `gate_b_p3.py` 的拷贝，候选加了 `P3_hr16/32/64`、`P3_H2`、`P3_H2f` |
| `tests/analyze_sat.py`、`tests/make_gateb25.py`、`tests/queue_*.sh`、`tests/retry.sh` | 饱和轨迹分析、门 B 合并表、排队脚本 |

## 3. nsys 分析过程

本轮没有写新 kernel，没有新的 nsys 抓取；整步代价用 bench（graph，三相加权，A-B-A-B 交替）直接测。
