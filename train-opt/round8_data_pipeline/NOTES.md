# Round8：数据管线（负责：Agent A，2026-10-02）

## 1. 目标与触发

R0 的真实 loader e2e 与固定 GPU step 差距触发本轮：ViT-B 33.14 vs 37.80 img/s（12.3%），ViT-L 14.94 vs 16.25 img/s（8.1%）。平均 DataLoader 等待约 14.1/17.0 ms，剩余差距还包含真实循环和增强开销。

## 2. 当前改动

- 已保留 `common.data.make_loader` 的真实 Lightly dataset/transform/collate 路径。
- 未改变数据分布、未在只读数据集旁生成缓存；暂未实施解码缓存或 GPU 增强。
- round1 的 loader variant 可以扫描 workers/prefetch/pin/persistent；`non_blocking` 已由 Fabric 的 `move_data_to_device` 默认处理。

## 3. 结果

| model | fixed GPU step img/s | e2e img/s | loader wait ms | 状态 |
|---|---:|---:|---:|---|
| ViT-B/16 B=14 | 37.80 | 33.14 | 14.06 | 触发，待参数扫描 |
| ViT-L/16 B=7 | 16.25 | 14.94 | 16.98 | 触发，待参数扫描 |

当前没有 GPU 增强或缓存的性能数字；不把实现计划当作结果。

## 4. 下一步

先在相同数据分布下扫描 `num_workers=16/20/24`、`prefetch_factor=2/4`、`pin_memory` 和 persistent worker 约束。只有扫描仍显示显著差距，才建立临时目录下的解码 memmap；GPU 增强需另过分布 C/D 检查。
