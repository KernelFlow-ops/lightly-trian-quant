# Round10: integration (Agent A measurement collaborator, 2026-10-02)

## 1. Goal and hypothesis

Expose the original Lightly loop and a patchable fast loop with compatible checkpoint/export files. The final throughput and 95% mIoU/head-IoU gates require M0, the selected best stack, and 20-epoch paired runs; none is claimed yet.

## 2. Changes

- `train.py`: the original script's data/model/epoch/batch/precision/seed/output arguments plus `--loop lightly|fast`, `--opt`, `--patches`, and `--patch-kwargs`. `auto` resolves to fixed `B_AUTO` for comparable training, without a new probe.
- `impl/lightly_patch.py`: scoped runtime injection of supported round1 forward/optimizer variants into Lightly's own training loop. Unsupported patch stacks are rejected; no silent baseline fallback is called optimized.
- `impl/fast_loop.py`: uses harness `train_step`, real Lightly loaders and step-aware transform updates; validation uses Lightly `validation_step` with classwise metrics. Best/last checkpoint and export use Lightly's own helper APIs and the same state/export keys as the source loop.
- `verify_export.py`: read-only state_dict key/shape and model metadata comparison; under dual GPU lock, runs `lightly_train.load_model(...).predict(...)` and optionally infer-opt's existing video entrypoint with output under train-opt tmp.
- `compare.py`: requires two distinct, matching baseline/final seeds at the same model, batch, and epoch budget; computes mean/std, the 95% D gate, per-seed curve figures and mean curve JSON for A's global summary, and only then updates the aggregate `final` row.
- `metrics.jsonl`, `run_config.json`, `train_opt_run.json`, `exported_models/exported_{last,best}.pt`, and copied `{last,best}.pth` are written under the run output directory when checkpoints are enabled.

## 3. Performance analysis process

No round10 nsys report exists yet. Once the best stack is selected, capture both loops under dual GPU locks, compare early/mid/late fixed-batch step and real e2e/epoch time, then record kernel counts, syncs, GPU busy ratio, top kernels and the exact command/CSV paths here. A full-step CUDA Graph is not reported until C's r3 padded loss and r7 capturable optimizer interfaces have passed B/C checks.

## 4. Results

短 GPU B 级与 segmented capture 已完成：r1 `nosync+adamw_fused` 在 ViT-B14 mid 的 loss_rel=0、grad_cos_min=0.9999961、grad_norm_rel=1.70e-7、update_cos=1；ViT-L7 mid 的 loss_rel=0、grad_cos_min=0.9999939、grad_norm_rel=1.35e-7、update_cos=1。r2 segmented 在两模型 mid 均真实捕获 plain 与 `zero/middle/one` joint 签名并通过 B 级：B14 capture 0.649/0.761 s，grad_cos_min=0.9999995；L7 capture 1.081/0.669 s，grad_cos_min=0.9995865。日志分别为 `/root/autodl-tmp/train-opt-tmp/A/r1_vitb14_mid.json`、`r1_vitl7_mid.json`、`r2_vitb14_mid.json`、`r2_vitl7_mid.json`。这些是正确性/capture smoke，未作为吞吐收益声明。

新增 `impl/full_graph.py`：静态图 batch 使用 r3 的 padded `{masks, labels, valid}`，图内执行 forward/loss/backward/r7 capturable optimizer，scheduler 在 replay 后图外更新；捕获失败保留错误并回退由调用方决定。完整图 C/D 尚未运行。

`results.json` has `null` for all unmeasured fields. Completed runs will be recorded under distinct epoch/seed/output identifiers so the two-seed evidence is not overwritten; the aggregate `final` row stays null until comparison. No `BEST_STACK` or final 20-epoch comparison has been selected. The fast loop's compatible ckpt/export path needs a real completed run and `lightly_train.load_model`/predict verification.

## 5. Correctness

- CPU tests: an actual CPU Fabric save through Lightly helpers produces the same checkpoint/export keys as the source; class method injection restores on exit and rejects unsupported components; the fast epoch updates transforms (including worker reinitialization), validates, saves last/best, and emits a D-gate-readable log; export metadata reader preserves state shapes; two-seed budget/gate comparison and both curve outputs are checked. Total r1/r2/r10 CPU suite: 21 passed.
- GPU B/C/D and export compatibility have not run. `ckpt_async` is excluded from the fast loop until its asynchronous snapshot is proven to reproduce Fabric's stateful checkpoint payload exactly; synchronous Lightly helpers remain the compatible path.

## 6. Static analysis

No new custom kernel in the integration code. Static analysis for included B/C kernels belongs to their rounds; graph replay and scheduling will be analyzed with nsys.

## 7. Conclusion and next bottleneck

Complete M0 and each round's B gate, then choose a measured best stack. Wire the r3/r7 graph inputs after their contracts are published, run both models and two seeds at equal 20-epoch budgets, and validate the exported model through Lightly and infer-opt's read-only inference entrypoint.
