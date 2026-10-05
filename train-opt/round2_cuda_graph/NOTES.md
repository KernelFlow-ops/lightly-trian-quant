# Round2: segmented CUDA Graph and compile comparison (Agent A measurement collaborator, 2026-10-02)

## 1. Goal and hypothesis

Reduce CPU launch overhead from H6 after round1 `nosync` while keeping loss and optimizer eager. The graph boundary is the plain backbone segment plus the three joint blocks (each joint segment includes predict, mask, attention, and block); patch embedding, final predict, loss, backward outside the captured module, and optimizer remain at the harness boundary. Graph signatures are keyed by the three mask states (`prob=1`, interior, `prob=0`) and are captured lazily from the first real batch.

## 2. Changes

- `impl/segmented.py`: `PlainBlocks`, `JointBlocks`, lazy `make_graphed_callables`, seven-signature cap, explicit eager fallback with a reason, and `torch.compile` comparison modes (`default`, `reduce-overhead`, `max-autotune-no-cudagraphs`).
- `impl/patch.py`: requires round1 `nosync` for the segmented graph; installs the additive `ctx.meta['forward_train_fn']` hook; no model parameters or state_dict keys are replaced.
- The graph uses the source fp32 path unless the explicit round1 `autocast_bf16` experiment is present. Autocast cache is disabled when it is enabled.

## 3. Capture and analysis process

No CUDA capture has been run yet. Before any performance claim, hold both locks, inspect `nvidia-smi`, use `common/profile_nsys.sh` for five captured steps per model and phase, and run `common/nsys_analyze.py`. Record graph capture time, replay step time, re-capture count after phase/signature changes, and CPU launch/GPU ratios.

## 4. Results

No GPU measurement or accuracy claim is recorded. `results.json` is a null schema for graph and compile variants. The seven-signature cap differs from the four-signature planning estimate because the source's `prob=0` path still consumes `(B,Q)` random values; capture count and any skipped signature must be reported.

## 5. Correctness

- CPU tests: three passed. They compare segmented plain/joint forward call order, outputs, input gradients, and all parameter gradients to `forward_train_v2`; they also verify the additive patch, round1 `nosync` dependency, and capture/compile report fields.
- B/C/D GPU gates remain pending. A failed capture falls back to eager and is reported as `captured=false` with the exact error; it is never counted as a graph speedup.

## 6. Static analysis

No custom kernel is introduced. CUDA graph replay will be analyzed through nsys kernel/API traces; `kernel_static.sh` is not applicable.

## 7. Conclusion and next bottleneck

Wait for M0 and the source-fp32 round1 `nosync` B gate. If launch gaps are below the measured threshold or capture requires too many signatures, record round2 as a controlled compile comparison and do not include it in `BEST_STACK`.
