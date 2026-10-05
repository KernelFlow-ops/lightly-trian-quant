#!/usr/bin/env bash
# finalize chain: margin variants first (primary), then the bare-gate variants
cd /root/lightly-train/infer-opt/round10_lowp_probe
L=/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/logs
for m in vitb16 vitl16; do for p in safe fast; do
  bash run_locked.sh $L/FINm_${m}_${p}.log bash batchFin.sh $m $p _margin 0.965 0.03 420
done; done
for m in vitb16 vitl16; do for p in safe fast; do
  bash run_locked.sh $L/FINb_${m}_${p}.log bash batchFin.sh $m $p _bare 0.0 1.0 420
done; done
