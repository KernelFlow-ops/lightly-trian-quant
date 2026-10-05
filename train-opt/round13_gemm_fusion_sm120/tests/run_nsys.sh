#!/bin/bash
# nsys: (a) one block fwd+bwd per variant (exact per-block kernel lists), (b) full steps (mid phase) T-BF16 vs T-BF16 + round13, ViT-L and ViT-B.
# profile_nsys.sh takes both GPU locks; raw .nsys-rep/.sqlite stay in the tmp dir.  (common/*.sh are not executable on the new container: call with bash.)
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round13_gemm_fusion_sm120
OUT=/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/nsys
LOG=/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/logs
TAG=${1:-a}
for v in autocast_default r5_ln_fuse r13_full; do
  echo "=== block $v $(date +%T)"
  bash ../common/profile_nsys.sh $OUT/blk_${v}_${TAG} tests/profile_block.py --variant $v --model vitl16 --joint 0 > $LOG/nsys_blk_${v}_${TAG}.log 2>&1
  python tests/nsys_categorize.py $OUT/blk_${v}_${TAG} --iters 5 --json profile/nsys_block_${v}.json > profile/nsys_block_${v}.txt 2>&1
  tail -3 profile/nsys_block_${v}.txt
done
for model in vitl16 vitb16; do
  for stack in TB1 TB1_R13; do
    echo "=== step $model $stack $(date +%T)"
    bash ../common/profile_nsys.sh $OUT/step_${model}_${stack}_${TAG} run.py --model $model --stack $stack --action profile --phases mid --warmup 20 > $LOG/nsys_step_${model}_${stack}_${TAG}.log 2>&1
    python ../common/nsys_analyze.py $OUT/step_${model}_${stack}_${TAG} --iters 5 --top 25 > profile/nsys_step_${model}_${stack}.txt 2>&1
    head -8 profile/nsys_step_${model}_${stack}.txt | cut -c1-200
  done
done
echo ALL_DONE $(date +%T)
