#!/bin/bash
# Re-run the nsys captures that failed because another agent's CUDA context was still tearing down when
# profile_nsys.sh asserted GPU exclusivity (it has no grace period).  Retries until the kernel summary exists.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round13_gemm_fusion_sm120
OUT=/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/nsys
LOG=/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/logs
TAG=a
run_one() {  # name, then the command args for profile_nsys.sh
  local name=$1; shift
  for try in $(seq 1 25); do
    if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then return 0; fi
    rm -f $OUT/${name}*
    bash ../common/profile_nsys.sh $OUT/$name "$@" > $LOG/nsys_${name}.log 2>&1
    if [ -f $OUT/${name}_cuda_gpu_kern_sum.csv ]; then echo "ok $name try $try"; return 0; fi
    echo "retry $name ($try)"; sleep 15
  done
  return 1
}
run_one blk_r5_ln_fuse_$TAG tests/profile_block.py --variant r5_ln_fuse --model vitl16 --joint 0
python tests/nsys_categorize.py $OUT/blk_r5_ln_fuse_$TAG --iters 5 --json profile/nsys_block_r5_ln_fuse.json > profile/nsys_block_r5_ln_fuse.txt 2>&1
for model in vitl16 vitb16; do
  run_one step_${model}_TB1_$TAG run.py --model $model --stack TB1 --action profile --phases mid --warmup 20
  python ../common/nsys_analyze.py $OUT/step_${model}_TB1_$TAG --iters 5 --top 25 > profile/nsys_step_${model}_TB1.txt 2>&1
done
echo RETRY_DONE
