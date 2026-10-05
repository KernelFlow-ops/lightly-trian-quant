#!/usr/bin/env bash
# nsys baseline of r5_v1 (bench9.py --profile), run under the dual lock (run_locked.sh). Usage: nsys_run.sh vitl16 vitb16
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
R=/root/lightly-train/infer-opt
T=/root/autodl-tmp/infer-opt-tmp/round10_lowp_probe/tmp
for M in "$@"; do
  OUT=$T/${M}_r5v1
  /usr/local/bin/nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true -o $OUT \
      /root/miniconda3/bin/python $R/round9_integration/bench9.py --model $M --profile r5_v1
  /usr/local/bin/nsys stats --force-export=true -r cuda_gpu_kern_sum,cuda_api_sum --format csv -o $OUT $OUT.nsys-rep > /dev/null
  ls -la $OUT*
done
