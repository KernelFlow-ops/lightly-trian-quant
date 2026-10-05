#!/bin/bash
cd /root/lightly-train/train-opt
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
O=/root/autodl-tmp/train-opt-tmp/round14_head_sm120
R=round14_head_sm120
while pgrep -f round14_head_sm120/queue1.sh > /dev/null; do sleep 20; done
# 1. CUDA-graph capturability
python -u $R/tests/test_graph.py --model vitl16 > $O/logs/m_graph_vitl16.log 2>&1
python -u $R/tests/test_graph.py --model vitb16 > $O/logs/m_graph_vitb16.log 2>&1
# 2. gate B (calibrated, 3 seeds)
for m in vitl16 vitb16; do
  python -u $R/tests/gate_b_multi.py --model $m --ref TB1_R13 --cands TB1_R13_R14 --seeds 0 1 2 --out $R/profile/gateB_${m}_r14.json > $O/logs/m_gateB_${m}.log 2>&1
done
# 3. gate S (50 steps, 2 batches)
for m in vitl16 vitb16; do
  python -u $R/tests/gate_s_multi.py --model $m --ref TB1_R13 --cands TB1_R13_R14 --phases schedule mid --tag r14a --out $R/profile/gateS_${m}_r14.json > $O/logs/m_gateS_${m}.log 2>&1
done
# 4. per-predict nsys
$O/run_nsys_head.sh torch vitb16
$O/run_nsys_head.sh r14 vitl16
$O/run_nsys_head.sh r14 vitb16
# 5. P2 step benches
for m in vitl16 vitb16; do
  for s in P2 P2_R14; do
    python -u $R/run.py --model $m --stack $s --action bench --phases all --name $s > $O/logs/m_bench_${m}_${s}.log 2>&1
  done
done
# 6. step nsys (mid)
for ms in vitl16:TB1_R13 vitl16:TB1_R13_R14 vitl16:P2_R14 vitb16:TB1_R13 vitb16:TB1_R13_R14; do
  m=${ms%%:*}; s=${ms##*:}
  for try in 1 2 3 4 5 6; do
    rm -f $O/nsys/step_${m}_${s}*
    bash common/profile_nsys.sh $O/nsys/step_${m}_${s} $R/run.py --model $m --stack $s --action profile --phases mid --warmup 20 > $O/logs/m_nsys_${m}_${s}.log 2>&1 && break
    sleep 20
  done
done
# 7. optional MXFP8 ConvT head
python -u $R/tests/gate_b_multi.py --model vitl16 --ref TB1_R13 --cands MXH_R14_bwd MXH_R14 --seeds 0 1 2 --out $R/profile/gateB_vitl16_r14mx8.json > $O/logs/m_gateB_vitl16_mx8.log 2>&1
for m in vitl16 vitb16; do
  python -u $R/run.py --model $m --stack MXH_P2_bwd --action bench --phases all --name MXH_P2_bwd > $O/logs/m_bench_${m}_MXH_P2_bwd.log 2>&1
done
# 8. robustness of the stage kernels (odd batch sizes / small grids)
( for cfg in '--C 1024 --B 1 --h 32' '--C 768 --B 3 --h 8' '--C 1024 --B 2 --h 16 --spiky' '--C 768 --B 5 --h 32 --spiky' '--C 1024 --B 7 --h 64 --spiky' '--C 768 --B 14 --h 64'; do
    echo "### cuda $cfg"; python -u $R/tests/test_triton_stage.py --impl cuda $cfg
  done ) > $O/logs/m_robust.log 2>&1
echo done > $O/logs/m_done.flag
