#!/bin/bash
# queue 3 (final code): block gate A, block timing, step timing, nsys, gates B/S + calibration.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round13_gemm_fusion_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/logs
until grep -q QUEUE2_DONE $LOG/queue2.log 2>/dev/null; do sleep 10; done
echo "gateA block $(date +%T)"
python tests/test_fused_block.py --json profile/gateA_block.json > $LOG/gateA_block.log 2>&1
tail -3 $LOG/gateA_block.log
echo "bench block $(date +%T)"
python tests/bench_block.py --json profile/bench_block_final.json > $LOG/bench_block_final.log 2>&1
cat $LOG/bench_block_final.log | tail -22
echo "step timing $(date +%T)"
bash tests/run_step_timing.sh > $LOG/step_timing_driver.log 2>&1
echo "nsys $(date +%T)"
bash tests/run_nsys.sh a > $LOG/nsys_driver.log 2>&1
echo "gate B / S $(date +%T)"
for seed in 0 1 2; do
  python tests/calibrate_gateb.py --model vitl16 --seed $seed --out profile/calibrate_gateb_vitl16_seed${seed}.json > $LOG/calib_vitl_s${seed}.log 2>&1
done
python run.py --model vitl16 --stack 'r11:autocast,r13:full' --ref AC --action gate_b --phases mid --seeds 0 1 2 > $LOG/gateB_vitl_mid_final.log 2>&1
python run.py --model vitb16 --stack 'r11:autocast,r13:full' --ref AC --action gate_b --phases mid --seeds 0 1 2 > $LOG/gateB_vitb_mid_final.log 2>&1
python tests/calibrate_gateb.py --model vitb16 --seed 0 --out profile/calibrate_gateb_vitb16_seed0.json > $LOG/calib_vitb_s0.log 2>&1
python run.py --model vitl16 --stack TB1_R13 --ref TB1 --action gate_b --phases mid --seeds 0 > $LOG/gateB_vitl_tb1_final.log 2>&1
python run.py --model vitl16 --stack 'r11:autocast,r13:full' --ref AC --action gate_s --steps 50 > $LOG/gateS_vitl_final.log 2>&1
python run.py --model vitb16 --stack 'r11:autocast,r13:full' --ref AC --action gate_s --steps 50 > $LOG/gateS_vitb_final.log 2>&1
echo QUEUE3_DONE $(date +%T)
