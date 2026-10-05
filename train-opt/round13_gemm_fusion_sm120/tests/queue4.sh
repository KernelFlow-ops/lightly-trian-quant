#!/bin/bash
# queue 4 (final code, priority order): gate B (check.step_equivalence) + calibration, gate S, nsys, remaining step timings.
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round13_gemm_fusion_sm120
LOG=/root/autodl-tmp/train-opt-tmp/round13_gemm_fusion_sm120/logs
echo "gateB vitl $(date +%T)"
python tests/gate_b_step_equivalence.py --model vitl16 --seeds 0 1 2 > $LOG/gateB_se_vitl.log 2>&1
for seed in 0 1 2; do python tests/calibrate_gateb.py --model vitl16 --ref TB1 --cand TB1_R13 --seed $seed --out profile/calibrate_gateb_vitl16_seed${seed}.json > $LOG/calib_vitl_s${seed}.log 2>&1; done
echo "gateB vitb $(date +%T)"
python tests/gate_b_step_equivalence.py --model vitb16 --seeds 0 1 2 > $LOG/gateB_se_vitb.log 2>&1
python tests/calibrate_gateb.py --model vitb16 --ref TB1 --cand TB1_R13 --seed 0 --out profile/calibrate_gateb_vitb16_seed0.json > $LOG/calib_vitb_s0.log 2>&1
echo "gateS $(date +%T)"
python run.py --model vitl16 --stack TB1_R13 --ref TB1 --action gate_s --steps 50 > $LOG/gateS_vitl.log 2>&1
python run.py --model vitb16 --stack TB1_R13 --ref TB1 --action gate_s --steps 50 > $LOG/gateS_vitb.log 2>&1
echo "nsys $(date +%T)"
bash tests/run_nsys.sh a > $LOG/nsys_driver.log 2>&1
echo "extra timing $(date +%T)"
for stack in AC "r11:autocast,r13:full"; do
  name=$(echo "$stack" | tr ',:' '__')
  python run.py --model vitb16 --stack "$stack" --action bench --phases all --name "$name" > $LOG/bench_vitb16_${name}.log 2>&1
done
echo QUEUE4_DONE $(date +%T)
