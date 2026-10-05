#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH PYTHONPATH=/root/lightly-train/train-opt:/root/lightly-train/src
cd /root/lightly-train/train-opt/round16_full_graph
L=/root/autodl-tmp/train-opt-tmp/round16_full_graph/logs
F=/echo
R15=/root/lightly-train/train-opt/round15_optimizer_flat
tests/locked_run.sh python tests/test_fallback_and_swap.py vitb16 > $L/fallback_vitb16.log 2>&1
echo FALLBACK_DONE
tests/locked_run.sh python $R15/tests/real_step_lr.py vitb16 > $R15/profile/real_step_lr_vitb16.log 2>&1
tests/locked_run.sh python $R15/tests/real_step_lr.py vitl16 > $R15/profile/real_step_lr_vitl16.log 2>&1
echo REALSTEP_DONE
tests/locked_run.sh python $R15/tests/check_real_flat.py vitl16 P2,r15 > $R15/profile/check_real_flat_vitl16.log 2>&1
echo CHECKREAL_DONE
tests/locked_run.sh python run.py --model vitl16 --stack P2,r15 --action graph_equiv --scenarios fixed_mid --equiv-steps 6 > $L/equiv_vitl16.log 2>&1
echo ALLFINAL
