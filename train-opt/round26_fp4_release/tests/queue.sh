#!/bin/bash
# round26 release verification queue.  Part 1: same-session whole-step bench of the release stacks (graph, 3 phases).  Part 2: the checks the ViT-B release default (V3 = NVFP4 backbone + MXFP8-backward head)
# never had: B gate draw 3 (n=24 with round25's two draws), S gate (4 scenarios), graph_equiv, real run_final + export / resume.  Part 3: export / resume of the ViT-L release stack.
source /root/lightly-train/train-opt/round26_fp4_release/tests/qlib.sh
bench() { for spec in "$@"; do model=${spec%%:*}; stack=${spec#*:}
          step bench_${model}_${stack}_$(date +%H%M%S) "" python run.py --model $model --stack "$stack" --runner graph --action bench --phases all; done; }
bench vitl16:FINAL vitl16:V3 vitl16:FINAL vitl16:V3 vitb16:FINAL vitb16:V3 vitb16:V3_speed vitb16:FINAL vitb16:V3 vitb16:V3_speed
echo BENCH_DONE $(date +%T)

step gb_vitb_d3 profile/gateB_h2_vitb16_draw3.json python tests/gate_b_p25.py --model vitb16 --cands P3 P3_H2 P3_hr32 --seeds 16 17 18 19 20 21 22 23 --out profile/gateB_h2_vitb16_draw3.json
step gs_vitb_a "" python run.py --model vitb16 --action gate_s_sweep --stacks V3 V3_speed --scenarios schedule compressed --tag a
step gs_vitb_b "" python run.py --model vitb16 --action gate_s_sweep --stacks V3 V3_speed --scenarios fixed_mid fixed_late --tag b
step ge_vitb "" python run.py --model vitb16 --stack V3 --action graph_equiv --equiv-steps 8
for m in vitb16 vitl16; do
  rm -rf $T/ckpt/$m
  step exp_run_$m "" python run_final.py --model $m --stack V3 --steps 20 --val --save-ckpt $T/ckpt/$m --workers 16 --persistent --json profile/run_final_${m}_V3_export.json
  step exp_check_$m "" python /root/lightly-train/train-opt/round19_final/tests/check_export.py --model $m --ckpt-dir $T/ckpt/$m --out profile/export_${m}_V3.json
done
step ge_vitl "" python run.py --model vitl16 --stack V3 --action graph_equiv --equiv-steps 8
echo ALL_DONE $(date +%T)
