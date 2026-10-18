#!/bin/bash
cd /home/jovyan/work/thuc/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev() { .venv/bin/python -m sred.evaluate --model "$1" --audio-root data/public_noisy --split public_noisy --tag "noisy_$2" > exp/logs/eval_noisy_$2.log 2>&1; }
ev redimnet_M_mix zeroshot_M & ev exp/ft/W_K21_M/avg.pt K21 & ev exp/ft/S_K21_s12k_a0.3/avg.pt K21_a03 & wait
ev exp/ft/W_E2b_sf2/avg.pt E2b_sf2 & ev exp/ft/C01_M/step3000.pt C01_M & ev exp/ft/W_B01_overlap/avg.pt B01_overlap & wait
ev exp/ft/S_K21_s12k_a0.4/avg.pt K21_a04 & ev redimnet2_b6_mix_teacher_pt zeroshot_B6 & ev exp/ft/W_G05_B6/avg.pt teacher_B6 & wait
echo ALLDONE > exp/logs/noisy_eval.done
