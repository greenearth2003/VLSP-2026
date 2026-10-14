#!/bin/bash
cd /home/jovyan/code/VLSP-2026/SRED && source env.sh
.venv/bin/python -m sred.evaluate --model exp/ft/F03_M_s3_aug_bnfz_var/best.pt --tag F03 > exp/logs/eval_F03.log 2>&1
for m in redimnet_S_mix wespeaker_r34_lm redimnet2_b4_vox2; do
  PYTHONPATH=. .venv/bin/python scripts/quick_subset_eval.py $m > exp/logs/quick_$m.log 2>&1
done
