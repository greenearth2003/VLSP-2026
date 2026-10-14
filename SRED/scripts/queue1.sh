#!/bin/bash
# queue after F01: full eval F01, F02 (RKD from B6), quick subset evals of approved models
cd /home/jovyan/code/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "name F01_M_s3_aug" > /dev/null; do sleep 30; done
.venv/bin/python -m sred.evaluate --model exp/ft/F01_M_s3_aug/best.pt --tag F01 \
  --submission submissions/F01_public.txt > exp/logs/eval_F01.log 2>&1
.venv/bin/python -m sred.train --base redimnet_M_mix --name F02_M_s3_aug_rkdB6 --train-from stage3 \
  --bs 32 --crop-sec 2.0 --amp bf16 --steps 3000 --lr 5e-5 --head-lr 2e-3 --warmup 200 \
  --head-only-steps 200 --eval-every 500 --dev-trials 2000 --aug 1.0 \
  --teacher redimnet2_b6_mix_teacher --rkd 1.0 > exp/logs/F02.log 2>&1
for m in redimnet_S_mix wespeaker_r34_lm redimnet2_b4_vox2; do
  PYTHONPATH=. .venv/bin/python scripts/quick_subset_eval.py $m > exp/logs/quick_$m.log 2>&1
done
