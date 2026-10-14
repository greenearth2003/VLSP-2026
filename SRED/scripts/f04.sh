#!/bin/bash
cd /home/jovyan/code/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -m sred.train --base redimnet_M_mix --name F04_M_s1_aug_bnfz_var --train-from stage1 \
  --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 4000 --lr 2e-5 --head-lr 2e-3 --warmup 200 \
  --head-only-steps 200 --eval-every 1000 --dev-trials 2000 --aug 1.0 --freeze-bn > exp/logs/F04.log 2>&1
