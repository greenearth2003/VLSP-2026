#!/bin/bash
# F05: F03 recipe, longer fixed schedule, no test-based checkpoint selection, weight averaging
cd /home/jovyan/code/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -m sred.train --base redimnet_M_mix --name F05_M_s3_bnfz_long --train-from stage3 \
  --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 9000 --lr 2e-5 --head-lr 2e-3 --warmup 200 \
  --head-only-steps 200 --dev-trials 0 --eval-every 9000 --save-every 1500 --avg-last 3 \
  --aug 1.0 --freeze-bn > exp/logs/F05.log 2>&1
