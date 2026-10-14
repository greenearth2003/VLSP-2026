#!/bin/bash
# After F05 (WAIT_PID) finishes: K01 = F05 recipe + relational KD from B6 (pre-registered, w=10),
# while F05/avg.pt embeddings are extracted in parallel for the P-dev protocol.
cd /home/jovyan/code/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
.venv/bin/python -m sred.protocol --extract exp/ft/F03_M_s3_aug_bnfz_var/best.pt exp/ft/F05_M_s3_bnfz_long/avg.pt \
  --n-boot 300 > exp/logs/protocol_F05.log 2>&1 &
.venv/bin/python -m sred.train --base redimnet_M_mix --name K01_M_s3_bnfz_long_rkdB6 --train-from stage3 \
  --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 9000 --lr 2e-5 --head-lr 2e-3 --warmup 200 \
  --head-only-steps 200 --dev-trials 0 --eval-every 9000 --save-every 1500 --avg-last 3 \
  --aug 1.0 --freeze-bn --teacher redimnet2_b6_mix_teacher --rkd 10 --teacher-clean 1 > exp/logs/K01.log 2>&1
wait
.venv/bin/python -m sred.protocol --extract exp/ft/F03_M_s3_aug_bnfz_var/best.pt exp/ft/F05_M_s3_bnfz_long/avg.pt \
  exp/ft/K01_M_s3_bnfz_long_rkdB6/avg.pt --n-boot 300 > exp/logs/protocol_K01.log 2>&1
