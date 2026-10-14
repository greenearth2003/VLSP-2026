#!/bin/bash
# Pre-registered runs (EXPERIMENTS.md, Protocol v2). Restart-safe: re-run this script after a
# container restart; training resumes from resume.pt and finished stages are skipped.
cd /home/jovyan/code/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 \
  --steps 9000 --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 9000 \
  --save-every 1500 --avg-last 3 --aug 1.0 --freeze-bn --resume"
F05=exp/ft/F05_M_s3_bnfz_long
K01=exp/ft/K01_M_s3_bnfz_long_rkdB6

[ -f $F05/avg.pt ] || .venv/bin/python -m sred.train $COMMON --name F05_M_s3_bnfz_long >> exp/logs/F05.log 2>&1
[ -f $F05/avg.pt ] || exit 1
.venv/bin/python -m sred.protocol --extract exp/ft/F03_M_s3_aug_bnfz_var/best.pt $F05/avg.pt \
  --n-boot 300 > exp/logs/protocol_F05.log 2>&1 &
[ -f $K01/avg.pt ] || .venv/bin/python -m sred.train $COMMON --name K01_M_s3_bnfz_long_rkdB6 \
  --teacher redimnet2_b6_mix_teacher --rkd 10 --teacher-clean 1 >> exp/logs/K01.log 2>&1
wait
[ -f $K01/avg.pt ] || exit 1
.venv/bin/python -m sred.protocol --extract exp/ft/F03_M_s3_aug_bnfz_var/best.pt $F05/avg.pt $K01/avg.pt \
  --n-boot 300 > exp/logs/protocol_K01.log 2>&1
echo PIPELINE_DONE
