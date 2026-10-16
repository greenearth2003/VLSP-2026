#!/bin/bash
# progressive pruning, stage 2: recovered 8.49M (P02) -> ~6.4M, then 18k-step recovery distillation
cd /home/jovyan/work/thuc/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q DONE exp/logs/sweep_P02_PB6s1.out 2>/dev/null; do sleep 30; done
PYTHONPATH=. .venv/bin/python -m sred.prune --base exp/ft/P02_PB6s1/last.pt --keep-1d 0.65 --keep-2d 0.8 --keep-head 0.9 \
  --err-1d 1e9 --err-2d 1e9 --data 1024 --out exp/ft/P_B6_s2/init.pt > exp/logs/prune_s2.log 2>&1
TE=exp/emb/W_G05_B6_avg/trainfull_modefull_f5574c4d.npy
WISE_STEP=12000 scripts/c_sweep.sh exp/ft/P_B6_s2/init.pt P03_PB6s2 stem lists/clean/train_clean.txt --teacher-emb $TE \
  --head-init-emb $TE --min-crop-sec 0.75 --temb-kd 2.0 --temb-rkd 20 --kd 1.0 --teacher redimnet2_b6_mix_teacher_pt \
  --lr 1e-4 --steps 18000 --save-every 2000 > exp/logs/sweep_P03_PB6s2.out 2>&1
