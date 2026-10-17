#!/bin/bash
cd /home/jovyan/work/thuc/VLSP-2026/SRED
until grep -q DONE exp/logs/sweep_E2a.out 2>/dev/null || grep -q DONE exp/logs/sweep_E2b.out 2>/dev/null || grep -q DONE exp/logs/sweep_E1b.out 2>/dev/null; do sleep 60; done
TE=exp/emb/W_G05_B6_avg/trainfull_modefull_f5574c4d.npy; TL=exp/emb/tlogits_C01_B6_trainfull.npy
WISE_STEP=12000 scripts/c_sweep.sh redimnet_M_mix E1a_dkd stem lists/clean/train_clean.txt --teacher-emb $TE --min-crop-sec 0.75 \
  --temb-kd 2.0 --temb-rkd 20 --steps 18000 --save-every 2000 --tlogits $TL --dkd 0.3 > exp/logs/sweep_E1a.out 2>&1
