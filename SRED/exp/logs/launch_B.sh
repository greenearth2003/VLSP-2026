#!/bin/bash
# Approach B: K21 recipe + overlapping quieter second speaker augmentation. Starts when BOTH E2a and E2b are done
# (E1a starts at the first of them), then E1b is re-run after B finishes training.
cd /home/jovyan/work/thuc/VLSP-2026/SRED
until grep -q DONE exp/logs/sweep_E2a.out 2>/dev/null && grep -q DONE exp/logs/sweep_E2b.out 2>/dev/null; do sleep 60; done
TE=exp/emb/W_G05_B6_avg/trainfull_modefull_f5574c4d.npy; TL=exp/emb/tlogits_C01_B6_trainfull.npy; L=lists/clean/train_clean.txt
BASE="--teacher-emb $TE --min-crop-sec 0.75 --temb-kd 2.0 --temb-rkd 20 --steps 18000 --save-every 2000"
WISE_STEP=12000 scripts/c_sweep.sh redimnet_M_mix B01_overlap stem $L $BASE --aug-overlap 0.3 > exp/logs/sweep_B01.out 2>&1
rm -rf exp/ft/E1b_trkd
WISE_STEP=12000 scripts/c_sweep.sh redimnet_M_mix E1b_trkd stem $L $BASE --tlogits $TL --dkd 0.3 --dkd-conf-k 50 > exp/logs/sweep_E1b.out 2>&1
