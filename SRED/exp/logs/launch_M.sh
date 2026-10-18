#!/bin/bash
# matched-noise grid (SphereFace2 + K21 KD recipe). Two at a time: start after N10/N11 finish.
cd /home/jovyan/work/thuc/VLSP-2026/SRED
until grep -q DONE exp/logs/sweep_N10.out 2>/dev/null && grep -q DONE exp/logs/sweep_N11.out 2>/dev/null; do sleep 60; done
TE=exp/emb/W_G05_B6_avg/trainfull_modefull_f5574c4d.npy; L=lists/clean/train_clean.txt
BASE="--teacher-emb $TE --min-crop-sec 0.75 --temb-kd 2.0 --temb-rkd 20 --steps 18000 --save-every 2000 --loss sf2 --scale 32 --margin 0.2"
run() { WISE_STEP=12000 scripts/c_sweep.sh redimnet_M_mix $1 stem $L $BASE ${@:2} > exp/logs/sweep_$1.out 2>&1; }
run M1_p50 --aug-matched 0.5 --aug-matched-shift -8 & run M2_p80 --aug-matched 0.8 --aug-matched-shift -8 & wait
run M3_p80_hard --aug-matched 0.8 --aug-matched-shift -13 & run M4_curr --aug-matched 0.3 --aug-matched-end 0.9 --aug-matched-shift -8 & wait
echo ALLDONE > exp/logs/M_grid.done
