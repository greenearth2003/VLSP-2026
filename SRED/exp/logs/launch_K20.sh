#!/bin/bash
cd /home/jovyan/work/thuc/VLSP-2026/SRED
until [ -f "/home/jovyan/work/thuc/VLSP-2026/SRED/exp/emb/W_G05_B6_avg/trainfull_modefull_f5574c4d.npy" ]; do sleep 30; done
WISE_STEP=6000 scripts/c_sweep.sh redimnet_M_mix K20_M stem lists/clean/train_clean.txt --teacher-emb /home/jovyan/work/thuc/VLSP-2026/SRED/exp/emb/W_G05_B6_avg/trainfull_modefull_f5574c4d.npy --min-crop-sec 0.75 --temb-kd 1.0 --temb-rkd 10 --steps 9000 --save-every 1000 > exp/logs/sweep_K20_M.out 2>&1
