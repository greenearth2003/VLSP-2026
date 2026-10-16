#!/bin/bash
cd /home/jovyan/work/thuc/VLSP-2026/SRED
until [ -f "/home/jovyan/work/thuc/VLSP-2026/SRED/exp/emb/C01_B6_step3000/trainfull_modefull_f5574c4d.npy" ]; do sleep 30; done
L=lists/clean/train_clean.txt; KD="--teacher-emb /home/jovyan/work/thuc/VLSP-2026/SRED/exp/emb/C01_B6_step3000/trainfull_modefull_f5574c4d.npy --min-crop-sec 0.75"
nohup scripts/c_sweep.sh redimnet_M_mix K10_M stage3 $L $KD --temb-kd 1.0 --temb-rkd 10 > exp/logs/sweep_K10_M.out 2>&1 &
nohup scripts/c_sweep.sh redimnet_M_mix K12_M stage3 $L $KD --temb-rkd 10 > exp/logs/sweep_K12_M.out 2>&1 &
WISE_STEP=6000 nohup scripts/c_sweep.sh redimnet_M_mix K11_M stem $L $KD --temb-kd 1.0 --temb-rkd 10 --steps 9000 --save-every 1000 > exp/logs/sweep_K11_M.out 2>&1 &
wait
