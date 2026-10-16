#!/bin/bash
cd /home/jovyan/work/thuc/VLSP-2026/SRED
until [ -f exp/ft/P_B6_s1/init.pt ]; do sleep 20; done
TE=exp/emb/W_G05_B6_avg/trainfull_modefull_f5574c4d.npy
WISE_STEP=6000 scripts/c_sweep.sh exp/ft/P_B6_s1/init.pt P02_PB6s1 stem lists/clean/train_clean.txt --teacher-emb $TE --head-init-emb $TE --min-crop-sec 0.75 --temb-kd 2.0 --temb-rkd 20 --kd 1.0 --teacher redimnet2_b6_mix_teacher_pt --lr 1e-4 --steps 9000 --save-every 1000 > exp/logs/sweep_P02_PB6s1.out 2>&1
