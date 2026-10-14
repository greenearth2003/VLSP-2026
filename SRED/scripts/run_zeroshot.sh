#!/bin/bash
# Zero-shot benchmark of pretrained models on public test (E00-E03)
cd /home/jovyan/code/VLSP-2026/SRED && source env.sh
MODELS=${MODELS:-"redimnet_M_mix redimnet2_b3_mix wespeaker_r34_lm redimnet2_b4_vox2"}
for m in $MODELS; do
  .venv/bin/python -m sred.evaluate --model $m --tag zeroshot > exp/logs/zeroshot_$m.log 2>&1
done
