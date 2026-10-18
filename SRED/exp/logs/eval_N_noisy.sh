#!/bin/bash
cd /home/jovyan/work/thuc/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in W_N10_sf2_noisy W_N11_aam_noisy; do
  .venv/bin/python -m sred.evaluate --model exp/ft/$m/avg.pt --audio-root data/public_noisy --split public_noisy --tag noisy_$m > exp/logs/eval_noisy_$m.log 2>&1
  .venv/bin/python -m sred.evaluate --model exp/ft/$m/avg.pt --audio-root data/public_realnoise --split public_realnoise --tag realnoise_$m > exp/logs/eval_realnoise_$m.log 2>&1
done
echo ALLDONE > exp/logs/eval_N_noisy.done
