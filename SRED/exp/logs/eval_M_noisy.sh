#!/bin/bash
# noisy + real-noise stand-in evals for each matched-noise run, as soon as its sweep (train + clean eval) is DONE
cd /home/jovyan/work/thuc/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in M1_p50 M2_p80 M3_p80_hard M4_curr; do
  until grep -q DONE exp/logs/sweep_$m.out 2>/dev/null; do sleep 60; done
  for sp in public_noisy public_realnoise; do
    tag=$([ $sp = public_noisy ] && echo noisy || echo realnoise)
    .venv/bin/python -m sred.evaluate --model exp/ft/W_$m/avg.pt --audio-root data/$sp --split $sp --tag ${tag}_W_$m > exp/logs/eval_${tag}_W_$m.log 2>&1
  done
  echo "$m done" >> exp/logs/eval_M_noisy.progress
done
