#!/bin/bash
cd /home/jovyan/work/thuc/VLSP-2026/SRED && source env.sh
for d in S_K21_s12k_a0.4 S_K21_s12k_a0.6 S_K21_s12k_a0.7 S_K21_avg4_a0.5 S_K21_avg4_a0.7 S_K21_avg4_a1.0; do
  .venv/bin/python -m sred.evaluate --model exp/ft/$d/avg.pt --tag soup_$d > exp/logs/eval_$d.log 2>&1
done
echo ALLDONE > exp/logs/queue_soups.done
