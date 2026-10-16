#!/bin/bash
# G-series: fine-tune every approved pretrained model with the W01/F03 recipe, then evaluate.
#   0) zero-shot eval with all back-ends (also caches train-cohort embeddings used for head init + QMF)
#   1) fine-tune (F03 recipe, no test data touched: dev-trials 0, fixed steps, early layers frozen)
#   2) eval fine-tuned final checkpoint (step3000) and WiSE-FT 0.5 of step2000 (= W01 recipe)
# Usage: scripts/g_sweep.sh <base> <run_name> <train_from>
set -e
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=$1; RUN=$2; FROM=$3
PY=.venv/bin/python

$PY -m sred.evaluate --model $BASE --tag zeroshot_$RUN > exp/logs/zeroshot_$RUN.log 2>&1

$PY -m sred.train --base $BASE --name $RUN --train-from $FROM \
  --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 \
  --head-only-steps 200 --dev-trials 0 --eval-every 3000 --save-every 500 --aug 1.0 --freeze-bn --seed 0 \
  --workers 10 > exp/logs/$RUN.log 2>&1

$PY -m sred.evaluate --model exp/ft/$RUN/step3000.pt --tag ft_$RUN > exp/logs/eval_$RUN.log 2>&1
$PY scripts/wise_interp.py exp/ft/$RUN/step2000.pt exp/ft/W_$RUN/avg.pt --alpha 0.5
$PY -m sred.evaluate --model exp/ft/W_$RUN/avg.pt --tag wise_$RUN > exp/logs/eval_W_$RUN.log 2>&1
echo "DONE $RUN"
