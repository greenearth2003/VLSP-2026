#!/bin/bash
# C-series: same recipe as g_sweep.sh, but with an explicit train list (original vs cleaned) and
# head init from full-train centroids. Evaluates ft step3000 and WiSE-FT 0.5 of step2000.
# Usage: scripts/c_sweep.sh <base> <run_name> <train_from> <train_list> [extra sred.train args, override earlier ones]
set -e
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=$1; RUN=$2; FROM=$3; LIST=$4; shift 4; EXTRA="$@"
PY=.venv/bin/python
$PY -m sred.train --base $BASE --name $RUN --train-from $FROM --train-list $LIST \
  --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 \
  --head-only-steps 200 --dev-trials 0 --eval-every 3000 --save-every 500 --aug 1.0 --freeze-bn --seed 0 \
  --workers 10 $EXTRA > exp/logs/$RUN.log 2>&1
$PY -m sred.evaluate --model exp/ft/$RUN/last.pt --tag ft_$RUN > exp/logs/eval_$RUN.log 2>&1  # last.pt = final step
$PY scripts/wise_interp.py exp/ft/$RUN/step${WISE_STEP:-2000}.pt exp/ft/W_$RUN/avg.pt --alpha 0.5
$PY -m sred.evaluate --model exp/ft/W_$RUN/avg.pt --tag wise_$RUN > exp/logs/eval_W_$RUN.log 2>&1
echo "DONE $RUN"
