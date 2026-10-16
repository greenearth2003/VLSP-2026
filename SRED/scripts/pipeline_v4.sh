#!/bin/bash
# Round v4 (EXPERIMENTS.md). Waits for v3, restart-safe.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=exp/logs/v4_results.jsonl
until grep -q V3_DONE exp/logs/pipeline_v3.out; do sleep 60; done
BASE="--base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 3000 \
  --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 --save-every 500 \
  --aug 1.0 --freeze-bn --seed 0 --resume --exclude-val"
run() {
  local name=$1; shift
  [ -f exp/ft/W_$name/avg.pt ] && return 0
  $PY -m sred.train $BASE --name $name "$@" >> exp/logs/v4_$name.log 2>&1 || return 1
  $PY scripts/wise_interp.py exp/ft/$name/step2000.pt exp/ft/W_$name/avg.pt --alpha 0.5 >> exp/logs/v4_$name.log 2>&1
}
val() { $PY -m sred.valset "$@" --n-boot 300 --out $OUT >> exp/logs/v4_val.log 2>&1; }
pre() { $PY -c "import sys; from sred.valset import ValSet; ValSet().embeddings('$1')" >> exp/logs/v4_extract.log 2>&1; }

pre redimnet_M_mix &
run S00_base || exit 1
pre exp/ft/W_S00_base/avg.pt && pre "exp/ft/W_S00_base/avg.pt::mc" &
run S01_balanced --crop-sec 6 --crop-dist balanced || exit 1
pre exp/ft/W_S01_balanced/avg.pt &
run S02_consistency --crop-sec 6 --consistency 0.1 --cons-mu 0.5 || exit 1
pre exp/ft/W_S02_consistency/avg.pt &
run S03_bal_pk_supcon --crop-sec 6 --crop-dist balanced --pk 4 --supcon 0.1 || exit 1
wait
[ -f exp/logs/v4_val.done ] || { val exp/ft/W_S00_base/avg.pt redimnet_M_mix "exp/ft/W_S00_base/avg.pt::mc" \
    exp/ft/W_S01_balanced/avg.pt exp/ft/W_S02_consistency/avg.pt exp/ft/W_S03_bal_pk_supcon/avg.pt && \
  val exp/ft/W_S01_balanced/avg.pt exp/ft/W_S03_bal_pk_supcon/avg.pt && touch exp/logs/v4_val.done; }
echo V4_TRAIN_DONE
