#!/bin/bash
# Round C: C01 (DAME), C06 (keep-dominant), C03 (prototypical); each vs S04 on clean V. Sequential, restart-safe.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
BASE="--base redimnet_M_mix --train-from stage3 --bs 32 --amp bf16 --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 \
  --head-only-steps 200 --dev-trials 0 --eval-every 3000 --save-every 500 --aug 1.0 --freeze-bn --seed 0 --resume \
  --workers 4 --exclude-val"
run() {
  local name=$1; shift
  [ -f exp/ft/W_$name/avg.pt ] && return 0
  $PY -m sred.train $BASE --name $name "$@" >> exp/logs/$name.log 2>&1 || return 1
  $PY scripts/wise_interp.py exp/ft/$name/step2000.pt exp/ft/W_$name/avg.pt --alpha 0.5 >> exp/logs/$name.log 2>&1
}
run C01_dame --crop-sec 6 --dame --train-list lists/train_drop_bad.txt || exit 1
run C06_keepdom --crop-sec 3.0 --min-crop-sec 1.5 --train-list lists/train_keepdom.txt || exit 1
run C03_proto --crop-sec 3.0 --min-crop-sec 1.5 --pk 4 --proto 0.1 --train-list lists/train_drop_bad.txt || exit 1
$PY -m sred.valset exp/ft/W_S04_dropbad/avg.pt exp/ft/W_C01_dame/avg.pt exp/ft/W_C06_keepdom/avg.pt \
  exp/ft/W_C03_proto/avg.pt --n-boot 300 --out exp/logs/c_results.jsonl > exp/logs/c_val.log 2>&1
echo C_DONE
