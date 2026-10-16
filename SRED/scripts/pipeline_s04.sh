#!/bin/bash
# S04 releases: (1) current W_S04 (no V speakers) after the S02 quick release; (2) S04 recipe (F03 recipe on
# lists/train_drop_bad.txt) retrained on the full train after S02_full is done. Restart-safe.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
W01=exp/ft/W01_wise_F03_M/avg.pt
release() {
  local name=$1 ckpt=$2 V=$(date +%d%m%y); local D=releases/version_${V}_$name
  [ -f $D/sred_${name}_public_${V}.zip ] && return 0
  mkdir -p $D && cp $ckpt $D/sred_${name}_redimnetM_${V}.pt
  scripts/make_submission.sh $SRED_DATA/public_test/public-test.txt $SRED_DATA/public_test/data public \
    $D/results.txt $D/sred_${name}_redimnetM_${V}.pt qmf_dur > exp/logs/release_$name.log 2>&1 || return 1
  (cd $D && zip -qj sred_${name}_public_${V}.zip results.txt && mv results.txt sred_${name}_public_${V}.txt)
  $PY -m sred.protocol $W01 $D/sred_${name}_redimnetM_${V}.pt --n-boot 300 --out exp/logs/clean_pdev.jsonl >> exp/logs/release_$name.log 2>&1
  echo "RELEASED $D"
}
until grep -q "RELEASED .*_S02$" exp/logs/pipeline_s02.out; do sleep 60; done
release S04 exp/ft/W_S04_dropbad/avg.pt || exit 1
until grep -q S02_PIPELINE_DONE exp/logs/pipeline_s02.out; do sleep 60; done
if [ ! -f exp/ft/W_S04_full/avg.pt ]; then
  $PY -m sred.train --base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 \
    --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 \
    --save-every 500 --aug 1.0 --freeze-bn --seed 0 --resume --workers 4 --train-list lists/train_drop_bad.txt \
    --name S04_full >> exp/logs/S04_full.log 2>&1 || exit 1
  $PY scripts/wise_interp.py exp/ft/S04_full/step2000.pt exp/ft/W_S04_full/avg.pt --alpha 0.5 >> exp/logs/S04_full.log 2>&1
fi
release S04full exp/ft/W_S04_full/avg.pt || exit 1
echo S04_PIPELINE_DONE
