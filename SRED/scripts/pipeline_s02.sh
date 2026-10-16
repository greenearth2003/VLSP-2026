#!/bin/bash
# S02 releases: (1) current W_S02 (trained without V speakers) -> immediate submission;
# (2) S02 recipe retrained on the FULL train -> second submission. Sequential, RAM-safe, restart-safe.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
W01=exp/ft/W01_wise_F03_M/avg.pt
release() {  # release NAME CKPT
  local name=$1 ckpt=$2 V=$(date +%d%m%y); local D=releases/version_${V}_$name
  [ -f $D/sred_${name}_public_${V}.zip ] && return 0
  mkdir -p $D && cp $ckpt $D/sred_${name}_redimnetM_${V}.pt
  scripts/make_submission.sh $SRED_DATA/public_test/public-test.txt $SRED_DATA/public_test/data public \
    $D/results.txt $D/sred_${name}_redimnetM_${V}.pt qmf_dur > exp/logs/release_$name.log 2>&1 || return 1
  (cd $D && zip -qj sred_${name}_public_${V}.zip results.txt && mv results.txt sred_${name}_public_${V}.txt)
  $PY -m sred.protocol $W01 $D/sred_${name}_redimnetM_${V}.pt --n-boot 300 --out exp/logs/clean_pdev.jsonl >> exp/logs/release_$name.log 2>&1
  echo "RELEASED $D"
}
release S02 exp/ft/W_S02_consistency/avg.pt || exit 1
if [ ! -f exp/ft/W_S02_full/avg.pt ]; then
  $PY -m sred.train --base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 6 --amp bf16 --steps 3000 \
    --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 --save-every 500 \
    --aug 1.0 --freeze-bn --seed 0 --resume --workers 4 --consistency 0.1 --cons-mu 0.5 --name S02_full \
    >> exp/logs/S02_full.log 2>&1 || exit 1
  $PY scripts/wise_interp.py exp/ft/S02_full/step2000.pt exp/ft/W_S02_full/avg.pt --alpha 0.5 >> exp/logs/S02_full.log 2>&1
fi
release S02full exp/ft/W_S02_full/avg.pt || exit 1
echo S02_PIPELINE_DONE
