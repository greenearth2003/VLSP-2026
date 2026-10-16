#!/bin/bash
# Sequential (RAM-safe) remainder of round v4 + final. Restart-safe; one heavy process at a time.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
W01=exp/ft/W01_wise_F03_M/avg.pt
BASE="--base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 3000 \
  --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 --save-every 500 \
  --aug 1.0 --freeze-bn --seed 0 --resume --workers 4"
run() {  # run NAME LOG [args]
  local name=$1 log=$2; shift 2
  [ -f exp/ft/W_$name/avg.pt ] && return 0
  $PY -m sred.train $BASE --name $name "$@" >> $log 2>&1 || return 1
  $PY scripts/wise_interp.py exp/ft/$name/step2000.pt exp/ft/W_$name/avg.pt --alpha 0.5 >> $log 2>&1
}
run S02_consistency exp/logs/v4_S02_consistency.log --exclude-val --crop-sec 6 --consistency 0.1 --cons-mu 0.5 || exit 1
run S04_dropbad exp/logs/v4_S04_dropbad.log --exclude-val --train-list lists/train_drop_bad.txt || exit 1
run S03_bal_pk_supcon exp/logs/v4_S03_bal_pk_supcon.log --exclude-val --crop-sec 6 --crop-dist balanced --pk 4 --supcon 0.1 || exit 1
echo V4_TRAIN_DONE
if [ ! -f exp/logs/v4_val.done ]; then
  $PY -m sred.valset exp/ft/W_S00_base/avg.pt redimnet_M_mix exp/ft/W_S01_balanced/avg.pt exp/ft/W_S02_consistency/avg.pt \
    exp/ft/W_S03_bal_pk_supcon/avg.pt exp/ft/W_S04_dropbad/avg.pt "exp/ft/W_S00_base/avg.pt::mc" \
    --n-boot 300 --out exp/logs/v4_results.jsonl >> exp/logs/v4_val.log 2>&1 || exit 1
  $PY -m sred.valset exp/ft/W_S01_balanced/avg.pt exp/ft/W_S03_bal_pk_supcon/avg.pt --n-boot 300 \
    --out exp/logs/v4_results.jsonl >> exp/logs/v4_val.log 2>&1 && touch exp/logs/v4_val.done
fi
echo V4_VAL_DONE
ARGS=$($PY scripts/finalize_v4.py); echo "decision: $ARGS"
[ "$ARGS" = "NONE" ] && { echo "FINAL_DONE keep W01"; exit 0; }
run FINAL_v4 exp/logs/final_v4.log $ARGS || exit 1
$PY -m sred.protocol $W01 exp/ft/W_FINAL_v4/avg.pt --extract --n-boot 300 --out exp/logs/v4_results.jsonl > exp/logs/final_pdev.log 2>&1
DIFF=$(grep '"cmp"' exp/logs/final_pdev.log | grep '"qmf"' | $PY -c "import sys,json; print(json.loads(sys.stdin.read())['diff'])")
echo "P-dev diff vs W01: $DIFF"
if $PY -c "import sys; sys.exit(0 if float('$DIFF') <= 0 else 1)"; then
  V=$(date +%d%m%y); D=releases/version_${V}_FINAL; mkdir -p $D
  cp exp/ft/W_FINAL_v4/avg.pt $D/sred_FINAL_redimnetM_${V}.pt
  scripts/make_submission.sh $SRED_DATA/public_test/public-test.txt $SRED_DATA/public_test/data public \
    $D/results.txt $D/sred_FINAL_redimnetM_${V}.pt qmf_dur >> exp/logs/final_v4.log 2>&1
  (cd $D && zip -j sred_FINAL_public_${V}.zip results.txt && mv results.txt sred_FINAL_public_${V}.txt)
  echo "FINAL_DONE new release $D"
else
  echo "FINAL_DONE keep W01 (new model worse on P-dev)"
fi
