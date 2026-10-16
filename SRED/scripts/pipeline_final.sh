#!/bin/bash
# Final: pick recipe on V (finalize_v4.py) -> retrain on FULL train -> WiSE 0.5 -> P-dev sign check vs W01 -> release.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
W01=exp/ft/W01_wise_F03_M/avg.pt
until grep -q V4B_DONE exp/logs/pipeline_v4b.out && grep -q V4_TRAIN_DONE exp/logs/pipeline_v4.out; do sleep 60; done
ARGS=$($PY scripts/finalize_v4.py)
echo "decision: $ARGS"
if [ "$ARGS" = "NONE" ]; then echo "FINAL_DONE keep W01"; exit 0; fi
if [ ! -f exp/ft/W_FINAL_v4/avg.pt ]; then
  $PY -m sred.train --base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 \
    --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 \
    --save-every 500 --aug 1.0 --freeze-bn --seed 0 --resume --name FINAL_v4 $ARGS >> exp/logs/final_v4.log 2>&1 || exit 1
  $PY scripts/wise_interp.py exp/ft/FINAL_v4/step2000.pt exp/ft/W_FINAL_v4/avg.pt --alpha 0.5 >> exp/logs/final_v4.log 2>&1
fi
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
