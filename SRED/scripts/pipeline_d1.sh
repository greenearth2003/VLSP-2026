#!/bin/bash
# D1 = S02 recipe (short-long consistency) on VLSP (drop-bad) + Vietnam-Celeb (leak/noise screened). Restart-safe.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
W01=exp/ft/W01_wise_F03_M/avg.pt
until grep -q VC_DONE exp/logs/vc_download.log; do sleep 60; done
[ -f lists/train_vlsp_vc.txt ] || $PY ext/vc_prepare.py > exp/logs/vc_prepare.log 2>&1 || exit 1
until grep -q S04_PIPELINE_DONE exp/logs/pipeline_s04.out && grep -q rescue_kept exp/logs/vc_rescue.log; do sleep 60; done
if [ ! -f exp/ft/W_D1_vc/avg.pt ]; then
  $PY -m sred.train --base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 6 --amp bf16 --steps 3000 \
    --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 --save-every 500 \
    --aug 1.0 --freeze-bn --seed 0 --resume --workers 4 --consistency 0.1 --cons-mu 0.5 \
    --train-list lists/train_vlsp_vc.txt --name D1_vc >> exp/logs/D1_vc.log 2>&1 || exit 1
  $PY scripts/wise_interp.py exp/ft/D1_vc/step2000.pt exp/ft/W_D1_vc/avg.pt --alpha 0.5 >> exp/logs/D1_vc.log 2>&1
fi
V=$(date +%d%m%y); D=releases/version_${V}_D1vc
if [ ! -f $D/sred_D1vc_public_${V}.zip ]; then
  mkdir -p $D && cp exp/ft/W_D1_vc/avg.pt $D/sred_D1vc_redimnetM_${V}.pt
  scripts/make_submission.sh $SRED_DATA/public_test/public-test.txt $SRED_DATA/public_test/data public \
    $D/results.txt $D/sred_D1vc_redimnetM_${V}.pt qmf_dur > exp/logs/release_D1vc.log 2>&1 || exit 1
  (cd $D && zip -qj sred_D1vc_public_${V}.zip results.txt && mv results.txt sred_D1vc_public_${V}.txt)
fi
$PY -m sred.protocol $W01 $D/sred_D1vc_redimnetM_${V}.pt --n-boot 300 --out exp/logs/clean_pdev.jsonl >> exp/logs/release_D1vc.log 2>&1
echo "RELEASED $D"
echo D1_PIPELINE_DONE
