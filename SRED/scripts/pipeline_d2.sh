#!/bin/bash
# D2a = S04 recipe (F03 hparams, 1.5-3 s crops) on VLSP(drop-bad)+Vietnam-Celeb, 6000 steps, step 6000 + WiSE 0.5.
# D2b = D2a + duration-balanced crops (S01: 55 % <= 2 s). Sequential, restart-safe.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
W01=exp/ft/W01_wise_F03_M/avg.pt
COMMON="--base redimnet_M_mix --train-from stage3 --bs 32 --amp bf16 --steps 6000 --lr 2e-5 --head-lr 2e-3 \
  --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 6000 --save-every 1000 --aug 1.0 --freeze-bn \
  --seed 0 --resume --workers 4 --train-list lists/train_vlsp_vc.txt"
run_release() {  # NAME [args]
  local name=$1; shift; local V=$(date +%d%m%y); local D=releases/version_${V}_$name
  if [ ! -f exp/ft/W_$name/avg.pt ]; then
    $PY -m sred.train $COMMON --name $name "$@" >> exp/logs/$name.log 2>&1 || return 1
    $PY scripts/wise_interp.py exp/ft/$name/step6000.pt exp/ft/W_$name/avg.pt --alpha 0.5 >> exp/logs/$name.log 2>&1
  fi
  if [ ! -f $D/sred_${name}_public_${V}.zip ]; then
    mkdir -p $D && cp exp/ft/W_$name/avg.pt $D/sred_${name}_redimnetM_${V}.pt
    scripts/make_submission.sh $SRED_DATA/public_test/public-test.txt $SRED_DATA/public_test/data public \
      $D/results.txt $D/sred_${name}_redimnetM_${V}.pt qmf_dur > exp/logs/release_$name.log 2>&1 || return 1
    (cd $D && zip -qj sred_${name}_public_${V}.zip results.txt && mv results.txt sred_${name}_public_${V}.txt)
  fi
  $PY -m sred.protocol $W01 $D/sred_${name}_redimnetM_${V}.pt --n-boot 300 --out exp/logs/clean_pdev.jsonl >> exp/logs/release_$name.log 2>&1
  echo "RELEASED $D"
}
run_release D2a --crop-sec 3.0 --min-crop-sec 1.5 || exit 1
run_release D2b --crop-sec 6 --crop-dist balanced || exit 1
echo D2_PIPELINE_DONE
