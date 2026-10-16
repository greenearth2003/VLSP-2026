#!/bin/bash
# Reproduce the S04+QS submission from scratch (see docs/version_161026_S04qs_report.md).
# Usage: scripts/reproduce_s04qs.sh [RUN_NAME]
set -e
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
RUN=${1:-S04_repro}

# 1) pretrained ReDimNet-M embeddings of all 58,404 train utterances (~70 min GPU, cached)
$PY eda/extract_train_emb.py
# 2) cleaned train list (drop mixed-voice IDs, suspicious labels, cross-ID duplicates) -> lists/train_drop_bad.txt
$PY eda/make_drop_bad_list.py
# 3) validation speakers V (300 train speakers, seed 0) are excluded from training (--exclude-val);
#    lists/val_v4.json is created on first use by sred.valset.build()
$PY -c "from sred.valset import build; build()"
# 4) fine-tune (F03 hyper-parameters, fixed schedule, no test data touched)
$PY -m sred.train --base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 \
  --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 \
  --save-every 500 --aug 1.0 --freeze-bn --seed 0 --resume --workers 4 --exclude-val \
  --train-list lists/train_drop_bad.txt --name $RUN > exp/logs/$RUN.log 2>&1
# 5) WiSE-FT: 0.5 * step-2000 checkpoint + 0.5 * pretrained ReDimNet-M
$PY scripts/wise_interp.py exp/ft/$RUN/step2000.pt exp/ft/W_$RUN/avg.pt --alpha 0.5
# 6) scores with QS calibration + zip
V=$(date +%d%m%y); D=releases/version_${V}_${RUN}qs; mkdir -p $D
cp exp/ft/W_$RUN/avg.pt $D/sred_${RUN}qs_redimnetM_${V}.pt
$PY scripts/make_submission_qs.py $SRED_DATA/public_test/public-test.txt $SRED_DATA/public_test/data public \
  $D/results.txt $D/sred_${RUN}qs_redimnetM_${V}.pt
(cd $D && zip -j sred_${RUN}qs_public_${V}.zip results.txt && mv results.txt sred_${RUN}qs_public_${V}.txt)
