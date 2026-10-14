#!/bin/bash
# Reproduce the final system W01 (= 0.5 * F03@step2000 + 0.5 * pretrained ReDimNet-M) from scratch.
# Usage: scripts/reproduce_w01.sh [RUN_NAME]
set -e
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=${1:-F03_repro}

# 0) train-cohort embeddings of the pretrained model (used to initialise the AAM head + fit QMF)
.venv/bin/python -m sred.protocol --extract redimnet_M_mix > exp/logs/repro_cohort.log 2>&1

# 1) F03 recipe. The original run kept the step-2000 checkpoint (chosen on a public subset);
#    here no test data is touched: every 500 steps is saved and step 2000 is used directly.
.venv/bin/python -m sred.train --base redimnet_M_mix --name $RUN --train-from stage3 \
  --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 \
  --head-only-steps 200 --dev-trials 0 --eval-every 3000 --save-every 500 --aug 1.0 --freeze-bn --seed 0 \
  > exp/logs/$RUN.log 2>&1

# 2) WiSE-FT interpolation with the pretrained weights
.venv/bin/python scripts/wise_interp.py exp/ft/$RUN/step2000.pt exp/ft/W01_$RUN/avg.pt --alpha 0.5

# 3) public-test scores with duration-aware QMF calibration
mkdir -p submissions/$RUN
scripts/make_submission.sh $SRED_DATA/public_test/public-test.txt $SRED_DATA/public_test/data public \
  submissions/$RUN/results.txt exp/ft/W01_$RUN/avg.pt qmf_dur
(cd submissions/$RUN && zip -j results.zip results.txt)
