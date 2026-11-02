#!/bin/bash
# Public Test benchmark of every submittable checkpoint (clean official audio; no Silver, no private statistics).
#   experiments/distill_noiseaug_durationaware/public_bench.sh [PARALLEL=3]
# Per model: (WiSE file if needed) -> posttrain_embed.py on the clean Public Test + train cohort -> public_bench.py.
# Outputs: runs/_calib/PUB_<name>/posttrain.npz, runs/_eval/public_bench.json, runs/public_bench.log.
export LC_ALL=C
cd "$(dirname "$0")/../.." && source env.sh
set -euo pipefail
E=experiments/distill_noiseaug_durationaware; PY=.venv/bin/python; R=$E/runs
export SILVER_DIR=$SRED_DATA/public_test/data
INIT=$E/artifacts/e5_init_lowrank_step6000.pt
P=${1:-3}

# name  model file  (WiSE 0.7 partner: the compressed start for E5, E5 step 8000 for E8, as in the submission)
mkdir -p $R/_calib/_wise_pub
for s in 8000 10000; do
  w=$R/_calib/_wise_pub/E5s${s}_wise0.7.pt
  [ -f $w ] || $PY $E/backend.py wise $R/E5-b6-lowrank-kd/step$s.pt $INIT $w --alpha 0.7 | tail -1
done
MODELS=(
  "E5s4000 $R/E5-b6-lowrank-kd/step4000.pt"
  "E5s6000 $R/E5-b6-lowrank-kd/step6000.pt"
  "E5s8000 $R/E5-b6-lowrank-kd/step8000.pt"
  "E5s10000 $R/E5-b6-lowrank-kd/step10000.pt"
  "E5s6000_wise0.7 $R/E5-b6-lowrank-kd/wise0.7_step6000.pt"
  "E5s8000_wise0.7 $R/_calib/_wise_pub/E5s8000_wise0.7.pt"
  "E5s10000_wise0.7 $R/_calib/_wise_pub/E5s10000_wise0.7.pt"
  "E8s5000 $R/E8-b6-lowrank-kd-continue/step5000.pt"
  "E8s5000_wise0.7 $R/_calib/E8s5000_wise0.7_A/wise0.7.pt"
)
one() {
  local name=$1 model=$2 tag=PUB_$1
  if [ ! -f $R/_calib/$tag/posttrain.npz ]; then
    echo "$(date +%T) embed $tag <- $model"
    $PY $E/posttrain_embed.py "$model" --tag $tag > $R/_calib/pub_$name.log 2>&1
  fi
  echo "$(date +%T) embedded $tag"
}
export -f one; export E PY R
printf '%s\n' "${MODELS[@]}" | xargs -P "$P" -L 1 bash -c 'one $0 $1'
$PY $E/public_bench.py $(printf 'PUB_%s ' $(printf '%s\n' "${MODELS[@]}" | cut -d' ' -f1)) 2>&1 | grep -v -i warn
echo "$(date +%T) public bench done"
