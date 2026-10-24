#!/bin/bash
# Run phases one after another on this GPU, e.g. night 1:
#   experiments/distill_noiseaug_durationaware/run.sh e0:E0 e1:E1
# Each argument is <config name>:<run name>; an existing run resumes from its last.pt. SEED=<n> sets the seed (default 0).
set -e
cd "$(dirname "$0")/../.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E=experiments/distill_noiseaug_durationaware
for job in "$@"; do
  cfg=${job%%:*}; name=${job##*:}
  mkdir -p $E/runs/$name
  echo "$(date '+%F %T') start $name ($cfg)"
  .venv/bin/python $E/train.py --config $E/configs/$cfg.json --name $name --seed ${SEED:-0} --resume > $E/runs/$name/train.out 2>&1 \
    || { echo "$(date '+%F %T') FAILED $name"; tail -5 $E/runs/$name/train.out; exit 1; }
  echo "$(date '+%F %T') done $name"
done
