#!/bin/bash
# Early review during training: quick-screen chosen step checkpoints of a run as soon as they are written.
#   scripts/early_review.sh <run> <wise|none> <step> [<step> ...]     e.g. scripts/early_review.sh M4_curr 0.5 4000 8000 12000
# Appends one line per checkpoint to exp/logs/early_review_<run>.log (screen EER, delta vs E2b, REJECT if >= +0.3).
cd "$(dirname "$0")/.." && source env.sh
RUN=$1; WISE=$2; shift 2
W=""; [ "$WISE" = none ] || W="--wise $WISE"
for st in "$@"; do
  # wait for the checkpoint or the end of training (last.pt is also written at every --eval-every, so not a signal)
  until [ -f exp/ft/$RUN/step$st.pt ] || grep -q "done best_dev_eer" exp/logs/$RUN.log 2>/dev/null; do sleep 60; done
  [ -f exp/ft/$RUN/step$st.pt ] || continue
  sleep 30  # let the writer finish
  .venv/bin/python scripts/quick_screen.py score exp/ft/$RUN/step$st.pt $W 2>&1 | grep -E "screen EER|Error" \
    >> exp/logs/early_review_$RUN.log
done
echo "EARLY_REVIEW_DONE $RUN" >> exp/logs/early_review_$RUN.log
