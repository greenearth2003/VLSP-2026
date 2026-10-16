#!/bin/bash
# Usage: scripts/make_submission.sh <trial_list> <audio_root> <split_name> <out.txt> [model] [backend]
# Example (private test):
#   scripts/make_submission.sh $SRED_DATA/private_test/private-test.txt $SRED_DATA/private_test/data private submissions/private.txt
set -e
cd "$(dirname "$0")/.." && source env.sh
MODEL=${5:-redimnet_M_mix}
BACKEND=${6:-center+qmf_dur}
.venv/bin/python -m sred.evaluate --model "$MODEL" --trials "$1" --audio-root "$2" --split "$3" \
  --submission "$4" --submission-backend "$BACKEND" --tag "submit_$3"
# sanity: same pairs, same order, finite scores
paste -d' ' <(cut -d' ' -f1,2 "$4") <(awk '{print $1, $2}' "$1") | awk '$1!=$3||$2!=$4{bad++} END{print "mismatched lines:", bad+0}'
awk 'NF!=3 || $3!=$3+0 {bad++} END{print "bad score lines:", bad+0}' "$4"
wc -l "$4" "$1"
