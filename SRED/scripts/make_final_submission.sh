#!/bin/bash
# Final system -> leaderboard zip, for the private test (or any trial list).
#
#   scripts/make_final_submission.sh <trial_list> <audio_root> <zip_name> [split_name]
#   TRANSDUCTIVE=1 scripts/make_final_submission.sh ...   # enable transductive kNN smoothing (default off;
#        only files in clusters of >= MIN_CLUSTER (default 3) test files are smoothed; check the rules first)
#
#   e.g. scripts/make_final_submission.sh $SRED_DATA/private_test/private-test.txt \
#            $SRED_DATA/private_test/data result-private-1020.zip private
#
# System: K21_M = ReDimNet-M (4.81M params) distilled from fine-tuned ReDimNet2-B6 on the cleaned train
# list (18k steps, long->short KD), WiSE-FT 0.5 of step 12000 with the pretrained M; cosine + QMF(dur).
# Public test: EER 4.50 % full / 2.92 % cleaned.
#
# Output format (as accepted by the leaderboard, see the sred_*_public_*.zip reference):
#   <zip_name> containing a single results.txt at the root; lines "file1 file2 score", space separated,
#   no header, same order as the trial list.
set -euo pipefail
export LC_ALL=C   # awk number parsing: a locale with decimal comma (e.g. vi_VN) fails every score check
cd "$(dirname "$0")/.." && source env.sh

MODEL=${FINAL_MODEL:-exp/ft/W_K21_M/avg.pt}
BACKEND=qmf_dur
TRIALS=$1; AUDIO=$2; ZIP=$3; SPLIT=${4:-private}
OUT=submissions/final_${SPLIT}
mkdir -p "$OUT"

[ -f "$MODEL" ] || { echo "missing checkpoint $MODEL (exp/ft is git-ignored; restore it from backup)"; exit 1; }
[ -d "$AUDIO" ] || { echo "missing audio dir $AUDIO"; exit 1; }

# use only the first two columns: an unlabeled list may carry a placeholder third column
awk 'NF>=2 {print $1, $2}' "$TRIALS" > "$OUT/trials_2col.txt"
n=$(wc -l < "$OUT/trials_2col.txt")
missing=$(awk '{print $1"\n"$2}' "$OUT/trials_2col.txt" | sort -u | while read -r f; do [ -f "$AUDIO/$f" ] || echo "$f"; done | wc -l)
echo "trials: $n   missing audio files: $missing"
[ "$missing" -eq 0 ] || { echo "abort: audio files missing"; exit 1; }
rm -f "lists/${SPLIT}_durations.npy"   # duration cache is keyed by split name only

if [ "${TRANSDUCTIVE:-0}" = "1" ]; then
  .venv/bin/python scripts/score_transductive.py --model "$MODEL" --trials "$OUT/trials_2col.txt" --audio-root "$AUDIO" \
    --split "$SPLIT" --out "$OUT/results.txt" --transductive --min-cluster "${MIN_CLUSTER:-3}"
else
  .venv/bin/python -m sred.evaluate --model "$MODEL" --trials "$OUT/trials_2col.txt" --audio-root "$AUDIO" \
    --split "$SPLIT" --submission "$OUT/results.txt" --submission-backend "$BACKEND" --tag "final_$SPLIT"
fi

# checks: same pairs in the same order, 3 fields, finite numeric scores
bad_order=$(paste -d' ' <(cut -d' ' -f1,2 "$OUT/results.txt") "$OUT/trials_2col.txt" | awk '$1!=$3||$2!=$4{b++} END{print b+0}')
bad_lines=$(awk 'NF!=3 || $3!=$3+0 || $3=="nan" || $3=="inf" || $3=="-inf" {b++} END{print b+0}' "$OUT/results.txt")
lines=$(wc -l < "$OUT/results.txt")
echo "lines: $lines / $n   mismatched pairs: $bad_order   bad score lines: $bad_lines"
[ "$lines" -eq "$n" ] && [ "$bad_order" -eq 0 ] && [ "$bad_lines" -eq 0 ] || { echo "abort: checks failed"; exit 1; }

rm -f "$OUT/$ZIP" && (cd "$OUT" && zip -q -j "$ZIP" results.txt)
unzip -l "$OUT/$ZIP"
echo "ready: $OUT/$ZIP"
