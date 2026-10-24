#!/bin/bash
# Final recipe (D15) on one checkpoint -> silver metrics + private submission zip. Runs on any server.
#
#   experiments/distill_noiseaug_durationaware/recipe.sh TRAINED.pt PARTNER.pt TAG
#   e.g. recipe.sh runs/E5-b6-lowrank-kd/step8000.pt artifacts/e5_init_lowrank_step6000.pt E5s8000_wise0.7
#        (paths relative to this folder or absolute; E4: partner = the approved pretrained B6 file)
#
# Steps: WiSE 0.7 (trained, partner) -> one embedding pass (silver + train cohort, posttrain_embed.py) -> QMF2 calibrator
# on train only (+norm +speech +asnorm, backend.py fit) -> silver metrics (cosine and calibrated) ->
# private scoring (backend.py score) -> checked zip with one results.txt.
# Inputs: silver audio in $SILVER_DIR (default SRED/data/public_rumble_v2), private test in $PRIVATE_DIR
# (default $SRED_DATA/private_test with private-test.txt and data/). Set SKIP_PRIVATE=1 for silver only.
# Outputs: runs/_calib/TAG/ (posttrain.npz, calibrator json + cohort npy), runs/_eval/final__TAG_qmf2_norm_speech_asnorm.json,
# runs/_submit/result-private-TAG.zip (+ results txt).
export LC_ALL=C
cd "$(dirname "$0")/../.." && source env.sh
set -euo pipefail
E=experiments/distill_noiseaug_durationaware; PY=.venv/bin/python
abspath() { case $1 in /*) echo "$1";; *) [ -e "$1" ] && echo "$PWD/$1" || echo "$PWD/$E/$1";; esac; }
TRAINED=$(abspath "$1"); PARTNER=$(abspath "$2"); TAG=$3
PRIVATE_DIR=${PRIVATE_DIR:-$SRED_DATA/private_test}
CALDIR=$E/runs/_calib/$TAG; CAL=$CALDIR/qmf2_norm_speech_asnorm.json
MODEL=$CALDIR/wise0.7.pt; SUB=$E/runs/_submit; ZIP=result-private-$TAG.zip
mkdir -p "$CALDIR" "$SUB" "$E/runs/_eval"

if [ ! -f "$CAL" ]; then
  $PY $E/backend.py wise "$TRAINED" "$PARTNER" "$MODEL" --alpha 0.7 | tail -1
  echo "$(date +%T) embedding pass $TAG (silver 17,850 files + cohort 8,657 files)"
  $PY $E/posttrain_embed.py "$MODEL" --tag "$TAG" 2>&1 | grep -v -i warn
  $PY $E/backend.py fit --tag "$TAG" --out "runs/_calib/$TAG/qmf2_norm_speech_asnorm.json" --norm --speech --asnorm 2>&1 | grep -v -i warn | tail -1
  $PY - "$TAG" <<'PYEOF'
import sys, json, numpy as np, pandas as pd
E = 'experiments/distill_noiseaug_durationaware'
sys.path[:0] = [E, '.']
import evaluate as ev
from posttrain_eval import run, l2n
from calibrate_eval import metrics
tag = sys.argv[1]
D = dict(np.load(f'{E}/runs/_calib/{tag}/posttrain.npz'))
tr = pd.read_csv(ev.DEV / 'trials.csv'); y = tr.label.to_numpy(); sw = np.load(ev.DEV / 'silver_weights.npz')
pos = {f: i for i, f in enumerate(D['silver_files'])}
ia, ib = np.array([pos[f] for f in tr.a]), np.array([pos[f] for f in tr.b])
se = l2n(D['silver_whole'])
cos = metrics(np.sum(se[ia] * se[ib], 1), tr, y, sw['private_like'], sw['in_cleaned'])
r = run(D, tr, ia, ib, y, sw['private_like'], sw['in_cleaned'], norm=True, speech=True, asnorm=True)
json.dump({'wise0.7_cosine': cos, 'wise0.7_calibrated': r},
          open(f'{E}/runs/_eval/final__{tag}_qmf2_norm_speech_asnorm.json', 'w'), indent=1)
for k, v in ((f'{tag} cosine', cos), (f'{tag} + calibrator (FINAL)', r)):
    print(k + ':', ' '.join(f'{a}={b:.3f}' for a, b in v.items() if isinstance(b, float)))
PYEOF
else
  echo "calibrator exists: $CAL (delete runs/_calib/$TAG to redo)"
fi
[ "${SKIP_PRIVATE:-0}" = 1 ] && { echo "$(date +%T) $TAG done (private skipped)"; exit 0; }

echo "$(date +%T) private scoring $TAG"
awk 'NF>=2 {print $1, $2}' "$PRIVATE_DIR/private-test.txt" > "$SUB/trials_2col.txt"
RES=$SUB/results_$TAG.txt
$PY $E/backend.py score "$MODEL" --calib "$CAL" --trials "$SUB/trials_2col.txt" --audio-dir "$PRIVATE_DIR/data" \
  --out "$RES" 2>&1 | grep -v -i warn | tail -1
n=$(wc -l < "$SUB/trials_2col.txt")
bad_order=$(paste -d' ' <(cut -d' ' -f1,2 "$RES") "$SUB/trials_2col.txt" | awk '$1!=$3||$2!=$4{b++} END{print b+0}')
bad_lines=$(awk 'NF!=3 || $3!=$3+0 || $3=="nan" || $3=="inf" || $3=="-inf" {b++} END{print b+0}' "$RES")
echo "lines: $(wc -l < "$RES") / $n   mismatched pairs: $bad_order   bad score lines: $bad_lines"
[ "$(wc -l < "$RES")" -eq "$n" ] && [ "$bad_order" -eq 0 ] && [ "$bad_lines" -eq 0 ] || { echo "abort: checks failed"; exit 1; }
tmp=$(mktemp -d) && cp "$RES" "$tmp/results.txt" && rm -f "$SUB/$ZIP" && (cd "$tmp" && zip -q -j "$OLDPWD/$SUB/$ZIP" results.txt) && rm -r "$tmp"
echo "$(date +%T) ready: $SUB/$ZIP ($(sha256sum "$SUB/$ZIP" | cut -c1-16))"
