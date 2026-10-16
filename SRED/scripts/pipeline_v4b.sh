#!/bin/bash
# S04: S00 recipe on train without mixed-voice IDs / suspicious / cross-ID duplicates. Runs after v4.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
if [ ! -f exp/ft/W_S04_dropbad/avg.pt ]; then
  $PY -m sred.train --base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 \
    --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 \
    --save-every 500 --aug 1.0 --freeze-bn --seed 0 --resume --exclude-val --train-list lists/train_drop_bad.txt \
    --name S04_dropbad >> exp/logs/v4_S04_dropbad.log 2>&1 || exit 1
  $PY scripts/wise_interp.py exp/ft/S04_dropbad/step2000.pt exp/ft/W_S04_dropbad/avg.pt --alpha 0.5 >> exp/logs/v4_S04_dropbad.log 2>&1
fi
until [ -f exp/emb/W_S00_base_avg/$(ls exp/emb/W_S00_base_avg 2>/dev/null | grep -m1 valv4) ] && grep -q V4_TRAIN_DONE exp/logs/pipeline_v4.out; do sleep 60; done
$PY -m sred.valset exp/ft/W_S00_base/avg.pt exp/ft/W_S04_dropbad/avg.pt --n-boot 300 --out exp/logs/v4_results.jsonl >> exp/logs/v4_val.log 2>&1
echo V4B_DONE
