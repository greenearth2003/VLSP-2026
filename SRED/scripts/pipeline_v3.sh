#!/bin/bash
# Round v3 (EXPERIMENTS.md). Restart-safe: finished stages are skipped, training resumes from resume.pt.
cd "$(dirname "$0")/.." && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=exp/logs/v3_results.jsonl
BASE="--base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 3000 \
  --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 --save-every 500 \
  --aug 1.0 --freeze-bn --seed 0 --resume"
W01=exp/ft/W01_wise_F03_M/avg.pt

run() {  # run NAME [extra args]  -> exp/ft/W_NAME/avg.pt
  local name=$1; shift
  [ -f exp/ft/W_$name/avg.pt ] && return 0
  $PY -m sred.train $BASE --name $name "$@" >> exp/logs/v3_$name.log 2>&1 || return 1
  $PY scripts/wise_interp.py exp/ft/$name/step2000.pt exp/ft/W_$name/avg.pt --alpha 0.5 >> exp/logs/v3_$name.log 2>&1
}
extract() {  # background embedding extraction for a system spec
  $PY -c "import sys; sys.argv=['x']; from sred.protocol import Protocol, extract_system; extract_system('$1', Protocol())" \
    >> exp/logs/v3_extract.log 2>&1
}

# inference-only experiments on W01 (E1, E5) run alongside R0 training
( extract "$W01::vad" && \
  [ -f exp/logs/v3_infer.done ] || { $PY -m sred.protocol $W01 "$W01::vad" "$W01::f0" "$W01::vad::f0" --extract \
     --n-boot 300 --out $OUT > exp/logs/v3_infer.log 2>&1 && touch exp/logs/v3_infer.done; } ) &

run R0_base || exit 1
extract exp/ft/W_R0_base/avg.pt &
run E2_clean --train-list lists/train_clean.txt || exit 1
extract exp/ft/W_E2_clean/avg.pt &
run E3a_noaug --aug 0 || exit 1
extract exp/ft/W_E3a_noaug/avg.pt &
run E3b_augmatched --aug-profile matched || exit 1
extract exp/ft/W_E3b_augmatched/avg.pt &
run E4_short_spkbal --crop-sec 2.5 --min-crop-sec 1.0 --spk-balanced || exit 1
wait
[ -f exp/logs/v3_train.done ] || { $PY -m sred.protocol exp/ft/W_R0_base/avg.pt exp/ft/W_E2_clean/avg.pt \
  exp/ft/W_E3a_noaug/avg.pt exp/ft/W_E3b_augmatched/avg.pt exp/ft/W_E4_short_spkbal/avg.pt $W01 --extract \
  --n-boot 300 --out $OUT > exp/logs/v3_train.log 2>&1 && touch exp/logs/v3_train.done; }
echo V3_DONE
