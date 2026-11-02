# Final submissions (2026-11-02): reproduce System A and System B

Two Private Test submissions, neither using AS-norm or any cohort score normalization. Run everything from `SRED/`
after `source env.sh`, with `.venv/bin/python`. `E=experiments/distill_noiseaug_durationaware`.

| | System A | System B |
|---|---|---|
| extractor | `systemA_E8s5000_wise0.7.pt` = 0.7 E8 step 5000 + 0.3 E5 step 8000 | `systemB_E5s6000_wise0.7.pt` = 0.7 E5 step 6000 + 0.3 compressed init |
| scoring | quality back-end (cosine, speech, SNR, norm; whole+speech view) fitted on training speakers only | plain cosine of whole-clip embeddings |
| Public Test (49,993 trials) EER all / cleaned, minDCF | 4.449 / 2.852, 0.271 | 4.685 / 3.008, 0.267 |
| private zip sha256 (prefix) | `72067d597c4bf11a` | `fa541b52f711df7f` |

Weights in this folder (also listed in `SHA256SUMS`): both extractors (6,947,208 params each) and
`systemA_calibration_posttrain.npz` (training-speaker calibration embeddings for the System A back-end).
`systemB_E5s6000_wise0.7.pt` is bit-identical to `backend.py wise` rebuilt from `runs/E5-b6-lowrank-kd/step6000.pt`
and `artifacts/e5_init_lowrank_step6000.pt` (checked).

## 1. Training (as run on server B, L40S)

1. Teacher E4: `$E/run.sh e4:E4-b6-teacher-finetune` (configs/e4.json, 8000 steps); step 6000 is used below.
2. KD cache: `prep/cache_b6.py` on E4 step 6000 -> `artifacts/teacher_cache/e4s6000.npy/.json`.
3. Compression: `dnd/lowrank.py runs/E4-b6-teacher-finetune/step6000.pt artifacts/e5_init_lowrank_step6000.pt`
   (activation-whitened SVD, budget 6,950,000 -> 6,947,208 params).
4. E5 recovery (configs/e5.json, seed 0), in two legs as logged in `runs/l40s.log` / `runs/E5-b6-lowrank-kd/train.out`:
   - `train.py --config $E/configs/e5.json --name E5-b6-lowrank-kd --seed 0 --resume --max-steps 6000`
     (8000-step LR schedule, stopped at 6000 -> `step6000.pt`, the System B checkpoint);
   - resumed from step 6000 with a 10000-step schedule -> `step8000.pt`, `step10000.pt` (the exact resume command was not
     recorded; `train.out` shows `steps 10000` after `resumed from step 6000`, e.g. `--resume --set steps=10000`).
5. E8 restart (configs/e8.json, init = E5 step 8000, new head/optimizer/schedule): `SEED=1 $E/run.sh e8:E8-b6-lowrank-kd-continue`.
6. Interpolation: `$E/backend.py wise TRAINED PARTNER OUT --alpha 0.7`
   - A: TRAINED = `runs/E8-b6-lowrank-kd-continue/step5000.pt`, PARTNER = `runs/E5-b6-lowrank-kd/step8000.pt`
   - B: TRAINED = `runs/E5-b6-lowrank-kd/step6000.pt`, PARTNER = `artifacts/e5_init_lowrank_step6000.pt`

## 2. Checkpoint selection (plain cosine, clean Public Test)

Audio: `gs://cake-ds-intern/VLSP-2026/SRED/Dataset/Vi-SRED26-PublicTest.zip` unzipped to `$SRED_DATA/public_test/`.

    $E/public_fast.py embed --shard i --nshards N --group G --models ...   # one process per shard
    $E/public_fast.py rank                                                # -> runs/_eval/public_fast.json
    $E/public_fast.py backend E8s5000_wise0.7 E8s5000_wise0.7_A           # System A back-end on Public Test

## 3. Private Test submissions (`$E/private_submit/`)

    P=$E/private_submit
    # System A: embeddings with speech view + qualities, then back-end fitted on the training cohort
    for i in 0 1 2 3; do .venv/bin/python $P/priv_cos.py embed --shard $i --nshards 4 --tag E8w --models E8s5000_wise0.7 & done; wait
    .venv/bin/python $P/priv_cal.py 4        # prints the Public Test check (4.449) first
    # System B: plain cosine
    for i in 0 1 2 3; do .venv/bin/python $P/priv_cos.py embed --shard $i --nshards 4 --tag E5 --models E5s6000_wise0.7 & done; wait
    .venv/bin/python $P/priv_cos.py score --nshards 4 --tag E5 --models E5s6000_wise0.7
    # format/order checks + zip as results.txt
    $P/pack.sh $E/runs/_submit_cos/results_cal_E8s5000_wise0.7.txt  result-private-A.zip
    $P/pack.sh $E/runs/_submit_cos/results_cos_E5s6000_wise0.7.txt  result-private-B.zip

`public_fast.MODELS` gives the model paths the scripts load (System A: `runs/_calib/E8s5000_wise0.7_A/wise0.7.pt`,
System B: `runs/E5-b6-lowrank-kd/wise0.7_step6000.pt`); copy the release files there, or run step 1.
System A also needs `runs/_calib/E8s5000_wise0.7_A/posttrain.npz` (= `systemA_calibration_posttrain.npz`).
