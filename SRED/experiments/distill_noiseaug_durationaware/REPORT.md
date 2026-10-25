---
title: "VLSP 2026 SRED: noisy two-view fine-tuning, low-rank compression and KD"
experiment: distill-noiseaug-durationaware
date: 2026-10-25
branch: thucth-distill-noiseaug-durationaware
code: SRED/experiments/distill_noiseaug_durationaware/
hf_repo: pieceofcakee/vlsp2026 (private dataset; artifacts, weights, results under the same path)
submission: submission-3.zip = E8-b6-lowrank-kd-continue step 5000 + final recipe
params: 6947208
silver_eer_all_cleaned_privlike: [5.109, 3.764, 5.656]
silver_minDCF: 0.375
---

# Technical report: distill-noiseaug-durationaware

## 1. Summary

**Submitted:** `E8-b6-lowrank-kd-continue` step 5000 (6.95M params) + final recipe. Silver EER (all / cleaned /
private-like) **5.109 / 3.764 / 5.656**, minDCF 0.375 (independent re-run on the second server: 5.105 / 3.755 / 5.624).
Earlier team best: 5.413 / 3.933 / 5.709. Teacher E4 (12.46M, not submittable): 4.645 / 3.079 / 4.868 with the recipe.

Pipeline: fine-tune the approved ReDimNet2-B6 on noisy two-view batches (teacher) → compress below 7M with an
activation-whitened SVD → recover with the same recipe + KD from the teacher → WiSE + train-fitted quality-aware
calibration. All components were built in this line; no teammate checkpoint or code is used.

## 2. Data and evaluation

- **Train:** cleaned official list, 1,823 speakers, 57,010 utterances; a fixed bank of 859,684 crops, so teacher
  embeddings are cached per crop.
- **Two-view batches:** 32 speakers/step × (short speech-dense crop 0.5–2 s, long random crop 2.5–6 s). Short bins
  cover the shorter side of 97 % of private trials. Batches are a function of (seed, step): exact resume (55 tests).
- **Noise on the fly** (allowed data only): RIRS_NOISES reverb (p 0.3); additive noise (p 0.6) from MUSAN
  noise/music/babble, RIRS point-source/isotropic and real train backgrounds (SNR 0–20 dB); shared background on
  speaker pairs (p 0.5); band-limit / EQ / μ-law. Teachers see the clean crop.
- **Silver** (`evaluate.py`, plain cosine unless stated): 49,993 public trials on `public_rumble_v2` audio; *all*,
  *cleaned* labels, *private-like* (frozen trial weights matching the private duration/noise profile), minDCF(0.01),
  natural speech-duration buckets. Seed noise ≈ 0.1 EER (E0 seeds 0/1: 7.755 / 7.639).

## 3. Method

- **E4 teacher:** B6 (`b6-vb2+vox2+cnc2_v0-lm`, 12.46M); SphereFace2 192-D head from speaker centroids, 300
  head-only steps; AdamW backbone 1e-5 / embedding 3e-5 / head 1e-3, warm-up 500, cosine to 5 %; BN frozen; 8k
  steps, best step 6000.
- **Compression** (`dnd/lowrank.py`): each conv/linear W → U·V; SVD of W·C^½ (C = input-activation covariance on
  train crops); ranks give every layer the same output-energy share. 6.95M params; output cosine to B6 0.815 (plain
  SVD 0.544; greedy rank allocation 0.31).
- **E5 recovery + KD:** compressed E4-s6000 trained like E4 + KD from cached E4-s6000 clean-crop embeddings:
  2.0·(1 − cos) via an identity-initialised projection + 1.0·relational SmoothL1 (β 0.1). Cache in fp32+TF32
  (bf16 failed a cos > 0.999 check). 10k steps.
- **E8 continuation:** fresh 5k-step run from E5 s8000 (new head, optimizer, schedule; seed 1).
- **Final recipe** (`recipe.sh`): WiSE θ = 0.7·θ_trained + 0.3·θ_start; QMF2 logistic calibration on cosine +
  min/max/products of speech duration, SNR and raw embedding norm, with AS-norm (top-300) as an extra input and the
  embedding = mean of whole-clip and speech-only views. Fitted on train-cohort trials only (5 utts/speaker, clean +
  noisy; same-file pairs excluded). Chosen once on E4, applied unchanged to E5/E8.

## 4. Main results (silver EER %)

| System | Params | All | Clean | Priv-like | minDCF | <0.75 s | >3 s |
|---|---|---|---|---|---|---|---|
| B4 pretrained | 6.66M | 13.96 | 12.75 | 14.25 | 0.652 | 22.3 | 6.14 |
| B6 pretrained | 12.46M | 8.80 | 7.28 | 8.82 | 0.402 | 17.5 | 3.85 |
| E0 B4 fine-tune (s6000) | 6.66M | 7.56 | 6.24 | 7.90 | 0.485 | 14.8 | 3.96 |
| E4 B6 teacher (s6000) | 12.46M | 5.35 | 3.90 | 5.95 | 0.352 | 12.6 | 3.11 |
| E5 low-rank + KD (s8000) | 6.95M | 5.81 | 4.34 | 6.32 | 0.378 | 13.3 | 3.32 |
| E8 continuation (s5000) | 6.95M | 5.81 | 4.33 | 6.31 | 0.383 | 13.3 | 3.32 |
| **E8 + final recipe (submitted)** | 6.95M | **5.11** | **3.76** | **5.66** | **0.375** | 12.9 | 2.98 |
| Earlier team best, cosine / + back-end | < 7M | 6.12 / 5.41 | 4.66 / 3.93 | 6.47 / 5.71 | 0.440 | 12.4 | 3.20 |

## 5. Ablations (silver EER %, all / clean / priv-like)

**5.1 Route to < 7M** (one change per row, plain cosine)

| Model | All / clean / priv-like | Δ all |
|---|---|---|
| B6 pretrained → plain weight SVD to 6.95M, no training | 15.78 / 14.77 / 16.45 | +6.98 |
| B6 pretrained → whitened SVD to 6.95M, no training | 9.70 / 8.33 / 9.62 | +0.90 (−6.08 vs plain) |
| E4 teacher → whitened SVD, no training (E5 init) | 6.43 / 5.06 / 6.62 | +1.08 vs teacher |
| + recovery with KD, 6k steps (E5 s6000) | 5.83 / 4.28 / 6.22 | −0.60 |
| Same budget, no teacher route: B4 fine-tune (E0) | 7.56 / 6.24 / 7.90 | +1.73 vs E5 |
| E1 = E0 + DAME prefix heads (B4) | 7.43 / 6.21 / 7.79; <0.75 s 16.0 vs 14.8 | rejected (short) |
| E4 with BN statistics re-estimated on noisy train | 8.33 / 6.98 / 9.16 | rejected |

**5.2 Training length** (cosine | final recipe)

| Checkpoint | Cosine | Final recipe |
|---|---|---|
| E5 s4000 / s6000 | 5.85 / 4.26 / 6.25 · 5.83 / 4.28 / 6.22 | – · 5.35 / 3.93 / 5.71 |
| E5 s8000 / s10000 | 5.81 / 4.34 / 6.32 · 5.84 / 4.38 / 6.38 | 5.33 / 3.90 / 5.70 · 5.35 / 3.87 / 5.75 |
| E8 s5000 (fresh restart from E5 s8000) | 5.81 / 4.33 / 6.31 | **5.11 / 3.76 / 5.66** |

Cosine EER saturates by 4k steps. The fresh restart leaves cosine unchanged but makes scores easier to calibrate
(recipe gain 0.70 vs 0.43; minDCF 0.375 vs 0.390), reproduced on both servers.

**5.3 Back-end on E4 s6000** (QMF2 = duration + SNR; each row adds one change; train-fitted only)

| Back-end | All / clean / priv-like | minDCF | Kept |
|---|---|---|---|
| plain cosine | 5.35 / 3.90 / 5.92 | 0.352 | |
| QMF duration only → QMF2 duration + SNR | 5.07 → 5.03 / 3.54 / 5.49 | 0.372 | base |
| + raw norm · + speech-only view · + AS-norm | 5.01 · 5.05 · 5.05 (clean 3.53 · 3.46 · 3.52) | 0.358 · 0.354 · 0.344 | ✓ all three |
| + short crops in fit · + centering | 5.07 · 5.24 | 0.346 · 0.378 | ✗ |
| boosted trees · per-duration-bin LR · 6 s chunking | 5.19 · 5.13 · 5.07 | – | ✗ |
| WiSE α 0.5 / 0.7 / 0.8, cosine | 5.31 / 5.09 / 5.08 (priv-like 5.40 / 5.34 / 5.44) | 0.317–0.321 | α 0.7 |
| **WiSE 0.7 + QMF2 + norm + speech + AS-norm** | **4.65 / 3.08 / 4.87** | 0.300 | final |

## 6. Submissions (private: 278,410 trials, 37,354 files)

| File | System | Silver all / clean / priv-like |
|---|---|---|
| `submission-3.zip` (recommended) | E8 s5000 + recipe | 5.11 / 3.76 / 5.66 |
| `submission-2.zip` | E5 s8000 + recipe | 5.33 / 3.90 / 5.70 |

One `results.txt` per zip ("fileA fileB score", trial order); line count, pair order and finite scores checked.
Submission-3 (sharded scorer on server B) agrees with an independent full re-run on server A at Spearman 0.999997
(mean |Δ| 0.014, score s.d. 8.8); vs submission-2 0.995, vs the earlier team best 0.969.

## 7. Reproduce and storage

```bash
cd SRED && source env.sh; E=experiments/distill_noiseaug_durationaware
hf download pieceofcakee/vlsp2026 --repo-type dataset --include "$E/*" --local-dir .
SEED=0 $E/run.sh e4:E4-b6-teacher-finetune            # teacher, >= 32 GB GPU
.venv/bin/python $E/prep/cache_b6.py --ckpt $E/runs/E4-b6-teacher-finetune/step6000.pt --out e4s6000
.venv/bin/python $E/dnd/lowrank.py $E/runs/E4-b6-teacher-finetune/step6000.pt $E/artifacts/e5_init_lowrank_step6000.pt
SEED=0 $E/run.sh e5:E5-b6-lowrank-kd; SEED=1 $E/run.sh e8:E8-b6-lowrank-kd-continue
$E/recipe.sh runs/E8-b6-lowrank-kd-continue/step5000.pt runs/E5-b6-lowrank-kd/step8000.pt E8s5000_wise0.7
```

- **git** (this branch): code, configs, this report, `NEW_PLAN.md`, `HANDOFF.md`.
- **HF** `$E/`: `artifacts/` (manifest, crop bank, centroids, teacher caches, compressed inits), `runs/<run>/` (config,
  logs, step*.pt, last.pt), `runs/_eval/*.json` (every number here), `runs/_calib/`, `runs/_submit/`, `submissions/`,
  `sync/` (decisions D1–D19, server logs), `tools_B/` (sharded private scorer).
- **W&B** (project `vlsp2026-sred`, group `distill-noiseaug-durationaware`): curves of E0, E0_s1, E1, E4, E5, E8 in
  one report: <https://forge.coreweave.com/thucth-qt/vlsp2026-sred/reports/distill-noiseaug-durationaware:-training-curves-(all-experiments)--VmlldzoxODE3NjgyMQ==>

## 8. Limitations and next steps

- Gains ≤ 0.1 EER are within seed noise; submission-3 beats the earlier best by only 0.05 on private-like.
- For E5 the calibrator hurts the < 0.75 s bucket (12.94 → 13.68); E8 does not show this.
- Untried: utterance-balanced sampling (speaker-uniform sampling had seen 83 % of utterances by step 6000), B4 with
  KD from E4 (E6), structured pruning instead of low-rank (E7).
