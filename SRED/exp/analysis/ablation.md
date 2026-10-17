# Ablation — VLSP 2026 SRED final system (public test)

EER % (lower is better). full = organisers' 49,993 trials (leaderboard metric); cleaned = 47,932 trials after removing label-noise trials; P-dev / P-hold = speaker-disjoint halves of cleaned; minDCF(0.01) on full; short = cleaned trials with < 1 s speech on the shorter side.

## 1. Build-up (each row adds one component)

| system | params_M | full | cleaned | P_dev | P_hold | minDCF | short_lt1s |
|---|---|---|---|---|---|---|---|
| A. pretrained ReDimNet-M, cosine | 4.81 | 5.87 | 4.33 | 3.58 | 4.71 | 0.35 | 7.70 |
| B. A + QMF duration calibration | 4.81 | 5.15 | 3.66 | 3.23 | 3.81 | 0.33 | 7.46 |
| C. B + fine-tune (AAM + aug, 3k steps, stem..stage2 frozen) | 4.81 | 5.03 | 3.44 | 3.12 | 3.53 | 0.32 | 7.25 |
| D. C + cleaned train list | 4.81 | 4.85 | 3.28 | 3.23 | 3.24 | 0.31 | 6.88 |
| E. D + long->short KD from fine-tuned B6, all layers, 18k steps | 4.81 | 4.62 | 3.15 | 3.11 | 3.09 | 0.31 | 6.16 |
| F. E + WiSE-FT 0.5 (step 12k ⊕ pretrained)  = submitted K21 | 4.81 | 4.50 | 2.92 | 2.95 | 2.86 | 0.27 | 5.82 |
| G. F + transductive kNN smoothing (k20 w0.5 tau0.6 x2) | 4.81 | 3.78 | 1.54 | 1.81 | 1.36 | 0.18 | 3.66 |

## 2. Leave-one-out / variants of the submitted system F

| system | params_M | full | cleaned | P_dev | P_hold | minDCF | short_lt1s | note |
|---|---|---|---|---|---|---|---|---|
| F without WiSE (= E) | 4.81 | 4.62 | 3.15 | 3.11 | 3.09 | 0.31 | 6.16 |  |
| F without KD (same schedule, all layers, 18k, cleaned) + WiSE | 4.81 |  |  |  |  |  |  | (not run / no cache) |
| F without cleaning (original train.txt) + WiSE | 4.81 |  |  |  |  |  |  | (not run / no cache) |
| F with KD but early layers frozen, 3k steps (K10) + WiSE | 4.81 | 4.83 | 3.27 | 3.04 | 3.34 | 0.30 | 7.04 |  |
| F with weaker teacher C01_B6 (K11, 9k) + WiSE | 4.81 | 4.73 | 3.13 | 2.89 | 3.17 | 0.29 | 6.51 |  |
| F with KD weights x1 (K20, 9k) + WiSE | 4.81 | 4.66 | 3.07 | 2.96 | 3.02 | 0.29 | 6.42 |  |
| F with 9k steps (K22) + WiSE | 4.81 | 4.64 | 3.02 | 2.92 | 3.00 | 0.28 | 6.33 |  |
| F with 36k steps (K23) + WiSE | 4.81 | 4.53 | 2.92 | 2.88 | 2.87 | 0.27 | 5.85 |  |
| F + adaptive LMFT stage (L02) + WiSE | 4.81 | 4.50 | 2.87 | 2.83 | 2.81 | 0.27 | 5.81 |  |
| F + duration-adaptive margin (K26) + WiSE | 4.81 | 4.52 | 2.91 | 2.89 | 2.89 | 0.27 | 5.90 |  |

## 3. Other students and the teacher

| system | params_M | full | cleaned | P_dev | P_hold | minDCF | short_lt1s |
|---|---|---|---|---|---|---|---|
| ReDimNet-S, same recipe as D (C01_S) | 3.15 | 4.92 | 3.44 | 3.25 | 3.39 | 0.32 | 6.56 |
| ReDimNet2-B4, KD recipe (K30) + WiSE | 6.66 | 4.96 | 3.15 | 3.63 | 3.28 | 0.30 | 6.65 |
| B6 pruned to 6.33M + 18k recovery (P01) | 6.33 | 5.33 | 3.93 | 3.23 | 4.24 | 0.35 | 7.16 |
| teacher: fine-tuned B6 + WiSE (not submittable) | 12.46 | 4.19 | 2.31 | 2.40 | 2.27 | 0.22 | 4.10 |

All rows: cosine + QMF(duration) back-end fitted on train-cohort trials (no test labels); WiSE-FT = 0.5 fine-tuned + 0.5 pretrained weights. Decisions were made on P-dev (paired speaker bootstrap, accept if P(better) > 0.95); P-hold is the check. Transductive row uses unlabelled test audio only.
