# Experiments — VLSP 2026 SRED

Metric: EER (%) on the **public test** (49,993 trials, labelled). "dev2000" = fixed random
2,000-trial subset of public used for checkpoint selection during training (cosine scoring);
"sub1500" = fixed 1,500-trial subset for quick model screening.

## Approved pretrained checkpoints (organiser list)

| model | params | role here | source used (GitHub release assets are blocked here) |
|---|---|---|---|
| ReDimNet-S `S-vb2+vox2+cnc-ft_mix` | 3.15M | candidate | HF `voice-is-cool/ReDimNet` (same file) |
| ReDimNet-M `M-vb2+vox2+cnc-ft_mix` | 4.81M | **main student** | HF `voice-is-cool/ReDimNet` (same file) |
| WeSpeaker ResNet34-LM | 6.63M | candidate | HF `Wespeaker/wespeaker-voxceleb-resnet34-LM` |
| ReDimNet2-B4 `b4-vox2-lm` | 6.66M | candidate | HF `causal/redimnet2-b4-vox2-lm-mlx` (MLX→torch, parity 4e-6) |
| ReDimNet2-B6 `b6-vb2+vox2+cnc2_v0-lm` | 12.46M | teacher only | HF `OpenASR/redimnet2-b6-cn` (GGUF, fp16 weights → torch) |
| XLS-R 300M, WavLM-Large | 300M+ | teacher only (not used yet) | — |

## Zero-shot (frozen, full utterance)

| ID | model | sub1500 cos | public cos | public best back-end |
|---|---|---|---|---|
| E00 | ReDimNet-M mix | 6.68 | 5.87 | 5.50 (mean-centering) |
| E01 | ReDimNet2-B4 vox2 | 10.40 | – | – |
| E03 | ReDimNet-S mix | 7.48 | – | – |
| E02 | WeSpeaker ResNet34-LM | 10.01 | – | – |
| T00 | ReDimNet2-B6 mix (teacher) | 5.53 | – | – |

Back-ends on E00 (cohort = 5 utts/spk of train, 9.5k embeddings):
cosine 5.87 · centering 5.50 · AS-norm(300) 5.65 · LDA-128 5.35 · WCCN 5.51 · whitening 5.94 · PLDA 6.39 ·
**centering + QMF(duration) 5.00** (`submissions/E00_qmf_public.txt`).

QMF = logistic regression on [s, log dmin, log dmax, s·log dmin, s·log dmax], fitted on ~320k
synthetic trials from train-cohort embeddings (no public labels). Adding AS-norm features looked
better (4.97) only due to a bug (same-speaker cohort entries leaked into train-trial AS-norm
stats); with the fix they hurt (5.15–5.31, k=50..1000) and are not used. Embedding-norm quality
features also hurt (5.10–6.50).

### Error analysis (E00 vs F01, cosine) by min duration of the pair

| min dur | trials | E00 | F01 |
|---|---|---|---|
| < 2 s | 8,994 | 10.65 | 9.90 |
| 2–4 s | 21,565 | 5.42 | 5.05 |
| 4–8 s | 14,220 | 3.08 | 3.39 |
| ≥ 8 s | 5,214 | 1.30 | 1.47 |

Short utterances dominate the error; F01 (2 s crops) helps short pairs and hurts long ones.

## Fine-tuning

All: sub-centre AAM (K=3, m=0.2, s=30, head initialised from train speaker centroids),
on-GPU augmentation (babble/coloured noise/synthetic RIR/band-limit/EQ/mu-law),
2 s crops, bs 32, bf16, AdamW lr 5e-5 (backbone) / 2e-3 (head), cosine, 3000 steps,
stem..stage2 frozen (GPU is shared, full fine-tuning of ReDimNet-M is ~10x slower).

| ID | setup | dev2000 (start → best) | public |
|---|---|---|---|
| F01 | M, AAM + aug | 6.70 → 5.56 @2500 | cos 5.51 · center 5.70 · center+QMF 5.36 · QMF 5.09 |
| F02 | F01 + relational KD from B6 (w=1.0) | 6.70 → 5.76 @500, 6.53 @1000 — stopped (no gain, 1.6x slower) | – |
| F03 | F01 + **frozen BN** + random crop 1.5–3 s + lr 2e-5 | 6.70 → 5.56 @3000 | cos 5.31 (minDCF .28) · **QMF 4.88** ← best |
| F04 | F03 + train from stage1 (more trainable layers), 4000 steps | paused (SIGSTOP) at step ~0–100 | |

## Statistical check (2026-10-11) — speaker-bootstrap (200×, grouped by first-file speaker) on public

| system | EER | 95% CI | H1 / H2 (speaker-disjoint halves) |
|---|---|---|---|
| E00 cos | 5.87 | [4.03, 7.10] | 4.02 / 6.54 |
| E00 qmf | 5.15 | [3.57, 6.26] | 3.57 / 5.83 |
| F01 cos | 5.51 | [3.88, 6.89] | 3.88 / 6.09 |
| F01 qmf | 5.10 | [3.51, 6.32] | 3.47 / 5.63 |
| F03 cos | 5.31 | [3.65, 6.58] | 3.61 / 5.90 |
| F03 qmf | 4.88 | [3.43, 6.06] | 3.32 / 5.47 |

Paired differences: QMF vs cosine −0.72 (E00) / −0.44 (F03), CI excludes 0 → robust.
F03 vs E00 (both QMF) −0.28 [−0.53, +0.05], P=0.94; F03 vs F01 (QMF) −0.22 [−0.45, +0.06], P=0.93 →
fine-tuning gain is real-looking but small and not conclusive. Speaker heterogeneity is large
(halves 3.3 vs 5.5), so public→private EER can shift by >1 point regardless of system.

Selection-bias caveats: dev2000 (checkpoint selection) is a subset of public; back-end choice and
the F01→F03 recipe change were made after seeing public results.

## Protocol v2 (2026-10-12) — anti-overfitting, pre-registered

`python -m sred.protocol [--extract] <systems...>`: public speakers split by md5 hash into
**P-dev** (228 spk, 21,085 trials) and **P-hold** (236 spk, 16,394 trials; cross-half trials dropped).
All decisions on P-dev only; P-hold is revealed once at the end. Accept B over A only if the paired
speaker-bootstrap P(B better) > 0.95 on P-dev. Back-end frozen = QMF(dur) fitted on train
(no more back-end search). Training no longer selects checkpoints on public: fixed schedule,
step checkpoints, final system = weight average of the last K checkpoints (`avg.pt`);
monitoring subset (if any) is drawn from P-dev only.

P-dev baseline (cos / qmf): E00 5.78 / 5.02 · F01 5.66 / 5.19 · F03 5.08 / 4.60
(F03 vs E00 qmf −0.42 [−0.89, +0.11], P=0.94 — not yet accepted under the rule).

Pre-registered runs (system = `avg.pt`, judged vs F03 qmf on P-dev):
- **F05**: F03 recipe, 9,000 steps (3x), no monitoring, save every 1,500, avg last 3.
- **T00**: B6 teacher zero-shot on full public → P-dev headroom. KD run **K01** only if
  B6 qmf ≤ F03 qmf − 0.5 on P-dev: F05 recipe + RKD from B6 (w=10, teacher on clean audio).

Results:
- T00 (B6, 12.46M, teacher only) P-dev cos 4.85 / **qmf 4.07** (vs F03 −0.53 [−1.15, +0.44], P=0.88).
  Gains on 2–8 s pairs (3.84 / 1.86 vs 4.44 / 2.47), none on <2 s (11.1 vs 11.3).
  Threshold met (barely) → K01 queued after F05 (`scripts/k01.sh`).
- (Container restart killed the first F05 at step ~4000; rerun with `--resume` support via
  `scripts/pipeline_v2.sh`.)
- **F05 rejected**: P-dev cos 5.47 / qmf 5.01 (vs F03 qmf +0.42 [−0.19, +0.77], P(better)=0.12;
  vs E00 ≈0). 3x longer schedule drifts away from the pretrained solution; worst on long pairs
  (≥8 s: 1.00 vs 0.57). → longer fine-tuning is not the way; F03's short schedule stays.
- Pre-registered next (no training): **W01** = WiSE-FT weight interpolation 0.5·F03 + 0.5·M (zero-shot),
  single α, judged vs F03 qmf on P-dev.
- **W01**: P-dev cos 5.08 / qmf 4.55 (bins 11.16 / 4.67 / 2.07 / 0.64). vs F03 qmf −0.05 [−0.27, +0.22]
  → tie, not accepted over F03. vs E00 qmf −0.47 [−0.76, −0.03], **P=0.98** — the only system so far
  that passes the rule against zero-shot (F03: P=0.94). Same P-dev EER as F03 with less drift from the
  pretrained weights (better on 4–8 s pairs, slightly worse on 2–4 s) → preferred tie-breaker for the final.
- **K01** (F05 + RKD from B6, w=10): P-dev cos 5.52 / qmf 4.51 (mindcf 0.36). vs F03 qmf −0.09 (P=0.73),
  vs W01 qmf −0.04 (P=0.69), vs F05 qmf −0.51 (P=0.94) but cos +0.05. KD partly counters the long-schedule
  drift, but the gain shows only after QMF and minDCF is worse → not accepted.
- **Decision (made before P-hold): final = W01 + QMF(dur).** P-hold revealed once below.

P-hold (revealed once, 2026-10-12; `exp/logs/protocol_reveal.log`), cos / qmf EER, qmf minDCF:

| system | cos | qmf | minDCF | qmf vs E00 |
|---|---|---|---|---|
| E00 zero-shot M | 5.81 | 5.32 | 0.210 | – |
| F03 | 5.47 | 5.17 | 0.156 | −0.15 (P=0.80) |
| **W01** (final) | 5.42 | 5.15 | 0.161 | −0.17 (P=0.94); cos −0.39 (P=0.99) |
| K01 | 5.55 | 5.05 | 0.166 | −0.27 (P=0.93) |

Gains replicate in sign on P-hold but shrink (−0.47 → −0.17 for W01): the P-dev numbers were optimistic.
Honest expectation for private: ~0.2–0.5 EER better than zero-shot M + QMF, minDCF clearly better.
Submission: `submissions/W01_qmf_public.txt` (4.81M params). Private command:
`scripts/make_submission.sh <trials> <audio_dir> private <out> exp/ft/W01_wise_F03_M/avg.pt qmf_dur`.

---

# Milestone M1 (2026-10-14 → 10-16, branch `thucth`) — data cleaning, failure analysis, distillation

New machine (L40S 46 GB, dedicated): data in `SRED/data/Vi-SRED26` (unzipped from
`gs://cake-ds-intern/VLSP-2026/SRED/Dataset/`, zip MD5/CRC32C verified against the bucket, zips deleted),
paths in `env.sh` are now relative to the repo. Pretrained weights re-downloaded from the **official**
GitHub releases (IDRnD/ReDimNet `latest`, PalabraAI/redimnet2 `v1.0.0`) + HF `Wespeaker/...-resnet34-LM`;
they reproduce the earlier HF/MLX/GGUF copies (M cos 5.873 vs 5.87, B6 5.691 vs 5.693).

Final system of this milestone: **K21_M** (`exp/ft/W_K21_M/avg.pt`, 4.81M) — public **EER 4.50 % full /
2.92 % cleaned** (from 4.85 / 3.28). Submitted 2026-10-16 as `result-1016.zip` (format: `results.txt`,
`file1 file2 score`). Private test: `scripts/make_final_submission.sh` (see end of this section).

## Evaluation sets used from here on

All are subsets of the organisers' `public-test.txt` (49,993 trials); labels are never modified.

| set | trials | built by |
|---|---|---|
| full public | 49,993 | as given (≈ leaderboard) |
| **cleaned public** | 47,932 | `lists/clean/public_trials_clean.txt`: drops 2,061 trials touching a flagged file (no speech, wrong-speaker file, minority voice of a mixed folder) or a non-target pair that is the same person |
| **P-dev** / **P-hold** | 13,489 / 22,776 | cleaned trials, md5 speaker halves on **merged identities** (`lists/clean/public_identity.json`); cross-half trials (11,667) dropped |

(The Protocol-v2 split above hashed raw folder IDs on uncleaned trials; numbers are not comparable.)
Acceptance rule unchanged: paired speaker bootstrap (300×) on P-dev, accept if P(better) > 0.95.
Comparison helper: `scripts/compare.py REF CKPT...` (P-dev, P-hold, cleaned, full, <1 s speech, minDCF).

## Data analysis (`scripts/data_features.py`, `scripts/analyze_data.py` → `exp/analysis/report.txt`)

- Train 58,404 files / 1,900 speakers / 146 h; public 17,850 files / 902 folders. All 16 kHz mono PCM16.
- Public is slightly harder: shorter (median 5.4 s vs 7.4 s), noisier (12 % < 15 dB SNR vs 7 %), more
  band-limited (31 % full-band vs 51 %). Gender (F0 proxy) ≈ 1 M : 2 F in both.
- **One speaker dominates public**: `id03289` = 29.5 % of target trials (14.6 % of trial slots).
- **Folder IDs are not identities**: `id03289`/`id03197` are one person (all 365 cross-folder target
  trials); ~19 further public folder pairs and ~123 train folder pairs look like duplicates; several
  folders contain 2–3 voices (TV-show host + guests). 3 train files are byte-identical to public files.
- Speaker ID namespaces of train and public are independent (`id00246` exists in both, different people).
  ~13–20 public speakers also occur in train under other IDs (~2.4 % of trials) — negligible leakage.

## Zero-shot, approved checkpoints (full public)

| model | params | cos | center | QMF | best back-end |
|---|---|---|---|---|---|
| ReDimNet-M mix | 4.81M | 5.87 | 5.50 | 5.15 | 5.00 (center+QMF) |
| ReDimNet-S mix | 3.15M | 6.53 | 5.92 | 5.70 | 5.39 |
| WeSpeaker R34-LM | 6.63M | 9.46 | 7.42 | 7.48 | 6.06 |
| ReDimNet2-B4 vox2 | 6.66M | 9.71 | 7.18 | 7.60 | 6.23 |
| ReDimNet2-B6 (teacher) | 12.46M | 5.69 | 5.47 | 4.79 | 4.79 |

## G-series: same recipe (F03: 3k steps, early layers frozen, aug, WiSE 0.5 of step 2k) on every model

Full-public QMF EER, zero-shot → fine-tuned → WiSE: M 5.15 → 5.01 → **4.88** (= W01 reproduced) ·
S 5.70 → **4.93** → 5.06 · B4 7.60 → **5.45** → 5.88 · R34 7.48 → 6.12 → **5.87** · B6 4.79 → 4.42 → **4.19**.
WiSE helps M/R34/B6, hurts S/B4. VoxCeleb-only models gain most but stay ~0.5 behind ReDimNet v1.

## Failure analysis (`scripts/error_analysis.py` → `exp/analysis/error_report.txt`)

Errors of M+WiSE at its EER threshold, by trial attribute:
- **Short speech** (energy-VAD speech in the shorter file): < 0.75 s → FRR 21 % (6.6 % of targets, 29 % of FRs).
- **Same speaker, pitch change** > 5 semitones → FRR 15–16 % (13 % of targets, 43 % of FRs); similar-F0
  non-targets FA 7 % vs 3.8 %. Models lean on pitch.
- **Male–male** non-targets FA 10.6 % vs 4.9 % female–female.
- Level mismatch > 10 dB: FRR 8.6 % (B6 4.8 %); SNR mismatch > 20 dB: FRR 15 %. Very clean pairs FA 8.4 %.
  Bandwidth and clipping: minor.
- **Errors concentrate on a few folders**: top-5 target speakers hold 67 % of FRs; 4 folders
  (`id04411`, `id02686`, `id02481`, `id04017`) hold 29 % — embedding 2-clustering + Gemini listening
  (analysis only, `scripts/gemini_listen.py`, `exp/analysis/gemini/`) show mixed-voice folders, clips with
  other voices / no speech, and one person split over several IDs. Removing those targets: 4.88 → 4.08.
- Errors are shared: 47 % of M's errors are made by all 4 submittable models (88 % of those by B6 too).

## Data cleaning (`scripts/clean_data.py` → `lists/clean/`)

Ensemble of approved models only: train = zero-shot M + B6 (not fine-tuned on train, so noisy labels are
not memorised), public = fine-tuned M + B6 WiSE. Steps: no-speech files, byte-identical duplicates (and
train copies of public files), duplicate identities (folder-centroid sim above the 99.99th pct → merged),
outlier files (LOO sim to own folder < 0.2, or < 0.5 and another identity closer by > 0.1), multi-voice
folders (2-cluster centroid sim below the 99th pct of cross-folder sims → drop minority cluster).
Train: 1,394 / 58,404 files dropped, 181 folder pairs merged (1,900 → 1,823 speakers) →
`lists/clean/train_clean.txt`. Public: 509 files flagged, 2,061 trials removed (labels untouched).

External SV models were downloaded and evaluated for cleaning (WeSpeaker R293/R221-LM, SimAM-R34-VB2,
3D-Speaker ERes2NetV2/CAM++ zh-200k; registry `EXTERNAL` in `sred/models.py`) but are **not used**:
individually 6.3–8.4 full-public EER (worse than zero-shot M) and they make the ensemble worse
(B6w 4.67 → +E2V2+R221 5.29 centered-cosine). Nothing external is used in training or submission.

Retraining on the cleaned list (same recipe, C-series, cleaned-public EER, P(better) vs original list):
M ft 3.44 → 3.28 (P 0.75), M WiSE 3.28 → 3.30, S ft 3.56 → 3.44 (P 0.69). Small, consistent, not
significant; cleaned fine-tuning ≈ WiSE (both limit learning from noisy labels). Kept as default.
Label noise mostly hurts *evaluation*: ≈ 1.5 EER of full-public error is mislabelled trials (floor ≈ 1.4).

## Negative results (do not repeat)

| idea | result (vs baseline) |
|---|---|
| label-preserving voice aug (speed 0.9–1.1, pitch ±2 st, tempo 0.85–1.15), `sred/augment.py` | +0.3 cleaned EER (P better ≈ 0.1); no gain on pitch-gap trials |
| channel aug (gain, time/freq masking, clipping) | ±0.0 |
| short crops 0.75–3 s (without KD) | worse on < 1 s trials (6.88 → 7.65) |
| QMF on speech duration / + SNR / + AS-norm / gender-matched AS-norm (`scripts/phase1_scoring.py`) | +0.06 … +0.45 on P-dev |
| energy-VAD trimming before embedding (`--mode vad`) | EER ±0.0; minDCF 0.142 → 0.130 on P-dev (P 0.95) — optional |
| KD into B4 (K30) | 3.15 cleaned (worse than M) |
| 36k steps instead of 18k (K23) | ±0.0 |
| WiSE α 0.4–0.7, soup of last 4 ckpts, 2-seed soup (K21+K24), 2-teacher targets (K25) | all within ±0.03 |
| one-shot pruning of B6 to 6.3M + 18k recovery (P01) | 3.93 cleaned |
| progressive pruning, stage 1 to 8.49M + 9k recovery (P02, still > 7M) | 2.90 cleaned but P-hold 3.24 vs 2.86 → stopped before stage 2 |

## What worked: long→short distillation from fine-tuned B6

`sred/train.py --teacher-emb <cached full-utterance teacher embeddings> --temb-kd --temb-rkd`:
the student sees 0.75–3 s augmented crops; targets are the teacher's embeddings of the **whole clean
utterance** (cosine via a learned projection + relational KD on batch similarity matrices) + AAM on the
cleaned list. Teacher embeddings cached once (`exp/emb/W_G05_B6_avg/trainfull_*.npy`).

| run | setup (all M, cleaned list) | P-dev | P-hold | cleaned | full | < 1 s | P vs C01 |
|---|---|---|---|---|---|---|---|
| C01_M | baseline (F03 recipe, cleaned) | 3.23 | 3.24 | 3.28 | 4.85 | 6.88 | – |
| K10 / K12 | KD, 3k steps, early layers frozen | 3.14 / 3.02 | 3.28 / 3.35 | 3.26 / 3.27 | 4.84 / 4.83 | – | 0.82 / 0.75 |
| K11 WiSE | KD, all layers, 9k steps, teacher C01_B6 | 2.89 | 3.17 | 3.13 | 4.73 | 6.51 | 0.99 |
| K20 WiSE | K11 with stronger teacher W_G05_B6 | 2.96 | 3.02 | 3.07 | 4.66 | 6.42 | 0.98 |
| **K21 WiSE** | **K20, KD weights ×2, 18k steps** | **2.95** | **2.86** | **2.92** | **4.50** | **5.82** | **0.98** |
| K22 WiSE | K21 weights, 9k steps | 2.92 | 3.00 | 3.02 | 4.64 | 6.33 | 0.97 |
| teacher W_G05_B6 | 12.46M, not submittable | 2.40 | 2.27 | 2.31 | 4.19 | 4.10 | – |

KD needs capacity (all layers) and length (18k); WiSE 0.5 (step 12k ⊕ pretrained) is still needed against
drift. Run-to-run noise (K21 vs seed-1 K24) ≤ 0.02. Milestone target (≤ 2.5 cleaned) not reached:
3.28 → 2.92 closes ~half of the student–teacher gap; within the rules (approved checkpoints, ≤ 7M) the
remaining gap is architectural. Pruning code: `sred/prune.py` (data-driven channel pruning of
ReDimNet2 with least-squares block repair; explainer video `scripts/make_pruning_video.py`).

## Reproduce / private test

```bash
# K21 (needs exp/emb/W_G05_B6_avg/trainfull_*.npy from: clean_data.py --models exp/ft/W_G05_B6/avg.pt --extract-only)
WISE_STEP=12000 scripts/c_sweep.sh redimnet_M_mix K21_M stem lists/clean/train_clean.txt \
  --teacher-emb exp/emb/W_G05_B6_avg/trainfull_modefull_f5574c4d.npy --min-crop-sec 0.75 \
  --temb-kd 2.0 --temb-rkd 20 --steps 18000 --save-every 2000

# private test -> leaderboard zip (results.txt inside; checks order/format; refuses on missing audio)
scripts/make_final_submission.sh <private-trials.txt> <private audio dir> result-private-MMDD.zip private
```

## Round 2 (2026-10-16): pipeline techniques from the SV literature, on top of K21

Mapped from the face-recognition pipeline (alignment / normalisation / embedding / margin losses).
All judged vs K21 WiSE (P-dev 2.95 / P-hold 2.86 / cleaned 2.92 / full 4.50, minDCF 0.229).

| idea | source | result |
|---|---|---|
| test-time multi-crop (`--mode crop`, `full+crop`) | aggregation | full 5.07 / 4.65 (crop 2 s / 3 s), 4.52 (full+crop) → no gain |
| dominant-speaker extraction (`--mode dom`: VAD, 1 s windows, 2-means on window embeddings) | target-speaker SV (TASE-SVNet, Guided Speaker Embedding) | full **5.42**. 1 s-window clustering has no power: between-cluster cosine of known multi-voice files (0.64) ≈ clean files (0.69); it cut 12 % of ordinary clips. Needs a diarizer (none approved). |
| large-margin fine-tuning L01 (6 s crops, m 0.5, lr 1e-5, KD anchor, 6k steps bs 16) | IDLab LMFT / ReDimNet2 stage 2 | ft 3.10 cleaned (overfits, train acc 0.99); + WiSE 2.88 / 4.54, minDCF 0.223 — tie (P 0.34); worse on < 1 s (5.92) as the literature warns |
| adaptive LMFT L02 (2–6 s crops, margin 0.2→0.5 with crop length, otherwise L01) | SJTU adaptive LMFT | + WiSE **2.87 cleaned, P-dev 2.83, P-hold 2.82, full 4.50, minDCF 0.218** — best on every split but −0.05, P 0.69 → not significant |
| duration-adaptive margin in the KD recipe K26 (m 0.3→0.1 for 3→0.75 s crops) | Adaptive Margin Circle Loss / AdaFace analogue | + WiSE 2.91 / 4.52 — tie (P 0.56) |
| speech-enhancement front-end | — | not tried: literature consistent that separately trained SE hurts SV; no approved model |

Conclusion: with the approved ≤ 7M checkpoints and this data the recipe is converged at ≈ 2.9 cleaned /
4.5 full; label noise bounds full-public EER from below at ≈ 1.4, and < 3.0 would need ≈ 1.1 cleaned
(half the 12.5M teacher's error). L02-WiSE (`exp/ft/W_L02_K21_almft/avg.pt`) is the preferred tie-breaker
if a second submission is allowed (better minDCF, same full-public EER).

## Round 3 (2026-10-17): transductive test-set smoothing; shifted-segment TTA

- **kNN smoothing of test embeddings** (`sred/transductive.py`, `scripts/score_transductive.py`): each test
  embedding += w · similarity-weighted mean of its k nearest other test files (cosine ≥ tau), 2 passes.
  Selected on P-dev: k 20, w 0.5, tau 0.6 → **full 3.78 / cleaned 1.54 / P-dev 1.81 / P-hold 1.36**
  (K21 alone 4.50 / 2.92 / 2.95 / 2.86). Conservative tau 0.7, w 1.0: 3.96, minDCF 0.148.
  Uses unlabelled test audio only (rules to be confirmed). Gain depends on utterances-per-speaker: on a
  ≤3-files-per-speaker stress set every setting *hurts* (+0.2…+0.8). Diagnostic printed by the script:
  share of files with ≥3 neighbours above tau — public 94 %, stress sets ≤3 %; do not smooth below ~50 %.
  Test-domain centering (4.84), test whitening (14.8) and test-cohort AS-norm (5.7–7.3) all hurt.
- **Shifted-segment TTA** (`--mode shift`: full embedding ×2 + 5 windows of 60 % length): 4.56 alone,
  3.89 with smoothing (full+shift average 3.79) → no gain over full-utterance embeddings; TTA is closed
  (fixed-crop TTA also negative, Round 2).

## Round 4 (2026-10-17, branch `thucth-edge-research`): edge-SV research + two-speaker hypothesis

Techniques from `docs/edge_sv_research_1017.md`, each on top of K21 (same recipe/schedule), + WiSE 0.5.
Full table: `exp/analysis/ablation.md` §3. P(better) vs K21 on P-dev in brackets.

| change | full | cleaned | P-dev | P-hold | verdict |
|---|---|---|---|---|---|
| plain AAM, no sub-centres (E2a) | 4.50 | 2.88 | 2.82 | 2.89 | tie (0.68) |
| **SphereFace2-C** (E2b, m 0.2, s 32) | **4.48** | **2.76** | 2.89 | **2.67** | tie on P-dev (0.45); best cleaned / P-hold / minDCF (0.207 cleaned) of all M students |
| decoupled logit KD DKD-γ (E1a, teacher C01_B6 head, w 0.3) | 4.53 | 2.99 | 2.92 | 3.05 | no gain on top of embedding KD (0.39) |
| confusion-set logit KD, top-50 curriculum (E1b, TRKD-like) | 4.54 | 3.01 | 2.95 | 3.03 | no gain (0.38) |
| QMF + embedding magnitude (E3) | 4.68 | 3.11 | 3.12 | 3.14 | worse — magnitude ≈ speech duration (r = −0.43) |
| A: keep louder voice at test time (`--mode loud`; cut only if 2 voices, cos < 0.45 and ≥ 6 dB louder) | 4.54 | 2.86 | 2.83 | 2.89 | tie (0.56); cuts ~1 % of clips, 0/36 clips of the mixed-voice folders (those are single-voice clips of another person) |
| B: overlapping quieter speaker aug (`--aug-overlap 0.3`, 5–15 dB, 30–100 % of crop) | 4.54 | 2.88 | 2.86 | 2.84 | tie (0.46) |
| A + B | 4.57 | 2.86 | 2.93 | 2.84 | tie |
| INT8 weights (per-channel PTQ) | 4.49 | — | — | — | free; irrelevant to the parameter-count rule, not pursued |

Conclusion: within-clip two-speaker audio is rare in this data and already tolerated; the speaker-level
label noise is in the folders, not the clips. The M student is converged at ≈ 2.9 cleaned / 4.5 full.
SphereFace2 is the only candidate worth a second seed if more model-side work is wanted.

## Private test inference (2026-10-18, branch `thucth-private-test`)

Data: `gs://cake-ds-intern/VLSP-2026/SRED/Dataset/Vi-SRED26-PrivateTest.zip` (MD5/CRC32C verified, extracted to
`data/Vi-SRED26/private_test/`, zip deleted): 278,410 unlabelled trials, 37,354 files, 1,868 folders.
Model K21 (`exp/ft/W_K21_M/avg.pt`), cosine + QMF(dur). Run: `exp/logs/run_private.sh`.

| zip (`submissions/final_private/`) | mode | notes |
|---|---|---|
| `result-private-nontrans.zip` | `TRANSDUCTIVE=0` | public equivalent: 4.50 EER |
| `result-private-trans.zip` | `TRANSDUCTIVE=1` (k 20, w 0.5, tau 0.6, 2 passes, min_cluster 3) | public equivalent: 3.79 EER. Private redundancy 91.2 % of files with ≥ 3 neighbours (public 94.2 %; stress sets ≤ 3 %) → expected to help; 95.0 % of files smoothed. Use only if the rules allow unlabelled test audio. |

Both: 278,410 lines in trial order, `file1 file2 score`, format checks passed.
