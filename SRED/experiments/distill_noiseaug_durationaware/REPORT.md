---
title: "Distilling a Fine-Tuned ReDimNet2-B6 into a Sub-7M Low-Rank Model for Noisy, Short-Utterance Speaker Verification"
challenge: VLSP 2026 Vi-SRED26 (speaker verification; metric EER; < 7M inference parameters)
date: 2026-10-25
submission: result-private-E8s5000_wise0.7
params: 6947208
dev_eer_all_cleaned: [5.109, 3.764]
dev_minDCF: 0.375
code: SRED/experiments/distill_noiseaug_durationaware/ (git branch thucth-distill-noiseaug-durationaware)
---

# Distilling a Fine-Tuned ReDimNet2-B6 into a Sub-7M Low-Rank Model for Noisy, Short-Utterance Speaker Verification

## Abstract

The VLSP 2026 Vi-SRED26 task limits the speaker verification model to 7M parameters and tests it on noisy Vietnamese
recordings, many with under two seconds of speech. Rather than fine-tuning a pretrained model that fits the budget, we
fine-tune a larger one (ReDimNet2-B6, 12.46M) on noisy short/long crop pairs, compress it to 6.95M parameters with an
activation-whitened low-rank factorisation, and recover accuracy with knowledge distillation (KD) from the uncompressed
model. Weight averaging and a quality-aware calibrator fitted on training speakers complete the system. On a noisy copy
of the public test set it reaches 5.11 % EER, against 7.56 % for the same-size ReDimNet2-B4 fine-tuned directly.

## 1. Introduction

Three difficulties meet in this task: background noise and channel effects, very short speech (53 % of development
trials have under 2 s of speech on the shorter side), and a 7M-parameter limit. The largest pretrained ReDimNet2 model
within the limit, B4 (6.66M), fine-tunes on our noisy data to 7.56 % EER; the larger B6 (12.46M) fine-tunes to 5.35 %
but cannot be submitted. We therefore train the best model we can without the limit and then fit it inside the limit.
Our contributions are (i) one fine-tuning recipe on noisy short/long crop pairs for both teacher and student;
(ii) evidence that whitening the SVD by layer-input statistics makes a 44 % parameter cut workable (6.43 % EER before
retraining, against 15.78 % with plain SVD); and (iii) a back-end fitted on training speakers only that adds 0.7 EER.

## 2. Method

**2.1 Noisy short/long crop pairs.** Each training step samples 32 speakers with two crops each: a short crop
(0.5–2 s) from the most speech-dense region and a long random crop (2.5–6 s). Crops come from a fixed bank of 859,684
crops, so teacher embeddings are computed once and cached. Each crop is corrupted on the fly: reverberation from
RIRS_NOISES [4] (p 0.3); additive noise (p 0.6, 0–20 dB SNR) from MUSAN [5] noise, music and babble, RIRS_NOISES
point-source and isotropic noise, and background cut from training recordings; the same background on both crops of a
pair (p 0.5); and band-limiting, equalisation and μ-law coding.

**2.2 Teacher.** ReDimNet2-B6 [1] with a SphereFace2 [2] head on its 192-D embedding, initialised from training-speaker
centroids and trained alone for 300 steps. The whole model is then trained with AdamW (learning rates 1e-5 backbone,
3e-5 embedding layer, 1e-3 head; 500 warm-up steps; cosine decay to 5 %; batch-norm statistics frozen) for 8,000 steps.
Step 6,000 is best and serves as the teacher.

**2.3 Activation-whitened low-rank compression.** Each convolution and linear weight W, reshaped to a matrix, becomes a
product U·V. Plain truncated SVD discards the directions of least *weight* energy, which need not matter least for the
layer's real inputs. Following activation-aware SVD [6], we factorise W·C^½, with C the covariance of the layer's
inputs on training crops, and fold C^-½ back into V, so truncation removes the least *output* energy. Every layer keeps
the same fraction of output energy, set so the model has 6,947,208 parameters. Untrained, the compressed embeddings keep
a mean cosine of 0.815 to the original (0.544 with plain SVD).

**2.4 Recovery with KD.** The compressed teacher is fine-tuned with the recipe of §2.2 plus two losses against the
teacher's embedding of the *clean* crop: 2.0·(1 − cosine) after a learned projection initialised to the identity, and a
relational loss [7], 1.0·SmoothL1 (β = 0.1) between student and teacher in-batch cosine-similarity matrices. The student
thus maps noisy input to the teacher's clean-speech embedding. We train 10,000 steps and select step 8,000, then restart
from it for 5,000 steps with a new head, optimiser, schedule and seed. The restart's last checkpoint is our model (§4.2).

**2.5 Back-end.** First, WiSE-FT weight averaging [3]: θ = 0.7·θ_trained + 0.3·θ_start, where θ_start is the model
training started from. Second, a logistic calibrator in the spirit of quality measure functions [8], whose inputs are
the cosine score; the min, max and product of the two sides' speech duration, SNR and raw embedding norm; and the
adaptive S-normalised score [9] (top-300 cohort). Each embedding is the mean of a whole-clip and a speech-only embedding.
The calibrator is fitted on trials from training speakers only (5 utterances per speaker, clean and noisy copies,
same-file pairs excluded), so no test label is used. It was designed on the teacher and applied unchanged to the student.

## 3. Experimental setup

**Data.** The official training list after removing non-speech and wrong-speaker files: 1,823 speakers, 57,010
utterances at 16 kHz. No external speaker data; MUSAN and RIRS_NOISES are used for noise only.
**Development set.** The 49,993 labelled public-test trials, scored on a fixed noisier copy of the audio: background
cut from training recordings at random SNR, plus synthetic low-frequency rumble on 30 % of files; labels unchanged.
The *cleaned* subset (47,932 trials) drops trials touching a file flagged as no-speech, wrong or mixed speaker.
**Metrics.** EER (%) on all and cleaned trials, minDCF (P_target = 0.01), and EER by the shorter side's speech duration
(< 0.75 s: 2,983 trials; > 3 s: 12,432). Two seeds of one configuration differ by about 0.1 EER, our noise level.
Scores are cosine similarities of 192-D embeddings unless a back-end is stated.

## 4. Results

**4.1 Main results.** Compression with KD loses only 0.46 EER against the 12.46M teacher and beats direct fine-tuning
of a same-size model by 1.75 EER (Table 1). The back-end brings the system to 5.11 %, ahead of our team's best earlier
system (5.41 %).

*Table 1. Development set, EER %.*

| System | Params | All | Cleaned | minDCF | <0.75 s | >3 s |
|:-------------------------------------------------|:-------|:------|:--------|:-------|:-------|:------|
| ReDimNet2-B4, pretrained | 6.66M | 13.96 | 12.75 | 0.652 | 22.3 | 6.14 |
| ReDimNet2-B6, pretrained | 12.46M | 8.80 | 7.28 | 0.402 | 17.5 | 3.85 |
| ReDimNet2-B4, fine-tuned (§2.1–2.2) | 6.66M | 7.56 | 6.24 | 0.485 | 14.8 | 3.96 |
| ReDimNet2-B6 teacher, fine-tuned | 12.46M | 5.35 | 3.90 | 0.352 | 12.6 | 3.11 |
| Compressed + KD, step 8,000 | 6.95M | 5.81 | 4.34 | 0.378 | 13.3 | 3.32 |
| Compressed + KD, restart step 5,000 | 6.95M | 5.81 | 4.33 | 0.383 | 13.3 | 3.32 |
| **Restart + back-end (submitted)** | 6.95M | **5.11** | **3.76** | **0.375** | 12.9 | 2.98 |
| Team's best earlier system + back-end | < 7M | 5.41 | 3.93 | – | – | – |
| Teacher + back-end (over the limit) | 12.46M | 4.65 | 3.08 | 0.300 | 12.4 | 2.72 |

**4.2 Ablations.** *Route to the limit (Table 2).* Whitening is the decisive step: at the same size it cuts the untrained
error of the pretrained B6 from 15.78 % to 9.70 %. Compressing the *fine-tuned* teacher starts recovery at 6.43 %, and
6,000 KD steps reach 5.83 %. Nested-prefix (Matryoshka [10]) heads meant to give short crops their own low-dimensional
embedding hurt the shortest bucket, and re-estimating batch-norm statistics on noisy data degraded the teacher.

*Table 2. Cosine scoring, one change per row.*

| Model | EER all / cleaned | Note |
|:----------------------------------------------------|:---------------|:---------------------------|
| Pretrained B6 → plain SVD to 6.95M, untrained | 15.78 / 14.77 | |
| Pretrained B6 → whitened SVD to 6.95M, untrained | 9.70 / 8.33 | −6.08 vs plain SVD |
| Fine-tuned teacher → whitened SVD, untrained | 6.43 / 5.06 | +1.08 vs teacher |
| + 6,000 steps of recovery with KD | 5.83 / 4.28 | −0.60 |
| Fine-tuned B4 (same budget, no compression) | 7.56 / 6.24 | +1.73 vs compressed |
| B4 + nested-prefix heads (24/48/96/192-D) | 7.43 / 6.21 | rejected: <0.75 s 16.0 vs 14.8 |
| Teacher, batch-norm re-estimated on noisy data | 8.33 / 6.98 | rejected |

*Training length and restart (Table 3).* Cosine EER stops improving after about 4,000 steps, and the restart does not
change it either, but its scores calibrate better: the back-end gains 0.70 EER on the restart against 0.43 on the
original run, and minDCF falls from 0.390 to 0.375. A re-run of the back-end on a second machine gave 5.105 / 3.755.

*Table 3. EER % all / cleaned of the compressed model.*

| Checkpoint | Cosine | With back-end |
|:----------------------------------------|:---------------------------|:---------------------------|
| Recovery step 4,000 / 6,000 | 5.85 / 4.26 · 5.83 / 4.28 | – · 5.35 / 3.93 |
| Recovery step 8,000 / 10,000 | 5.81 / 4.34 · 5.84 / 4.38 | 5.33 / 3.90 · 5.35 / 3.87 |
| Restart from step 8,000, step 5,000 | 5.81 / 4.33 | **5.11 / 3.76** |

*Back-end components (Table 4).* On the teacher, the duration-and-SNR calibrator gains 0.32 EER. Embedding norm, the
speech-only embedding and AS-norm each move EER by less than the noise level but together lower minDCF from 0.372 to
0.344, so we keep them. Weight averaging gives the largest single gain.

*Table 4. Back-end on the teacher, each row adds one change.*

| Back-end | EER all / cleaned | minDCF | Kept |
|:---------------------------------------------------|:------------------------|:----------------|:------|
| Cosine | 5.35 / 3.90 | 0.352 | |
| Calibrator: duration → duration + SNR | 5.07 → 5.03 / 3.54 | 0.372 | base |
| + embedding norm · + speech-only emb. · + AS-norm | 5.01 · 5.05 · 5.05 | .358 · .354 · .344 | all |
| + short crops in fit · + mean centring | 5.07 · 5.24 | .346 · .378 | no |
| Boosted trees · per-duration-bin LR · 6 s chunks | 5.19 · 5.13 · 5.07 | – | no |
| Weight averaging α 0.5 / 0.7 / 0.8 (all) | 5.31 / 5.09 / 5.08 | .317–.321 | 0.7 |
| **Averaging 0.7 + full calibrator** | **4.65 / 3.08** | 0.300 | final |

**4.3 Submission.** `result-private-E8s5000_wise0.7` is the restarted model with the back-end: one "fileA fileB score"
line per test trial (278,410), in trial order, checked for line count, pair order and finite scores. Scores from a
sharded scorer agree with a full re-run on another machine at Spearman 0.999997.

## 5. Conclusion and limitations

Fine-tuning a model that is too large, then compressing it with whitened low-rank factorisation and distillation, beats
fine-tuning a model that fits the budget (5.81 vs 7.56 % EER, cosine), and a back-end fitted only on training speakers
adds 0.7 EER. Several ablation differences are within seed noise, and the margin over our best earlier system is 0.30
EER (0.17 cleaned). Very short speech remains hard (12.9 % EER below 0.75 s, 2.98 % above 3 s). Untried: utterance-
balanced sampling, distilling the teacher into B4, and structured pruning instead of low-rank factorisation.

## References

[1] I. Yakovlev et al., Reshape Dimensions Network for Speaker Recognition (ReDimNet), Interspeech 2024; ReDimNet2
checkpoints from the authors' release. [2] Y. Wen et al., SphereFace2, ICLR 2022. [3] M. Wortsman et al., Robust
Fine-Tuning of Zero-Shot Models (WiSE-FT), CVPR 2022. [4] T. Ko et al., A Study on Data Augmentation of Reverberant
Speech for Robust Speech Recognition, ICASSP 2017. [5] D. Snyder et al., MUSAN: A Music, Speech, and Noise Corpus,
arXiv:1510.08484. [6] Z. Yuan et al., ASVD: Activation-aware Singular Value Decomposition for Compressing LLMs,
arXiv:2312.05821. [7] W. Park et al., Relational Knowledge Distillation, CVPR 2019. [8] J. Thienpondt et al., The
IDLAB VoxSRC-20 Submission: Large Margin Fine-Tuning and Quality-Aware Score Calibration, ICASSP 2021. [9] P. Matějka
et al., Analysis of Score Normalization in Multilingual Speaker Recognition, Interspeech 2017. [10] A. Kusupati et al.,
Matryoshka Representation Learning, NeurIPS 2022.

## Appendix: reproduction

Code: git branch in the front matter; checkpoints and results: private Hugging Face dataset `pieceofcakee/vlsp2026`
(same path); curves: Weights & Biases `vlsp2026-sred`. Run names: B4 fine-tuned `E0`, nested-prefix `E1`, teacher `E4`,
recovery `E5`, restart `E8`.

```bash
cd SRED && source env.sh; E=experiments/distill_noiseaug_durationaware
SEED=0 $E/run.sh e4:E4-b6-teacher-finetune                       # teacher §2.2
.venv/bin/python $E/prep/cache_b6.py --ckpt $E/runs/E4-b6-teacher-finetune/step6000.pt --out e4s6000
.venv/bin/python $E/dnd/lowrank.py $E/runs/E4-b6-teacher-finetune/step6000.pt \
    $E/artifacts/e5_init_lowrank_step6000.pt                     # compression §2.3
SEED=0 $E/run.sh e5:E5-b6-lowrank-kd                             # recovery §2.4
SEED=1 $E/run.sh e8:E8-b6-lowrank-kd-continue                    # restart
$E/recipe.sh runs/E8-b6-lowrank-kd-continue/step5000.pt \
    runs/E5-b6-lowrank-kd/step8000.pt E8s5000_wise0.7            # back-end §2.5 + submission
```
