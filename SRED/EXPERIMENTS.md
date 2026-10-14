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
