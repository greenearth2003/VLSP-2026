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

## Round v3 (2026-10-14) — EDA-driven experiments, pre-registered

Source: `docs/version_141026_eda_report.md` §8. No test information enters any system: cleaning uses train +
pretrained embeddings only; VAD / F0 are computed from audio; public labels/IDs only score P-dev (P-hold is spent).

Common training recipe (= reproducible W01 recipe): F03 hyper-parameters, 3000 steps, seed 0, no test monitoring,
checkpoint **step 2000**, then WiSE 0.5 with pretrained M → `exp/ft/W_<run>/avg.pt`; back-end QMF(dur).

| ID | change | judged against |
|---|---|---|
| R0 | none (re-run of the W01 recipe, measures re-training noise) | W01 (sanity) |
| E1 | energy VAD at inference (drop silences > 0.4 s; durations for QMF = speech duration) on W01 | W01 |
| E5 | QMF + \|log F0 ratio\| features (F0 from audio) on W01 | W01 |
| E2 | cleaned train list `lists/train_clean.txt` (`eda/make_clean_list.py`) | R0 |
| E3a | `--aug 0` (SpecAugment only) | R0 |
| E3b | `--aug-profile matched` (no band-limit/μ-law, reverb p0.15 RT60≤0.5, + pause insertion p0.3) | R0 |
| E4 | `--crop-sec 2.5 --min-crop-sec 1.0 --spk-balanced` | R0 |

Rule: accept if paired speaker-bootstrap P(better) > 0.95 on P-dev (QMF back-end). Final system = W01-recipe +
all accepted training changes (one combined run C1) + accepted inference changes; if nothing is accepted, W01 stays.

Results v3 (P-dev, QMF; vs W01):
- **E1 VAD** −0.06 [−0.25, +0.08], P=0.81 (cosine −0.12, P=0.86) → not accepted (small, consistent-sign gain).
- **E5 F0-QMF** +0.01 [−0.04, +0.03], P=0.56 → rejected (embedding already encodes gender).
- E1+E5 −0.05, P=0.83 → not accepted.
- (v3) R0 re-run of the W01 recipe: P-dev qmf 4.70 vs W01 4.55 (+0.15 [−0.07, +0.24]) = re-training noise +
  W01's original public-based checkpoint pick. Training changes are judged against R0.

- **v3 training results** (P-dev qmf, vs R0 4.70 / minDCF 0.306): E2 clean 4.67 (−0.03, P=0.71, minDCF 0.326);
  E3a no-aug 4.58 (−0.11, P=0.69, minDCF **0.274**); E3b matched-aug 4.63 (−0.07, P=0.68, minDCF 0.277);
  E4 short+spk-balanced 4.60 (−0.10, P=0.72, minDCF 0.280). **None accepted.** All point the same way (small
  gains, better minDCF when the heavy synthetic augmentation is removed/softened), none above re-training noise.

## Round v4 (2026-10-15) — short-duration training, judged on a NEW train-held-out validation set

Motivation: an external plan (short-duration focus); P-dev is over-used, so decisions move to **V** =
300 train speakers held out (`sred/valset.py`, `lists/val_v4.json`): 4,374 files, 16,306 trials in SS / SL / LL
(S = first 1.5 s), 60 % same-gender non-targets, distant-index targets first. Runs use `--exclude-val`
(46,772 utts left). Public P-dev is only a final sign-check. No W01 teacher (it saw V speakers): consistency
uses the student's own long view (stop-grad).

QMF 2.0 checked first on P-dev with W01 (inference-only): Q01 +speech-duration +0.05 (P=0.28, minDCF worse);
Q02 +hard negatives −0.04 (P=0.80, minDCF 0.315 vs 0.288) → both rejected.

| ID | recipe (all: F03 hparams, 3000 steps, step-2000 + WiSE 0.5, --exclude-val) | vs |
|---|---|---|
| Z00 | zero-shot ReDimNet-M (anchor) | – |
| S00 | baseline recipe | – |
| S01 | `--crop-sec 6 --crop-dist balanced` (0.8–1.5 30 %, 1.5–2 25 %, 2–3 25 %, 3–6 20 %) | S00 |
| S02 | `--crop-sec 6 --consistency 0.1 --cons-mu 0.5` (short 0.8–2 s + long 4–6 s views) | S00 |
| S03 | S01 + `--pk 4 --supcon 0.1` (8 spk × 4 utt) | S01 |
| I01 | S00 with full+multi-crop embeddings (`::mc`) | S00 |
| W02 | WiSE α ∈ {0.25, 0.4, 0.6, 0.75} for the best of S00–S03 | α = 0.5 |

Rule: P(better) > 0.95 on V (QMF). Final = winning recipe + α **retrained on the full train**, sign-checked on
P-dev against W01; if nothing wins, W01 stays.

### Final selection rule (pre-registered 2026-10-15, before any v4 result)
`scripts/finalize_v4.py`: recipe = best of S01–S03 beating S00 on V (QMF, P > 0.95); data = S04
(`lists/train_drop_bad.txt`: −56 mixed-voice IDs, −371 suspicious, −51 cross-ID duplicates) if it beats S00
(P > 0.95). Winner retrained on the FULL train (`scripts/pipeline_final.sh`), step 2000 + WiSE 0.5, released only
if not worse than W01 on P-dev; otherwise W01 stays. Error analysis: W01 public EER without mixed-voice target
trials = 3.73 % (vs 4.85 %); FN are the severe errors (29 % with score < −3, 61 % of those from mixed-voice IDs),
FP are mild and 86 % same-gender.
- **SOUP6** (2026-10-15, inference-only, decided before evaluation): uniform weight average of all six full-train
  fine-tuned models (W01, R0, E2, E3a, E3b, E4; no member selection). Released if not worse than W01 on P-dev.

### Clean evaluation (2026-10-15, user request)
From now on P-dev and V exclude trials touching label-noise IDs found in the EDA (public: 42 mixed-voice IDs +
same-voice ID pairs; V: 15 mixed-voice train IDs). Clean P-dev = 19,569 trials / 217 spk; clean V = 15,105 trials.
Clean P-dev (qmf EER / minDCF): zero-shot M 4.54/.347 · W01 **4.09**/.274 · F03 4.05/.268 · R0 4.21/.292 ·
E2 4.23/.315 · E3a 4.15/**.256** · E3b 4.13/.260 · E4 4.08/.263. W01 vs zero-shot −0.46 (P=0.99); all v3 runs
within ±0.15 of W01 (not significant).
- **v4 results (clean V, QMF EER / minDCF; SS/SL/LL)**: Z00 zero-shot 5.35/.232 (7.29/5.59/3.24) · S00 4.63/.196
  (6.47/4.52/2.85) · S01 balanced 4.59/.192 (6.23/4.53/2.86) · **S02 consistency 4.51/.186** (6.29/4.55/2.84) ·
  S03 PK+SupCon 4.78/.197 (worse, P(better)=0.007) · S04 drop-bad 4.54/.192 · S00+multi-crop 4.60.
  vs S00: S01 −0.03 (P=0.42), S02 −0.11 [−0.19, +0.06] (P=0.85), S04 −0.09 (P=0.62), S03 +0.15 (significantly
  worse). Fine-tuning vs zero-shot −0.72 (P=1.0) → V is sensitive. **Decision: NONE → W01 stays** (finalize_v4.py).

## External data (2026-10-15, user-approved): Vietnam-Celeb (HF doof-ferb/Vietnam-Celeb, ~1,000 spk, 187 h)
Leak audit on a 20-speaker sample: no VC file within cos 0.90 of any VLSP file, no identical PCM; VC→VLSP
nearest-speaker centroid cos 0.41–0.71 (≈ VLSP nearest-other-speaker p50 0.57) → independent source.
`ext/vc_prepare.py` (full audit, 10 utt/spk): drop VC speakers with centroid cos ≥ 0.80 to any VLSP train/public
speaker, drop mixed-voice VC speakers → `lists/train_vlsp_vc.txt` (= train_drop_bad + VC).
**D1** = S02 recipe on that list, 3000 steps, step 2000 + WiSE 0.5; judged on clean P-dev vs W01
(V cannot be used: D1 trains on V speakers).
- **D1vc** (S02 recipe + VC, 3000 steps, step 2000 + WiSE): public 4.94, clean P-dev 4.16 (vs W01 +0.07, P=0.24),
  minDCF 0.339 → not better. Full-train releases: S02full public 4.86 / clean 4.11; S04full 4.82 / 4.12;
  S04 (no-V) 4.82 / 4.09 — all within noise of W01 (4.85 / 4.09).
- **D2a** = S04 recipe + VC, 6000 steps, step 6000 + WiSE 0.5; **D2b** = D2a + duration-balanced crops.
- **D2a** (S04 recipe + VC, 6000 steps): public 4.94, clean P-dev 4.24 (+0.15 vs W01), minDCF .329 → worse.
  Vietnam-Celeb hurts twice (D1, D2a): domain mismatch (celebrity YouTube vs VLSP), not under-training.
- **QS calibration** (QMF fitted on train cohort full + 1.5 s crops, same-file pairs excluded; `::qs`):
  clean P-dev W01 4.09 → 4.04, S04 4.09 → 4.05 (both −0.04/−0.05, P≈0.4–0.6, minDCF also better); <2 s bucket
  unchanged — gain comes from better cross-duration score alignment. **S04 + QS: public 4.78** (best so far).
  Release `releases/version_161026_S04qs/`; private: `scripts/make_submission_qs.py <trials> <audio> private <out>`.
- **D2b** (D2a + duration-balanced crops): public 4.91, clean P-dev 4.31 (+0.22 vs W01; vs D2a +0.07), minDCF .329;
  <2 s bucket 10.74 (no gain). Not better. Final submission stays **S04 + QS** (public 4.78).

**RULE (user, 2026-10-16): only the organisers' data may be used.** Vietnam-Celeb runs (D1vc, D2a, D2b) are
INVALID for submission (they were also worse). Final submission S04+QS uses BTC data only.

## Round C (2026-10-16) — short-utterance proposals, BTC data only, judged on clean V vs S04
Diagnosis first: on clean P-dev the "<2 s = 10.66 %" bucket is dominated by ONE speaker (id03289, ~72 % of P-dev
targets): without it <2 s EER = 5.01 %, overall 2.92 %. Public is therefore a poor guide for short-utterance work;
decisions use clean V (SS/SL/LL buckets). Baseline C00 = S04 (W_S04_dropbad, trained without V).
| ID | change vs S04 recipe | notes |
|---|---|---|
| C01 | DAME-FT hard: views 1/2/6 s → prefixes 48/96/192, AAM margins 0.0/0.2/0.4, `--crop-sec 6 --dame` | views are crops of the same recording (paper uses different utterances) |
| C06 | `lists/train_keepdom.txt`: mixed-voice IDs keep the dominant cluster (≥60 %) instead of dropping | 56,731 utts / 1,885 spk |
| C03 | `--pk 4 --proto 0.1`: query (1–2 s crop) vs prototype of 3 supports + AAM | S03 (PK+SupCon) was significantly worse → low prior |
Rule: P(better) > 0.95 on clean V (QMF). Skipped with reasons: uncertainty/quality QMF features (Q01/norm features
already hurt), phonetic scoring (needs ASR model; parameter/rule risk).
