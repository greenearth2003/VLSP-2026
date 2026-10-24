# Frozen Training Plan — E0 to E3

# 0. Locked decisions (2026-10-24) — these override any conflicting text below

**Scope split.** This plan covers the *training pipeline*. The *data* (crops, noise, sampling) follows `README.md`:
MUSAN + RIRS_NOISES + shared train backgrounds + channel effects stay ON, applied on the fly to every student crop.
§2's "explicitly excluded" list therefore does not apply to the data side. Teachers always see the CLEAN crop
(cached offline); the student sees the same crop with noise (noisy student / clean teacher).

**Sampling.** Speaker-uniform: speaker → utterance → crop (§6).

**Duration views (replaces "fixed 2 s" in §8 and "1 s + 2 s" in §9).** Every step = 32 speakers × 2 views:

| View | Bins (s), uniform within the group | Placement | DAME row |
|---|---|---|---|
| short | 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 | speech-dense (energy-based active intervals, no VAD model) | C_short = C_1s |
| long | 2.5, 3, 4, 5, 6 | random (keeps pauses, like real inputs) | C_long = C_2s |

One bin per view per step (all 32 crops of a view share a length). The two views of a speaker come from different
utterances when the speaker has ≥ 2 usable ones (1,807 / 1,823 do), otherwise non-overlapping parts of one file.
A long-view crop is taken from an utterance at least that long (no tiling); a speaker with no such utterance is
skipped in steps that draw that long bin (70 speakers at 6 s) and drawn in the others. Rationale: the shorter side of 97 % of private trials has < 7 s of speech and 49 % < 2 s; 7 s adds
1.4 % of trials at the highest cost. Long view runs in 2 sub-batches of 16 (6 s × 32 does not fit in 20 GB).

**E0 control.** E0 uses the same two-view batches with a single 192-D SphereFace2 head (no DAME), so E0 → E1 differs
only in the loss.

**DAME (exact; replaces §9 "warm introduction" stages A/B/C and the descriptive soft-weighting text).**
K = 4 prefixes D = {24, 48, 96, 192}, J = 2 views. For view j and prefix k, L_{j,k} = SF2(z_{j,:d_k}, y; m_k, s).

    C = {short: [1.0,    1.0,   0.25, 0.50],     # c_{j,k}: 1 inside the band b_{j-1} < k <= b_j (b_j = floor(jK/J)),
         long:  [0.0625, 0.125, 1.0,  1.00]}     #          gamma_k = 2^-(K-k+1) outside it (DAME-SW)
    L_short = sum_k C[short][k] * L_{short,k};   L_long = sum_k C[long][k] * L_{long,k}
    alpha(p) = max(0.5, 1 - p),  p = step / total_steps
    L_DAME = (1 - alpha) * L_short + alpha * L_long

Margins m = [0.0, 0.0, 0.1, 0.2] for 24/48/96/192-D. This is our adaptation: DAME-SW with separate heads and the
DAME-GT alpha schedule on a pretrained B4. The paper's best fine-tuning setup was DAME-FT-HW (tied head, fixed
alpha 0.5); it reports SW in fine-tuning can hurt full↔full, so watch that metric in E1.
Note: the C rows sum to 2.75 / 2.19, so L_DAME is ~2.2–2.75× one SF2 term; read the E2/E3 KD lambdas against that.

**Heads.** Separate training-only SF2 head per prefix (B4 has no head for our speakers, so tying buys nothing);
`tied` (prefix k uses the first d_k columns of one 192-D head) stays a config option, to try only if E1 hurts
full↔full.

**Margin warm-up (E0 and E1–E3).** Each head's margin ramps linearly 0 → target over the first 20 % of steps.

**Training recipe as implemented (`train.py`, `configs/e0.json`, `configs/e1.json`; identical for E0–E3 except `loss`
and the KD terms).**

| Item | Setting |
|---|---|
| student | approved B4 `pretrained/redimnet2/b4-vox2-lm.pt` (6,659,744 params), all layers trainable after the head-only phase |
| head init | every SF2 head (each prefix) starts from pretrained-B4 speaker centroids of the clean train utterances, restricted to its prefix and re-normalised (`prep/head_init.py`, `artifacts/b4_centroids.npy`) |
| head-only phase | first 300 steps: student frozen (lr 0, no autograd), only heads + SF2 bias train. Without it the 24-D prefix head started at a per-sample loss of ≈ 300 (192-D: ≈ 5) and its gradients reached the backbone |
| LR | AdamW, weight decay 1e-4 (0 on heads). Backbone 1e-5, embedding (ASTP + bn + linear) 3e-5: 500-step linear warm-up after the head-only phase, then cosine to 5 %. Heads 1e-3: no warm-up, cosine to 5 % over all steps |
| SF2 | scale 32, lambda 0.7, t = 3; margins warm 0 → target over the first 20 % of steps |
| E0 objective | one 192-D head, L = 0.5 L_short + 0.5 L_long |
| BatchNorm | statistics frozen (eval mode), so chunked forwards equal one full-batch forward |
| chunks | each view runs in chunks of ≤ 96 audio-seconds (6 s × 32 does not fit), per-sample losses, exact gradient accumulation |
| steps / checkpoints | 8,000 steps (300 head-only), 32 speakers × 2 views per step; `step<k>.pt` every 1,000 |
| precision / clipping | bf16 autocast; gradient norm clipped at 5 |

**Measured (2026-10-24).** Head-only steps ≈ 0.25 s (1.9 GB); full steps 0.65–1.0 s depending on the bins (11.4 GB
peak), so one 8k-step run ≈ 2–2.5 h. Zero-shot pretrained B4 on silver (plain cosine, EER %): all 13.96, cleaned
12.75, private-like 14.25; natural <0.75 s 22.26, 0.75–1 s 17.48, 1–2 s 11.90, 2–3 s 9.93, >3 s 6.14; full-3s 13.32,
full-2s 14.48, full-1s 18.73, full-0.75s 22.04, full-0.5s 26.94, 2s-2s 15.15, 1s-1s 24.18; minDCF 0.652.

**Weights** (`pretrained/`, verified): B4 `redimnet2/b4-vox2-lm.pt`, B6 `redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt`,
WavLM-Large `wavlm/wavlm_large.pth` (torchaudio `wavlm_large`, strict load).

**Teachers.** B6 = the raw approved `b6-vb2+vox2+cnc2_v0-lm` (not our fine-tuned B6). WavLM-Large is approved as a
training-only teacher. Both are run OFFLINE on a fixed crop bank (2 positions per short bin, 1 per long bin, only
where the utterance is long enough: 859,684 crops, 434 h, `artifacts/crop_bank.npz`); training draws crops from that bank, so teacher targets
are exact. WavLM teacher preparation (layer mix + pooling + projection + SF2 head, WavLM frozen) runs online once.

**Controlled-duration evaluation (§7).** Crops = the most energetic N-second window (energy-based, deterministic),
computed once on clean public audio and applied to the silver audio (`data/public_rumble_v2`, same timelines) for every
model. Silver is the only evaluation set; public_realnoise / public_noisy are no longer used.
Only the test side is cropped (both sides for 2↔2 and 1↔1). A file whose estimated speech is < N is left out of
condition N; exclusions are counted in the report.

**Budget (RTX A4500 20 GB, B4 bf16 ≈ 8.6 ms per training audio-second).**

| Item | Estimate |
|---|---|
| step (32 × (1.17 s + 4.1 s) ≈ 169 audio-s) | estimated ≈ 1.5 s; measured 0.65–1.0 s |
| one run: 8k steps (≈ 9 utterance passes) + eval | ≈ 2–2.5 h measured; resume to 12k if still improving |
| one-time: manifest + crop bank + eval set / B6 cache (180× RT) / WavLM prep + cache (864× RT, measured) | 10 min (done) / ≈ 2.4 h / < 1 h |
| E0–E3 + teachers | ≈ 13–15 GPU-h: E0+E1 (started 2026-10-24 on the reference server) → gate → teachers + E2 → E3 |
| validation | reference server only (`evaluate.py`); training servers return runs through the HF repo (HANDOFF.md) |

---

## 1. Goal

Build the strongest `<7M` speaker-verification model under the challenge constraints by adapting the approved **ReDimNet2-B4 pretrained checkpoint**, then progressively adding:

1. challenge-domain adaptation,
2. duration-aware representation learning,
3. WavLM representation distillation,
4. optional B6 relational distillation.

The student always remains **ReDimNet2-B4 (~6.66M parameters)** and must remain below the 7M inference limit.

The known main weakness is short-duration speech: trials where the shorter side has less than 1 second of speech have approximately 11% EER and account for roughly one third of current errors.

---

# 2. Frozen Scope

## Included

- original audio loading
- sample-rate handling
- log-Mel frontend
- training manifest
- speaker-balanced sampling
- duration sampling
- ReDimNet2-B4 architecture
- pooling and 192-D embedding
- SphereFace2 training
- DAME / Matryoshka supervision
- WavLM teacher preparation and distillation
- ReDimNet2-B6 relational distillation
- optimizer and learning-rate strategy
- duration-aware evaluation

## Explicitly excluded

(§0: the data side follows README.md, so the noise items below are ON as data; this list now only means
"no new augmentation research inside E0–E3".)

Do not modify or experiment with:

- background/noise augmentation
- codec augmentation
- reverberation
- babble
- EQ
- shared-background sampling
- synthetic noise
- external noise datasets
- score calibration/Q3
- TTA
- cohort normalization
- VAD-model experiments

Those belong to separate workstreams.

---

# 3. Student Model

Use the approved checkpoint:

`ReDimNet2-B4 — b4-vox2-lm`

Approximate parameter count:

`6.66M`

## Initialization

Load pretrained weights for:

- feature-processing layers
- ReDimNet2 backbone
- temporal/spectral blocks
- pooling
- embedding projection

Discard the original pretrained speaker-classification head.

Create a new SphereFace2 head for the challenge training speakers.

All layers of ReDimNet2-B4 remain trainable.

Do not freeze the backbone.

However, use a substantially smaller learning rate for pretrained layers than for newly initialized heads.

Example LR groups:

```yaml
backbone_lr: 1.0e-5
embedding_lr: 3.0e-5
new_head_lr: 1.0e-3
```

These are starting values, not hard-coded constants.

All values must be configurable.

---

# 4. Input Frontend

Use the same frontend across E0–E3: the pretrained B4's own built-in frontend (the checkpoint's `spec` module,
`TFMelBanks`), unchanged. It is part of the model and its mel filterbank is stored in the checkpoint, so the student
(training and submission) takes a raw 16 kHz waveform.

```text
original waveform
      ↓
mono
      ↓
16 kHz
      ↓
── inside ReDimNet2-B4 (spec module) ──
signal normalization          (norm_signal: true)
      ↓
pre-emphasis                  (do_preemph: true)
      ↓
72-bin log-Mel                (F = 72, 20–7600 Hz)
25 ms Hamming window (400 samples), 10 ms shift (160), n_fft 512
      ↓
per-utterance mean normalization over time
      ↓
ReDimNet2-B4 backbone
```

72 bins, not 80: B4 (and the B6 teacher) were trained with `F = 72`, and a different bin count breaks the
pretrained backbone. Keep the checkpoint's `spec_params` as stored (`do_spec_aug: false`; no SpecAugment).

Do not feed WavLM features directly into the student.

The submitted model always consumes the normal B4 acoustic frontend.

WavLM exists only during training.

---

# 5. Dataset Manifest

Create one canonical manifest containing at least:

```yaml
audio_path:
speaker_id:
total_duration:
usable_duration:
active_intervals:
sample_rate:
num_samples:
```

Do not permanently modify source audio.

`active_intervals` may be obtained using simple energy-based speech-region detection.

Its purpose is only to avoid drawing very short training crops that contain mostly silence.

Do not introduce a neural VAD.

---

# 6. Speaker Sampling

Always sample speakers first.

Correct:

```text
uniform speaker
      ↓
utterance from speaker
      ↓
crop
```

Avoid:

```text
uniform audio file
```

because speakers with more recordings would otherwise be overrepresented.

---

# 7. Common Evaluation

Every E phase must be evaluated using the same evaluation code.

Report the existing challenge metrics plus duration-conditioned metrics.

Required natural-duration buckets:

```text
shorter side < 0.75 s
0.75–1.0 s
1–2 s
2–3 s
>3 s
```

Also generate controlled-duration tests:

```text
full ↔ full
full ↔ 3 s
full ↔ 2 s
full ↔ 1 s
full ↔ 0.75 s
full ↔ 0.5 s
2 s ↔ 2 s
1 s ↔ 1 s
```

Primary metric:

`EER`

Optionally report `minDCF` if the existing evaluator already supports it.

Do not select models based only on overall EER.

Always compare at minimum:

```text
overall
<1 s
1–2 s
full ↔ full
```

---

# 8. E0 — Pretrained B4 Standard Adaptation

## Purpose

Establish the clean control baseline.

Question answered:

> How strong is the approved pretrained B4 after ordinary low-LR adaptation to the challenge speakers?

## Model

```text
approved pretrained ReDimNet2-B4
        ↓
192-D embedding
        ↓
new SphereFace2 head
```

## Training duration

> **Superseded by §0:** E0 uses the same two-view batches (short 0.5–2 s, long 2.5–6 s) as E1, with one 192-D head.


Use fixed approximately:

`2-second crops`

for the baseline.

Do not use DAME.

Do not use WavLM.

Do not use B6 KD.

## Loss

```text
L = SphereFace2(z_192)
```

Use the existing/preferred SphereFace2-C implementation.

Target margin:

```text
m = 0.2
```

Do not apply the full margin abruptly at step 0.

Use a short margin warm-up.

## Learning rates

Pretrained model:

```text
very small LR
```

New classification head:

```text
much larger LR
```

Example:

```yaml
backbone: 1e-5
embedding: 3e-5
sf2_head: 1e-3
```

No head-only freeze stage is required.

The entire B4 can update from the beginning.

## Output

Store:

```text
E0_best.pt
```

and duration-stratified evaluation results.

E0 is the control for E1.

---

# 9. E1 — Pretrained B4 + DAME

## Purpose

Directly address short-duration representation failure.

Question answered:

> Does duration-aware Matryoshka supervision improve short-duration speaker verification compared with ordinary B4 fine-tuning?

Everything except the duration-aware objective should remain as close as possible to E0.

## Student

Same approved pretrained B4.

Same 192-D output.

Introduce nested prefixes:

```text
24-D
48-D
96-D
192-D
```

Conceptually:

```text
192-D embedding
│
├─ z[:24]
├─ z[:48]
├─ z[:96]
└─ z[:192]
```

Each prefix receives its own training-only SphereFace2 head.

The submitted model still outputs the full 192-D embedding.

## Duration views

> **Superseded by §0:** short view {0.5–2 s} and long view {2.5–6 s} replace the fixed 1 s / 2 s views.


Use:

```text
1-second view
2-second view
```

For each sampled speaker, preferably obtain:

```text
1s crop from utterance A
2s crop from utterance B
```

when multiple recordings are available.

Fallback:

- two different regions of the same recording;
- avoid identical/overlapping crops where possible.

Do not discard speakers simply because they have limited recordings.

## DAME target margins

Initial target configuration:

| Prefix | Target margin |
|---:|---:|
| 24D | 0.0 |
| 48D | 0.0 |
| 96D | 0.1 |
| 192D | 0.2 |

## Warm introduction of DAME

> **Superseded by §0:** no A/B/C stages; the exact C matrix and alpha(p) = max(0.5, 1 − p) provide the ramp.


Because B4 already has a meaningful pretrained embedding geometry, do not activate all prefix losses strongly at step 0.

Recommended progression:

### Stage E1-A

First ~10% of training:

```text
mostly 192-D supervision
```

### Stage E1-B

Introduce:

```text
96-D loss
```

### Stage E1-C

Gradually introduce:

```text
48-D
24-D
```

until full DAME supervision is active.

Implement this through configurable loss weights rather than separate model checkpoints.

## Soft weighting

> **Superseded by §0:** use the exact C matrix and alpha schedule given there.


Use soft DAME weighting rather than hard duration→single-prefix assignment.

The full objective is conceptually:

```text
L_DAME =
w24 * SF2(z24)
+
w48 * SF2(z48)
+
w96 * SF2(z96)
+
w192 * SF2(z192)
```

with weights dependent on duration and training progress.

The longer-duration 192-D task should dominate early training.

Gradually move toward balanced 1s/2s representation learning.

## Inference

Always use:

```text
full 192-D embedding
```

Do not select prefix dimension based on input duration during inference.

## Success criterion

E1 should improve:

```text
<1 s EER
1–2 s EER
```

without materially degrading:

```text
full ↔ full
overall EER
```

If E1 does not provide meaningful short-duration improvement, stop before E2 and diagnose E1.

---

# 10. E2 — B4 + DAME + WavLM KD

## Purpose

Add representation knowledge that is fundamentally different from B4's supervised speaker pretraining.

Question answered:

> Does WavLM's large-scale SSL speech representation provide useful information beyond pretrained B4 + DAME?

Start again from the approved B4 pretrained checkpoint.

Do not initialize E2 from E1.

This keeps the experiment clean:

```text
same B4 starting point
+
DAME
+
WavLM KD
```

rather than:

```text
E1 checkpoint
+
additional fine-tuning
```

---

## 10.1 WavLM Teacher

Use the approved:

`WavLM-Large`

Training only.

Do not use only the final WavLM layer.

Construct a speaker-oriented representation from multiple hidden layers.

Architecture:

```text
waveform
   ↓
WavLM-Large
   ↓
hidden layers H1...HL
   ↓
learnable weighted layer combination
   ↓
temporal/statistical pooling
   ↓
projection
   ↓
speaker embedding
```

## Teacher preparation

Initially freeze WavLM itself.

Train only:

- layer-combination weights
- pooling
- projection
- speaker-classification head

using the official challenge training speakers.

Do not fine-tune all 316M WavLM parameters in E2.

The resulting network becomes the fixed WavLM speaker teacher.

---

## 10.2 WavLM Distillation

Use two distillation signals.

### Embedding KD

```text
student B4 embedding
         ↓
training-only projection if necessary
         ↓
match WavLM speaker embedding
```

Loss:

```text
L_wav_emb = 1 - cosine(student, teacher)
```

### Relational KD

For batch embeddings construct:

```text
R_teacher[i,j] = cosine(t_i, t_j)
R_student[i,j] = cosine(s_i, s_j)
```

then minimize:

```text
SmoothL1(R_student, R_teacher)
```

or MSE if already implemented and more stable.

Prefer SmoothL1 initially.

## Do NOT include yet

Do not implement frame-by-frame WavLM→ReDimNet intermediate-feature KD in E2.

That adds substantial complexity and is a separate future experiment.

## E2 total loss

Conceptually:

```text
L =
L_DAME
+
lambda_wav_rel * L_wav_rel
+
lambda_wav_emb * L_wav_emb
```

Initial starting weights:

```yaml
lambda_wav_rel: 1.0
lambda_wav_emb: 0.5
```

Make them fully configurable.

Log separately:

```text
L_DAME
L_wav_rel
L_wav_emb
```

Monitor gradient dominance.

If KD overwhelms the task loss, reduce KD weights.

## Success criterion

Compare:

```text
E1 vs E2
```

The key question is whether WavLM improves:

```text
overall
<1s
1–2s
```

without losing normal-duration accuracy.

If E2 does not beat E1 consistently, do not proceed automatically to dual-teacher training.

---

# 11. E3 — B4 + DAME + WavLM + B6

## Purpose

Test whether a strong supervised speaker teacher still provides useful geometry beyond:

```text
pretrained B4
+
DAME
+
WavLM
```

This is optional and should only run after E2 is working well.

Question answered:

> Does B6 provide complementary speaker-discriminative geometry beyond the WavLM teacher?

Again start from the approved pretrained B4 checkpoint.

Do not initialize from E2.

---

## 11.1 B6 Teacher

Use approved:

`ReDimNet2-B6 — b6-vb2+vox2+cnc2_v0-lm`

Freeze it entirely.

Do not use B6 as a generic feature teacher.

Its specific responsibility is:

```text
speaker-embedding geometry
```

The previous system already successfully used B6 teacher embedding matching and batch-similarity matching.

---

## 11.2 B6 Relational KD

For each batch:

```text
B6 embeddings
      ↓
pairwise cosine matrix R_B6
```

Student:

```text
B4 embeddings
      ↓
pairwise cosine matrix R_student
```

Loss:

```text
L_b6_rel = SmoothL1(R_student, R_B6)
```

Primary B6 supervision should be relational.

Optionally include a weak direct embedding term:

```text
L_b6_emb = 1 - cosine(P(student), B6)
```

but keep its coefficient small.

---

# 12. E3 Teacher Responsibilities

Keep teacher roles deliberately different.

```text
WavLM
    ↓
representation knowledge
    ↓
what speech information should be retained?


B6
    ↓
speaker geometry
    ↓
how should speaker embeddings be arranged?
```

Do not simply average WavLM and B6 embeddings.

Do not force the student to regress equally toward two unrelated embedding spaces.

---

# 13. E3 Loss

Conceptually:

```text
L =
L_DAME

+ λ_wr * L_wav_rel
+ λ_we * L_wav_emb

+ λ_br * L_b6_rel
+ λ_be * L_b6_emb
```

Recommended initial values:

```yaml
lambda_wav_rel: 1.0
lambda_wav_emb: 0.5

lambda_b6_rel: 0.5
lambda_b6_emb: 0.1
```

These are starting values only.

The B6 contribution should initially be weaker than WavLM because the pretrained B4 already contains supervised SV knowledge.

---

# 14. Training Optimizer

For all phases use parameter groups.

Suggested optimizer:

```yaml
optimizer: AdamW or existing stable fine-tuning optimizer
```

If the existing ReDimNet2 implementation is strongly built around SGD/Nesterov, retain that implementation instead of introducing unnecessary optimizer changes.

The important requirement is discriminative learning rates.

Example:

```yaml
backbone_lr: 1e-5
embedding_lr: 3e-5
new_heads_lr: 1e-3
teacher_heads_lr: separate config
```

Use:

```text
warm-up
→ main fine-tuning
→ LR decay
```

Do not reuse the old C2X fine-tuning schedule blindly.

All scheduler values must be configuration parameters.

---

# 15. Checkpoint Selection

For every phase keep checkpoints based on a Pareto comparison of:

```text
overall EER
<1s EER
1–2s EER
full-full EER
```

Do not select solely by overall Silver EER.

A model that gains 0.03 overall but substantially worsens `<1s` should not automatically replace the previous phase.

---

# 16. Required Final Comparison

Produce this table after the campaign:

| System | Init | DAME | WavLM KD | B6 KD | Overall EER | <1s | 1–2s | full-full |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | pretrained B4 | No | No | No | | | | |
| E1 | pretrained B4 | Yes | No | No | | | | |
| E2 | pretrained B4 | Yes | Yes | No | | | | |
| E3 | pretrained B4 | Yes | Yes | Yes | | | | |

Interpretation:

```text
E0 → E1
= value of duration-aware representation

E1 → E2
= value of SSL/WavLM knowledge

E2 → E3
= marginal value of B6 speaker geometry
```

---

# 17. Go / No-Go Gates

Do not blindly run every stage.

## After E0

Always continue to E1.

## After E1

Continue to E2 only if:

- `<1s` or `1–2s` meaningfully improves; and
- full-duration verification is not badly degraded.

If not, debug DAME first.

## After E2

Run E3 if E2 is clearly competitive with/better than E1.

E3 is intended as refinement, not rescue.

## Final selection

Possible outcomes:

```text
E1 wins
→ DAME is valuable, teachers are unnecessary.

E2 wins
→ WavLM provides useful complementary knowledge.

E3 wins
→ WavLM + B6 knowledge is complementary.
```

There is no requirement that E3 must become the final submission.

---

# 18. Implementation Requirements

Coding agent should expose these through configuration rather than hard coding:

```yaml
model:
  checkpoint:
  embedding_dim:
  train_all_layers:

duration:
  crop_lengths:
  dame_enabled:
  dame_prefix_dims:
  dame_margins:
  dame_warmup:

optimizer:
  backbone_lr:
  embedding_lr:
  head_lr:
  weight_decay:
  scheduler:

wavlm_teacher:
  enabled:
  checkpoint:
  layer_mix:
  embedding_kd_weight:
  relational_kd_weight:

b6_teacher:
  enabled:
  checkpoint:
  embedding_kd_weight:
  relational_kd_weight:

evaluation:
  duration_buckets:
  controlled_duration_trials:
```

Use a single training implementation where E0–E3 are configuration variants, not four independent codebases.

---

# 19. Reproducibility

Every run must save:

```text
Git commit
full YAML config
random seed
dataset manifest version/hash
training speaker count
number of samples
model parameter count
optimizer/scheduler state
best checkpoint
duration evaluation table
```

Run at least one common fixed seed for E0–E3 before interpreting differences.

If a gain is small enough to plausibly be seed noise, rerun the relevant comparison with another seed.

---

# 20. Frozen Experiment Sequence

Final sequence:

```text
                    Approved B4
                        │
                        ▼
                E0 — adaptation
                        │
                  + DAME
                        ▼
                E1 — duration aware
                        │
                  + WavLM KD
                        ▼
                E2 — SSL teacher
                        │
             + B6 relational KD
                        ▼
                E3 — dual teacher
```

Final submitted inference model remains:

```text
original audio
      ↓
log-Mel frontend
      ↓
ReDimNet2-B4
      ↓
192-D embedding
      ↓
L2 normalization
      ↓
cosine score
```

No WavLM.

No B6.

No DAME auxiliary heads.

No teacher components.

Only the `<7M` ReDimNet2-B4 inference model remains.