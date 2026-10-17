# Edge speaker-verification techniques — research note (2026-10-17)

Deep-research run (5 search angles, 19 primary sources fetched, 89 claims extracted, 25 verified by 3-vote
adversarial check: 24 confirmed, 1 refuted). Question: what improves SV for models **< 7M params** beyond
what we already do (ReDimNet-M 4.81M, sub-center AAM, aug, long→short embedding KD from B6, WiSE-FT, QMF).

Caveat that applies to everything below: almost every number is one lab's self-reported VoxCeleb result,
mostly models trained from scratch with baselines far weaker than ours (2.9 % cleaned EER); several are
unreviewed preprints. Expect our gains to be a fraction of the published relative numbers.

## Ranked techniques

| # | technique | evidence (model size) | expected for us | cost | fit |
|---|---|---|---|---|---|
| 1 | **Logit KD emphasising non-target speakers**: TRKD (confusion-set KD, tau curriculum 1.0→0.05, added to AAM) / DKD-γ (L = L_target + 2·L_non-target) | TRKD: ReDimNet-B5 7.65M → **ReDimNet-B2 4.89M**, Vox1-O 0.867 → 0.627 (no KD) vs 0.729 (DKD) / 0.697 (GKD); 18.7 % avg rel. over 6 students 3.8–6M. DKD-γ: WavLM-TDNN → ResNet34 6.6M, ~12 % rel., minDCF worse on Vox1-O | **2–9 %** on top of our embedding KD (TRKD's margin over earlier logit KD) | low — `C01_B6/last.pt` already has a head over the 1,823 cleaned speakers; add teacher logits on the full clean utterance | high |
| 1b | keep embedding KD, *add* the logit term | IML-KD study: plain logit KD alone hurt a ResNet18 (3.76 → 4.01) while embedding L2 helped (3.55) | — | — | design rule |
| 2 | **Semi-supervised DA with pseudo-labels** (k-NN graph + Infomap, thresholds set on labelled target speakers, sub-center purification, progressive merging — MoPC / DKU VoxSRC-23 T3) | SimAM-ResNet100: FFSVC 10.1 → 4.1 %; ECAPA ~14.7M VoxCeleb→CN-Celeb 13.4 → 8.1 % | unknown, likely small (we are already in-domain) | needs a large pool of **unlabelled Vietnamese speech** — we have none besides the test set | low unless such audio exists |
| 3 | **SphereFace2-C loss** (or simply drop sub-centres) | ReDimNet-B2 ablation, avg Vox1 EER: sub-center AAM 1.03, AAM 0.99, SF2-A 0.94, **SF2-C 0.88** (~11 % rel.); ReDimNet B2/B5/B6 checkpoints were trained with SF2 | 0–10 %; part may come just from dropping sub-centres | very low — SF2 code vendored in `third_party/wespeaker/models/projections.py` | high |
| 4 | QMF with more quality measures (embedding magnitudes, SNR, + AS-norm score) | DKU VoxSRC-23, SimAM-ResNet100: 3.04 → 2.39 on VoxSRC23-val | small: our SNR/AS-norm QMF variants were negative; **embedding magnitude untested** | minutes | medium |
| 5 | Short-utterance methods (segment aggregation + teacher-student, MR-RawNet, vector archives, diffusion extension) | 20–55 % rel. at 1 s, but on 15–18 % EER baselines or >15M models; several get worse at ≥3 s | segment-aggregation idea ≈ our long→short KD; rest over budget | — | low |
| 6 | Architecture change | Under 7M ReDimNet B2/B4 beat ECAPA-512, CAM++, ResNet34, DF-ResNet56; ERes2NetV2 gains need 17.8M. Claim that ReDimNet2-B4 is a better sub-7M student was **refuted** (0-3) | none | — | not recommended |

## Gaps (no verified evidence found)
Quantization-aware training / INT8 accuracy cost for ReDimNet (measure directly if the edge target needs
quantization); VLSP / edge-SV small-footprint challenge systems; WavLM/XLS-R feature- or attention-level
distillation into compact students; pooling variants; targeted fixes for male–male or within-speaker
pitch/style confusions.

## Proposed experiments on this branch (all on top of K21, judged on P-dev with the bootstrap rule)
1. **E1 — TRKD/DKD-γ logit term** added to the K21 recipe: teacher = `C01_B6` (head over the cleaned
   1,823 speakers), teacher logits from the full clean utterance (cache once), student sees short crops.
2. **E2 — loss swap**: K21 recipe with (a) plain AAM (no sub-centres), (b) SphereFace2-C.
3. **E3 — QMF + embedding magnitude** (no retraining).
4. Measure INT8 post-training quantization EER of K21 if edge deployment requires it.

## Sources
- TRKD — https://arxiv.org/html/2601.14699v1
- DKD-γ (ICASSP 2024) — https://arxiv.org/pdf/2309.14838
- IML-KD — https://arxiv.org/html/2409.09389v1
- MoPC — https://arxiv.org/pdf/2310.04760 · DKU-MSXF VoxSRC-23 — https://arxiv.org/pdf/2308.08766 · cluster-guided UDA — https://arxiv.org/pdf/2303.15944
- ReDimNet — https://arxiv.org/pdf/2407.18223 · ReDimNet2 — https://arxiv.org/pdf/2603.11841
- MFA-TDNN — https://arxiv.org/pdf/2202.01624 · ERes2NetV2 — https://arxiv.org/pdf/2406.02167
- Short-utterance: https://arxiv.org/pdf/2005.03329 · https://arxiv.org/pdf/2406.07103 · https://arxiv.org/pdf/2609.25007 · https://arxiv.org/pdf/2310.04681
