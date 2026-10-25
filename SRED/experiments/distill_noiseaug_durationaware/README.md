# distill-noiseaug-durationaware: data side

The data side of the E0–E3 campaign in `NEW_PLAN.md` (§0 has the locked decisions). The training pipeline (student,
DAME heads, teachers, optimizer) will be built on top of `build_loader`.

- **Two views per speaker.** Every step = 32 speakers × (short view, long view). The short view holds speech-dense
  crops of 0.5–2 s and the long view random crops of 2.5–6 s. The bins cover the shorter side of 97 % of private
  trials, and half of private trials have < 2 s of speech there.
- **Fixed crop bank.** Every crop the student can see is listed once, so WavLM and B6 embeddings of the clean crops
  can be cached offline and looked up per crop (`batch[view]["crop"]`).
- **Noise on the fly.** MUSAN, RIRS_NOISES, shared real train backgrounds on speaker pairs and channel effects are
  added in the loader workers. Teachers see the clean crop; the student sees the noisy one.
- **Duration-aware evaluation set.** Natural duration buckets and controlled-duration trials on the labelled public
  trials, applied to the silver audio.

From `sred/` the code imports only the data paths.

**Results: see `REPORT.md`** (technical report: method, results, ablations, submissions, reproduction).

## Files

| File | Role |
|---|---|
| `NEW_PLAN.md` | training plan E0–E3; §0 = locked decisions (views, DAME weights, teachers, budget) |
| `configs/base.json` | loader settings; override any key with `--set a.b=value` (JSON values) |
| `prep/build_manifest.py` | → `artifacts/manifest.jsonl` (+ `.sha256`): one line per utterance with speech intervals |
| `prep/build_crop_bank.py` | → `artifacts/crop_bank.npz`: the fixed crops, tied to the manifest hash |
| `prep/build_duration_trials.py` | → `artifacts/duration_eval/`: crops, trials, buckets and conditions for evaluation |
| `dnd/vad.py` | energy-based speech regions (same frame rule as `scripts/data_features.py`; no VAD model) |
| `dnd/data.py` | `build_loader(cfg)` → `TwoViewSampler` → `BankCropDataset` → `TwoViewCollate` |
| `dnd/noise.py` | `NoiseAugmenter`: reverb, shared backgrounds, additive noise, channel effects (numpy, per view) |
| `inspect_loader.py` | draws steps exactly as training would; writes `runs/_inspect/report.txt` and example wavs |
| `pretrained/` | approved weights for this experiment (git-ignored, see below) |

Rebuild everything (≈ 10 min):

```bash
cd SRED && source env.sh
.venv/bin/python experiments/distill_noiseaug_durationaware/prep/build_manifest.py        # ~5 min
.venv/bin/python experiments/distill_noiseaug_durationaware/prep/build_crop_bank.py       # ~1.5 min
.venv/bin/python experiments/distill_noiseaug_durationaware/prep/build_duration_trials.py # ~2.5 min
.venv/bin/python experiments/distill_noiseaug_durationaware/inspect_loader.py --steps 300
```

Noise banks (built once, git-ignored): `python scripts/make_ext_banks.py` writes `lists/rir_bank.npz` (4,825 RIRs) and
`lists/musan_bank.npz` (12.35 h); `lists/realnoise_bank.npz` comes from `scripts/make_noise_bank.py`.

## Pretrained weights (`pretrained/`)

| File | Model | Check |
|---|---|---|
| `redimnet2/b4-vox2-lm.pt` | ReDimNet2-B4, student (6.660M, 192-D, ASTP, mel frontend inside) | md5 `c756a165…`, = `SRED/pretrained/` |
| `redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt` | raw approved ReDimNet2-B6, teacher (12.459M) | md5 `996e5f31…`, = `SRED/pretrained/` |
| `wavlm/wavlm_large.pth` | WavLM-Large, teacher (315.5M) | sha256 `8c1e5637…`; loads strictly into `torchaudio.models.wavlm_large()`, 24 layers, 864× real time (64 × 2 s, bf16) |

## Use

```python
import sys; sys.path.insert(0, "experiments/distill_noiseaug_durationaware")   # from SRED/
from dnd.data import build_loader, load_config

cfg = load_config("experiments/distill_noiseaug_durationaware/configs/base.json")   # ["aug=null"] for clean crops
loader, info = build_loader(cfg, seed=0, workers=8)                                  # endless
for step, b in zip(range(n_steps), loader):
    short, long_ = b["short"], b["long"]   # same 32 speakers, same order
    short["wav"]     # [32, T_short] noisy crops, T = b["bins"][0] * 16000
    short["label"]   # [32] speaker ids (info["spk2id"])
    short["crop"]    # [32] crop-bank index -> row of the cached teacher embeddings of the CLEAN crop
    short["row"]     # [32] manifest row
    # short["rir_cat"], short["noise_cat"], short["snr"]: what was applied to each crop (-1 / nan = nothing)
```

| Setting | `--set` |
|---|---|
| clean crops | `aug=null` |
| fewer / more speakers per step | `speakers_per_step=16` |
| other bins (must exist in the bank) | `views.long_bins=[2.5,3.0,4.0]` |
| more shared-background pairs | `aug.p_shared=0.8` |
| drop a noise category | `aug.noise_cats.musan_speech=0` |

## Data flow, per step

```
manifest.jsonl (57,010 utts, 1,823 speakers, speech intervals)  ──►  crop_bank.npz (859,684 crops, 434 h)
  short bins 0.5 0.75 1 1.25 1.5 2 s : 2 speech-dense crops per utterance (non-overlapping where possible)
  long bins  2.5 3 4 5 6 s           : 1 random crop per utterance long enough (≥ 40 % speech), no tiling
        │
TwoViewSampler (main process)
  draw one short bin and one long bin (uniform)
  draw 32 distinct speakers uniformly among those with crops in both bins
  per speaker: long crop, then a short crop from ANOTHER utterance (else a disjoint part of the same file)
  order: [32 short | 32 long], same speaker order  ->  pairs (2k, 2k+1) inside a view = different speakers
        │
BankCropDataset (worker): read (start, length) from the source wav
        │
TwoViewCollate -> NoiseAugmenter, per view (worker, numpy)
  1 reverb      p 0.30  RIR from {sim small, sim medium, sim large, real}, RMS kept
  2 shared bg   p 0.50 per pair: the SAME train background in both crops, each at its own SNR 3–18 dB
  3 noise       p 0.60 for crops without a shared bg; one category, equal weight:
                  musan_noise 0–15 · musan_music 5–15 · musan_speech (3–7 talkers) 13–20 ·
                  rirs_pointsource 0–15 · rirs_isotropic 0–15 · train_bg 3–18 dB
  4 channel     band-limit 0.2 · mic EQ 0.3 · mu-law 0.1
        │
  {short: {wav, label, crop, row, rir_cat, noise_cat, snr}, long: {...}, bins: (short_s, long_s)}
```

## What it delivers (`inspect_loader.py`, base config, 300 steps = 19,200 crops)

| Check | Result |
|---|---|
| throughput, 8 workers | 5.3 steps/s (B4 training needs ≈ 0.7 steps/s at ≈ 1.5 s/step) |
| distinct speakers per step / different-speaker pairs | 300 / 300 · 300 / 300 |
| short and long view from different utterances | 99.3 % (same file: 39 disjoint, 35 overlapping) |
| speaker-uniform draws | sd 2.30 per speaker vs Poisson 2.29 |
| bins | short 12–20 % each, long 18–24 % each |
| speech fraction, median (10th pct) | short 1.00 (0.93) · long 0.89 (0.74) |
| reverb / shared bg / one noise type / no added noise | 29–30 % / 50 % / 30 % / 20 %, in both views |

Speakers without an utterance long enough for the drawn long bin are skipped for that step (70 speakers at 6 s);
they are still drawn in steps with shorter long bins.

## Duration-aware evaluation set (`artifacts/duration_eval/`, public labels, silver audio)

Speech per file = unpadded energy speech frames (= `features_public.csv` speech_dur). Controlled crops = the most
speech-dense N-second window, computed on clean public audio and applied to the silver audio (identical timelines).
Only the test side is cropped, except 2s-2s and 1s-1s; files with less speech than N are left out of condition N.

| Natural bucket (shorter side) | <0.75 s | 0.75–1 s | 1–2 s | 2–3 s | >3 s |
|---|---|---|---|---|---|
| trials | 2,983 | 4,520 | 19,042 | 11,016 | 12,432 |

| Controlled | full-full | full-3s | full-2s | full-1s | full-0.75s | full-0.5s | 2s-2s | 1s-1s |
|---|---|---|---|---|---|---|---|---|
| trials | 49,993 | 24,160 | 34,806 | 46,265 | 48,564 | 49,750 | 23,448 | 42,490 |

Silver is the only evaluation set; public_realnoise / public_noisy are not used.
