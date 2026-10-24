# Handoff: distill-noiseaug-durationaware on a second training server

Branch `thucth-distill-noiseaug-durationaware`. Plan: `NEW_PLAN.md` (§0 = locked decisions). Data side: `README.md`.

## Roles

| Server | Does | Does not |
|---|---|---|
| **Reference** (`tsharp-B650M-AORUS-ELITE`, `/mnt/tsharp/thucth/dataset/vlsp2026-sred/VLSP-2026/SRED`) | builds the reference artifacts; runs ALL validation (silver, duration-aware set, Pareto table); keeps the comparison table | — |
| **Training server** (this handoff) | trains the phases assigned to it and sends checkpoints back | no evaluation, no public / silver / private audio needed |

Both servers must use byte-identical data-side inputs: crop ids index the teacher caches, so a crop bank that differs
by one crop silently mismatches every teacher target. `prep/verify_artifacts.py` checks this.

## Status (2026-10-24)

| Part | State |
|---|---|
| data side: manifest, crop bank, two-view loader with noise, duration-aware eval set | **done**, verified on the reference server |
| pretrained weights B4 / B6 / WavLM-Large | verified (hashes below) |
| training pipeline (E0–E3 `train.py`, DAME heads, teacher prep + caching) | **not in the branch yet**. It lands on this branch; `git pull` before training. Until then this server can only be set up and verified (steps 1–5). |

## Setup on the training server

`REF=<user>@<reference host>` and `R=/mnt/tsharp/thucth/dataset/vlsp2026-sred/VLSP-2026/SRED` below.

**0. Requirements.**

| Resource | Need |
|---|---|
| disk | ≈ 50 GB free: train zip 14.6 GB (delete after step 2) + train audio 16 GB + HF inputs 3.4 GB + teacher caches ≈ 0.7 GB + ≈ 0.4 GB per run; step 4 fails below 10 GB free |
| GPU | ≥ 20 GB (B4 at 32 speakers × 6 s long view, in two halves: ≈ 11 GB; WavLM-Large caching ≈ 5 GB) |
| CPU / disk speed | loader needs ≥ 0.7 steps/s; 8 workers gave 5.3 steps/s on the reference server |
| HF access | token with read access to `thucth/edge-device-speaker-recognition` (train zip) and `pieceofcakee/vlsp2026` |

**1. Code.**

```bash
git clone git@github.com:greenearth2003/VLSP-2026.git && cd VLSP-2026
git checkout thucth-distill-noiseaug-durationaware
cd SRED && uv sync            # or reuse a .venv with torch 2.5 + torchaudio 2.5 (cu124), numpy, scipy, soundfile, pandas
```

**2. Train audio: the original competition zip, nothing else.** Only `Vi-SRED26/train` is needed. Use the original
zip from the private HF repo so the audio is byte-identical to the reference server's; do not re-encode, resample or
convert anything. From `SRED/`:

```bash
hf download thucth/edge-device-speaker-recognition data/Vi-SRED26-Train.zip --local-dir hf_train
echo "f2201ab78b37ea1998fa437224fa55b76be1bcdd603f307488f6c1b287f357ce  hf_train/data/Vi-SRED26-Train.zip" | sha256sum -c -
mkdir -p data/Vi-SRED26/train && unzip -q hf_train/data/Vi-SRED26-Train.zip -d data/Vi-SRED26/train
ls data/Vi-SRED26/train          # must show data/ and train.txt; if the zip adds a top folder, move its contents up
ls data/Vi-SRED26/train/data | wc -l   # 1900 speaker folders
```

If the audio lives elsewhere, `export SRED_DATA=/path/to/Vi-SRED26` before `source env.sh`. Training uses the 57,010
utterances of `lists/clean/train_clean.txt` (tracked in git); step 4 checks every one of them.

**3. Pull the git-ignored inputs from Hugging Face** (private dataset repo `pieceofcakee/vlsp2026`, ≈ 3.4 GB; needs an
`hf auth login` token with access to the `pieceofcakee` org). Run from `SRED/`:

```bash
E=experiments/distill_noiseaug_durationaware
hf download pieceofcakee/vlsp2026 --repo-type dataset --include "$E/*" --local-dir hf_dnd
H=hf_dnd/$E
mkdir -p $E/pretrained $E/artifacts .cache/torch/hub
cp -r $H/pretrained/. $E/pretrained/                                # B4, raw B6, WavLM-Large (1.3 GB)
cp -r $H/artifacts/.  $E/artifacts/                                 # manifest, crop bank, duration eval set (41 MB)
cp $H/deps/lists/*.npz lists/                                       # rir / musan / realnoise banks (2.0 GB)
cp -r $H/deps/torch_hub/PalabraAI_redimnet2_main .cache/torch/hub/   # ReDimNet2 model code
```

The HF folder also holds a snapshot of this folder's code, but use the git branch for code (it has `sred/`, which
the experiment imports).

Alternative, copy from the reference server directly:

```bash
rsync -avP $REF:$R/$E/pretrained/ $E/pretrained/                 # 1.3 GB: B4, raw B6, WavLM-Large
rsync -avP $REF:$R/$E/artifacts/manifest.jsonl $REF:$R/$E/artifacts/manifest.sha256 \
           $REF:$R/$E/artifacts/crop_bank.npz $E/artifacts/     # 41 MB
rsync -avP $REF:$R/lists/rir_bank.npz $REF:$R/lists/musan_bank.npz $REF:$R/lists/realnoise_bank.npz lists/   # 2.0 GB
rsync -avP $REF:$R/.cache/torch/hub/PalabraAI_redimnet2_main/ .cache/torch/hub/PalabraAI_redimnet2_main/    # ReDimNet2 model code (2.4 MB)
```

If copying is not possible:
- pretrained weights: `prepare_pretrained.py` copies B4 / B6 from `SRED/pretrained/redimnet2/` (put the two `.pt`
  files there first), downloads WavLM-Large from `download.pytorch.org` and test-loads all three; step 4 must pass.
- manifest + crop bank are deterministic: `prep/build_manifest.py` then `prep/build_crop_bank.py` (≈ 7 min); step 4
  must still pass.
- `rir_bank` / `musan_bank`: download MUSAN (`EaseZh/musan`, `musan.tar.gz`, sha256 `86d1061c…955ee1`) and RIRS_NOISES
  (`EaseZh/rirs_noises`, `rirs_noises.zip`, sha256 `3b50cfde…c05dc7`) from Hugging Face, extract to one folder and run
  `python scripts/make_ext_banks.py --root <folder>`.
- `realnoise_bank` (train backgrounds) depends on reference-only files: copy it.
- ReDimNet2 code: `torch.hub` clone of `PalabraAI/redimnet2` (branch `main`) into `.cache/torch/hub/PalabraAI_redimnet2_main`.

**4. Verify: identical and complete data.** Must end with `all identical to the reference server`; do not train
otherwise. About 2 minutes.

```bash
source env.sh
.venv/bin/python $E/prep/verify_artifacts.py | tee $E/artifacts/verify_$(hostname).txt
```

It checks:
- **train audio**: all 57,010 wavs of the cleaned list exist, are 16 kHz with the manifest's sample count, and match
  the reference sha256 per file (`artifacts/train_audio.sha256`); 1,823 speakers, 143.7 h; raw `train.txt`
  58,404 lines / 1,900 speakers. Any missing or different file is listed by name.
- **inputs**: manifest `3e812ab8d273eb7a`, train-audio list `cf7abbdb4a4c7a90`, crop bank `3a3499fbd42c2a78`,
  rir `32dd50b5dfb45f0c`, musan `dc585c4c1a7b8001`, realnoise `3752564cd3a87852`, B4 `4f0b1e8bc6bcc104`,
  B6 `287365f6f485b19e`, WavLM `8c1e5637ae88657f` (`prep/expected_hashes.json`).
- **disk**: at least 10 GB free for teacher caches and run outputs.

If a train file differs, re-extract it from the zip (step 2). Never rebuild the manifest or crop bank on this server
to "fix" a mismatch: the crop ids would then differ from the reference server's.

**5. Loader smoke test.**

```bash
.venv/bin/python $E/inspect_loader.py --steps 300 --workers 8
```

Compare with the README table: all checks 300/300, views from different utterances ≈ 99.3 %, noise shares
≈ 30 / 50 / 30 / 20 %. Throughput must stay above ≈ 0.7 steps/s (B4 trains at ≈ 1.5 s/step); raise `--workers`
if the disk is slow.

## Training (once the pipeline is on the branch)

- Run only the phases assigned to this server (table below), with the phase config unchanged, seed 0, and the run
  name `<phase>_<server>` (e.g. `E2_srv2`).
- Teacher caches (B6, WavLM) are computed once per server from the same crop bank. With step 4 passing they are
  interchangeable, so a cache built on one server can be copied instead of recomputed.
- Budget per run ≈ 3.3 h for 8k steps on an RTX A4500-class GPU (20 GB); keep `speakers_per_step=32` (memory was
  measured for it).

| Phase | Server |
|---|---|
| E0 | TBD |
| E1 | TBD |
| E2 (+ WavLM teacher prep + cache) | TBD |
| E3 (+ B6 cache) | TBD |

Note: NEW_PLAN §17 gates E2 on E1 and E3 on E2. Running them in parallel skips those gates, so treat parallel results
as provisional until the gate comparisons are made on the reference server.

## Sending results back (validation stays on the reference server)

For each run copy the whole run folder into the same place on the reference server:

```bash
rsync -avP $E/runs/<run>/ $REF:$R/$E/runs/<run>/      # config.json, log.txt, step*.pt, last.pt
```

Each run folder must contain `config.json` (full resolved config), the `git rev-parse HEAD` of the code that trained
it, and the output of `prep/verify_artifacts.py` from the training server. The reference server runs every evaluation
(silver all / cleaned / private-like, natural duration buckets, controlled-duration conditions) so all numbers come
from one evaluator.
