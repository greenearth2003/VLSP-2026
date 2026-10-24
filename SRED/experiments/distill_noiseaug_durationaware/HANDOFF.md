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
| E0 / E1 training (`train.py`, `configs/e0.json`, `configs/e1.json`, `run.sh`, head-init centroids) | **ready**; E0 → E1 running on the reference server since 2026-10-24 |
| E2 / E3 (WavLM teacher prep, B6 / WavLM caches, KD terms) | **not in the branch yet**; lands on this branch, `git pull` + re-download `artifacts/` before E2 / E3 |

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

## Training

```bash
cd SRED && source env.sh
git pull                                                                   # latest trainer
hf download pieceofcakee/vlsp2026 --repo-type dataset --include "$E/artifacts/*" --local-dir hf_dnd
cp -r hf_dnd/$E/artifacts/. $E/artifacts/                                  # incl. b4_centroids.npy (head init)
.venv/bin/python $E/prep/verify_artifacts.py --skip-audio                  # still all ok
SEED=1 setsid nohup $E/run.sh e0:E0_s1_<server> e1:E1_s1_<server> > $E/runs/<server>.log 2>&1 &
tail -f $E/runs/E0_s1_<server>/log.txt
```

- `run.sh` takes `<config>:<run name>` pairs and runs them one after another; `SEED` (default 0) sets the seed.
  Re-running the same command resumes each run from its `last.pt`.
- Keep the phase configs unchanged; name runs `<phase>[_s<seed>]_<server>` so they never collide with the reference
  server's `E0`, `E1`, ...
- `artifacts/b4_centroids.npy` must come from the HF copy: recomputing it on another GPU gives slightly different
  head initialisations.
- Per run: 8,000 steps, ≈ 2–2.5 h on an RTX A4500 (20 GB), 11.4 GB peak; the first 300 steps train the heads only.

| Phase | Server |
|---|---|
| E0, E1 (seed 0) | reference server (running) |
| E0, E1 (seed 1): run-to-run noise for the E0 → E1 gate (NEW_PLAN §19) | training server, suggested |
| E2 (+ WavLM teacher prep + cache) | TBD, once the code lands |
| E3 (+ B6 cache) | TBD, once the code lands |

NEW_PLAN §17 gates E2 on E1 and E3 on E2. Running them in parallel skips those gates, so treat parallel results as
provisional until the gate comparisons are made on the reference server.

## Teacher caches (spec for E2 / E3; frozen 2026-10-24)

One file per teacher, embeddings of every CLEAN crop of the crop bank, computed once and read by `train.py`.

| Item | B6 (`prep/cache_b6.py`, owner B) | WavLM (`prep/wavlm_teacher.py`, owner A) |
|---|---|---|
| file | `artifacts/teacher_cache/b6.npy` + `b6.json` | `artifacts/teacher_cache/wavlm.npy` + `wavlm.json` |
| array | `float16`, shape `[859684, 192]`, C-contiguous, saved with `np.save` | same |
| row i | crop `i` of `artifacts/crop_bank.npz` (the index in `batch[view]["crop"]`) | same |
| model | raw approved B6 `pretrained/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt`, `sred.models._load_redimnet2_pt`, `eval()` | WavLM-Large frozen + the prepared speaker head (layer mix → attentive statistics pooling → linear 192) |
| output | `model(wav)`: the 192-D output of the final `linear` (the embedding the model is scored with) | the 192-D projection output of the speaker head |
| normalisation | L2-normalise each row in float32, then cast to float16 | same |

Input audio, exactly as `BankCropDataset` reads it (no augmentation, no extra gain or normalisation; each model
applies its own front-end):

```python
bank = np.load("artifacts/crop_bank.npz"); man = dnd.data.load_manifest("artifacts/manifest.jsonl")
r = bank["row"][i]
wav, _ = sf.read(TRAIN_DIR / man[r]["audio_path"], start=int(bank["start"][i]), frames=int(bank["length"][i]),
                 dtype="float32", always_2d=False)          # mono 16 kHz, length = bin_sec * 16000
```

Batching: every crop of one bin has the same length (`bank["length"]` takes 11 values), so batch crops of the same
bin together: no padding, no truncation. Any batch size, `torch.inference_mode()`, embeddings cast to float32 before
normalising. Process bins in any order and write each row to its index.

Precision (amended 14:20): use the cheapest setting that passes the 1,000-crop check below on crops of EVERY bin.
For B6, bf16 autocast fails it on 0.18 % of crops (0.5–1 s bins, min cos 0.9966), so B6 is cached in fp32 with TF32
matmul/conv (min cos 0.9997). Record the setting in the json `autocast` field (`"bf16"` or `"none (fp32, TF32)"`).

`<teacher>.json` (written with the cache):

```json
{"teacher": "b6", "model_file": "pretrained/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt", "model_hash16": "287365f6f485b19e",
 "crop_bank_hash16": "3a3499fbd42c2a78", "rows": 859684, "dim": 192, "dtype": "float16", "l2_normalised": true,
 "autocast": "bf16", "host": "<hostname>", "seconds": 0, "sha256": "<sha256 of the .npy file>"}
```

`model_hash16` / `crop_bank_hash16` are the `prep/verify_artifacts.py` digests. Checks before upload: no all-zero row,
every row norm within 1 ± 0.01, and cos(cache, fresh full-fp32 embedding) > 0.999 on 1,000 crops drawn from all bins.

How `train.py` uses it (E2 / E3): `T = torch.from_numpy(np.load(path)).to("cuda")` (330 MB), `t = T[batch[view]["crop"]]`
for the clean-crop targets of that view; the student sees the same crop with noise. Upload to HF at the same path:
`hf upload pieceofcakee/vlsp2026 $E/artifacts/teacher_cache/b6.npy $E/artifacts/teacher_cache/b6.npy --repo-type dataset`
(and `b6.json`).

## Sending results back through Hugging Face (validation stays on the reference server)

The training server has no SSH access to the reference server, so runs travel through the private HF repo, under the
same path as on disk. On the training server, after each checkpoint or at the end (re-running uploads only new files):

```bash
cd SRED && R=$E/runs/<run>
cp $E/artifacts/verify_$(hostname).txt $R/ ; git rev-parse HEAD > $R/git_commit.txt
hf upload pieceofcakee/vlsp2026 $R $R --repo-type dataset --exclude "last.pt" --exclude "*.tmp" \
  --commit-message "run <run> from $(hostname)"
```

This uploads `config.json`, `log.txt`, `train.out`, `step*.pt` (27 MB each) and the verification output;
`last.pt` (optimizer state, for resuming only) stays on the training server.

On the reference server:

```bash
cd SRED
hf download pieceofcakee/vlsp2026 --repo-type dataset --include "$E/runs/<run>/*" --local-dir .
.venv/bin/python $E/evaluate.py $E/runs/<run>/step8000.pt && .venv/bin/python $E/evaluate.py --table
```

Every run folder must contain `config.json` (resolved config with the git commit), `git_commit.txt` and the training
server's `verify_<host>.txt` ending in `all identical to the reference server`. All numbers (silver all / cleaned /
private-like, natural duration buckets, controlled-duration conditions) come from this one evaluator.
