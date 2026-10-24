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

**1. Code.**

```bash
git clone git@github.com:greenearth2003/VLSP-2026.git && cd VLSP-2026
git checkout thucth-distill-noiseaug-durationaware
cd SRED && uv sync            # or reuse a .venv with torch 2.5 + torchaudio 2.5 (cu124), numpy, scipy, soundfile, pandas
```

**2. Train audio.** Only `Vi-SRED26/train` is needed. Put it at `SRED/data/Vi-SRED26/train/{data,train.txt}` or
export `SRED_DATA=/path/to/Vi-SRED26` before `source env.sh`.

**3. Copy from the reference server** (all git-ignored; ≈ 3.4 GB).

```bash
E=experiments/distill_noiseaug_durationaware
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

**4. Verify** (must print `all identical to the reference server`).

```bash
source env.sh
.venv/bin/python $E/prep/verify_artifacts.py
```

Reference hashes (`prep/expected_hashes.json`): manifest `3e812ab8d273eb7a`, crop bank `3a3499fbd42c2a78`,
rir `32dd50b5dfb45f0c`, musan `dc585c4c1a7b8001`, realnoise `3752564cd3a87852`, B4 `4f0b1e8bc6bcc104`,
B6 `287365f6f485b19e`, WavLM `8c1e5637ae88657f`.

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
