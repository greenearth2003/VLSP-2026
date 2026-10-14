# VLSP 2026 SRED — Speaker Verification on Edge Devices

Pipeline for the VLSP 2026 Vi-SRED26 challenge (EER metric, < 7M inference params).

## Setup

```bash
cd /home/jovyan/code/VLSP-2026/SRED
uv sync                       # creates .venv (torch 2.5.1 cu124)
source env.sh                 # LD_LIBRARY_PATH for the L4 driver, data root, torch hub cache
```

Data (unzipped from the official zips) lives in `$SRED_DATA` (default `/home/jovyan/data/VLSP2026/Vi-SRED26`):

```
train/train.txt         "<spk> <spk>/<utt>.wav"   58,404 utts, 1,900 speakers, ~144 h, 16 kHz mono
train/data/...
public_test/public-test.txt  "<a> <b> target|nontarget"  49,993 trials, 17,850 files, 902 speakers
public_test/data/...
```

Pretrained checkpoints are in `pretrained/` (downloaded from HuggingFace mirrors — GitHub release
assets are blocked from this machine). ReDimNet2 weights only exist as MLX safetensors on HF and are
converted to PyTorch layout on load (`sred/models.py::_load_redimnet2_mlx`).

## Layout

| file | purpose |
|---|---|
| `sred/models.py` | model zoo (`REGISTRY`), loaders for ReDimNet v1/v2 and WeSpeaker, param counting |
| `sred/data.py` | lists, trials, audio loading |
| `sred/embed.py` | embedding extraction (full / multi-crop), cached under `exp/emb/` |
| `sred/scoring.py` | cosine, mean-centering, AS-norm |
| `sred/backend.py` | LDA / WCCN / whitening / PLDA fitted on train embeddings |
| `sred/augment.py` | on-GPU augmentation: babble, coloured noise, synthetic RIR, band-limit, mic EQ, mu-law |
| `sred/train.py` | fine-tuning: sub-centre AAM-softmax, optional clean→noisy KD, partial freezing, bf16 |
| `sred/evaluate.py` | evaluate a model on a trial list with all back-ends, append to `exp/results.jsonl`, write submission |
| `scripts/` | experiment launchers |

## Usage

```bash
# zero-shot evaluation of a pretrained model (all back-ends)
python -m sred.evaluate --model redimnet_M_mix --tag zeroshot

# fine-tune
python -m sred.train --base redimnet_M_mix --name F01 --train-from stage3 --bs 32 --crop-sec 2 --amp bf16

# evaluate fine-tuned checkpoint + write submission file
python -m sred.evaluate --model exp/ft/F01/best.pt --submission submissions/F01.txt
```

Submission format: `file1.wav file2.wav score`, same order as the trial list.

## Notes / caveats

* Public-test paths contain speaker-ID directories; this is **not** used by any system here.
* The GPU (L4) is shared with another container (~15.5 GB used, 100 % util), so throughput is
  ~80× real-time for inference and training is slow; experiments are sized accordingly.
* Pretrained checkpoints must be on the organisers' approved list before use in a submission.

See `EXPERIMENTS.md` for results.
