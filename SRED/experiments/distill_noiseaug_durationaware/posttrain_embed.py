"""One embedding pass for the post-training study (posttrain_eval.py): everything later variants need, cached once.

    .venv/bin/python experiments/distill_noiseaug_durationaware/posttrain_embed.py MODEL --tag NAME

Writes runs/_calib/<tag>/posttrain.npz with, for the silver files and for the train cohort (5 utts / speaker, clean and
a noisy copy made with our training augmentation, same seeds as calibrate_eval.py):
  whole   raw (un-normalised) embedding of the clip (first 30 s)
  speech  raw embedding of the clip's speech only: energy active intervals (dnd/vad.py, no VAD model) concatenated;
          the whole clip when < 0.5 s of speech is found
  short   (cohort only) raw embedding of one speech-dense crop per file, length drawn from {0.5, 0.75, 1, 1.5, 2} s
  q_*     speech duration (s) and SNR (dB) of exactly the audio that was embedded
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

import evaluate as ev  # noqa: E402
from calibrate_eval import CAP, quality  # noqa: E402
from dnd.data import load_config  # noqa: E402
from dnd.noise import NoiseAugmenter  # noqa: E402
from dnd.vad import SR, active_intervals, dense_windows  # noqa: E402
from sred.data import TRAIN_DIR, cohort_subset, read_train_list  # noqa: E402

SHORT = [0.5, 0.75, 1.0, 1.5, 2.0]


def speech_only(x):
    iv = active_intervals(x)
    seg = np.concatenate([x[int(s * SR):int(e * SR)] for s, e in iv]) if iv else x[:0]
    return seg if len(seg) >= SR // 2 else x


@torch.inference_mode()
def raw(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(torch.from_numpy(np.ascontiguousarray(x[:CAP]))[None].cuda()).float()[0].cpu().numpy()


@torch.inference_mode()
def raw_batch(model, xs):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(torch.from_numpy(np.stack(xs)).cuda()).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = EXP / "runs/_calib" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    model, _ = ev.load_student(a.model)
    model = model.cuda().eval()
    R = defaultdict(list)

    tr = pd.read_csv(ev.DEV / "trials.csv")
    files = sorted(set(tr.a) | set(tr.b))
    for i, fn in enumerate(files):
        x, _ = sf.read(str(ev.SILVER / fn), dtype="float32", always_2d=False)
        x = x if x.ndim == 1 else x.mean(1)
        sp = speech_only(x[:CAP])
        R["silver_whole"].append(raw(model, x))
        R["silver_speech"].append(raw(model, sp))
        R["q_silver_whole"].append(quality(x[:CAP]))
        R["q_silver_speech"].append(quality(sp))
        if i % 5000 == 0:
            print(f"  silver {i}/{len(files)}", flush=True)

    items = read_train_list(SRED / "lists/clean/train_clean.txt")
    coh = cohort_subset(items, per_spk=5)
    cfg = load_config(EXP / "configs/base.json")
    b = cfg["banks"]
    aug = NoiseAugmenter(dict(cfg["aug"], p_shared=0.0), SRED / b["rir"], SRED / b["noise"], SRED / b["train_bg"])
    rng = np.random.default_rng(11)
    shorts = defaultdict(list)  # crop length -> [(cohort index, version, audio)]
    for i, (_, rel) in enumerate(coh):
        x, _ = sf.read(str(TRAIN_DIR / rel), dtype="float32", always_2d=False)
        x = (x if x.ndim == 1 else x.mean(1))[:CAP]
        y = aug(x[None].copy(), np.random.default_rng([7, i]))[0][0]  # same noisy copy as calibrate_eval.py
        L = int(SHORT[rng.integers(len(SHORT))] * SR)
        for v, z in (("clean", x), ("noisy", y)):
            sp = speech_only(z)
            R[f"coh_{v}_whole"].append(raw(model, z))
            R[f"coh_{v}_speech"].append(raw(model, sp))
            R[f"q_coh_{v}_whole"].append(quality(z))
            R[f"q_coh_{v}_speech"].append(quality(sp))
            if len(z) >= L:
                s0 = dense_windows(active_intervals(z), len(z), L, 1)[0]
                shorts[L].append((i, v, z[s0:s0 + L]))
        if i % 2000 == 0:
            print(f"  cohort {i}/{len(coh)}", flush=True)
    for v in ("clean", "noisy"):
        R[f"coh_{v}_short"] = [None] * len(coh)
        R[f"q_coh_{v}_short"] = [(np.nan, np.nan)] * len(coh)
    for L, xs in shorts.items():
        for j in range(0, len(xs), 64):
            chunk = xs[j:j + 64]
            embs = raw_batch(model, [z for _, _, z in chunk])
            for (i, v, z), e in zip(chunk, embs):
                R[f"coh_{v}_short"][i] = e
                R[f"q_coh_{v}_short"][i] = quality(z)
    dim = len(R["silver_whole"][0])
    res = {}
    for k, v in R.items():
        if k.startswith("q_"):
            res[k] = np.array(v, np.float32)
        else:
            res[k] = np.stack([e if e is not None else np.full(dim, np.nan, np.float32) for e in v])
    res["coh_spk"] = np.array([s for s, _ in coh])
    res["silver_files"] = np.array(files)
    np.savez(out / "posttrain.npz", **res)
    print("->", out / "posttrain.npz", {k: v.shape for k, v in res.items() if not k.startswith("q_")})


if __name__ == "__main__":
    main()
