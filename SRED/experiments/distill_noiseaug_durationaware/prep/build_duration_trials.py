"""Duration-aware evaluation set (NEW_PLAN §7, §0) on the labelled public trials.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/build_duration_trials.py

Crops: for every public file and N in {3, 2, 1, 0.75, 0.5} s, the N-second window with the most speech
(energy-based active intervals, deterministic), computed once on the CLEAN public audio and applied to the silver
audio (data/public_rumble_v2), which has identical timelines. Silver is the only evaluation set.
A file whose estimated speech (unpadded speech frames, = features_*.csv speech_dur) is < N is left out of
condition N. Natural buckets use the same estimate on the shorter side of each trial.

Writes artifacts/duration_eval/:
  crops.csv   file, N, start, length (samples), speech_frac
  trials.csv  a, b, label (1 target), clean (in lists/clean/public_trials_clean.txt), shorter_speech (s), bucket
  conditions.json  per condition: kind (natural / controlled), enrol / test crop length, trial indices kept
  summary.txt      trial counts and exclusions per condition
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
sys.path.insert(0, str(SRED))
sys.path.insert(0, str(EXP))

from dnd.vad import HOP, SR, active_intervals, dense_windows, frame_db, speech_frames, speech_mask  # noqa: E402
from sred.data import PUBLIC_DIR, PUBLIC_TRIALS  # noqa: E402

LENGTHS = [3.0, 2.0, 1.0, 0.75, 0.5]
BUCKETS = [(0, 0.75, "<0.75s"), (0.75, 1.0, "0.75-1s"), (1.0, 2.0, "1-2s"), (2.0, 3.0, "2-3s"), (3.0, 1e9, ">3s")]
# controlled conditions: (name, enrol crop or None = full, test crop or None = full)
CONTROLLED = [("full-full", None, None)] + [(f"full-{n:g}s", None, n) for n in LENGTHS] + \
             [("2s-2s", 2.0, 2.0), ("1s-1s", 1.0, 1.0)]


def one(f):
    x, sr = sf.read(str(PUBLIC_DIR / f), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    iv = active_intervals(x)
    # unpadded speech-frame total = features_*.csv speech_dur (the estimate behind the failure analysis and the
    # private-test duration stats); stricter than the padded intervals, so it also gates crop eligibility
    usable = float(speech_frames(frame_db(x)).sum() * HOP / SR)
    c = np.r_[0, np.cumsum(speech_mask(iv, len(x)))]
    crops = []
    for n in LENGTHS:
        L = int(round(n * SR))
        if usable < n or len(x) < L:
            continue
        s = dense_windows(iv, len(x), L, 1)[0]
        crops.append((f, n, s, L, float((c[s + L] - c[s]) / L)))
    return f, usable, crops


def main():
    out = EXP / "artifacts/duration_eval"
    out.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(PUBLIC_TRIALS, sep=r"\s+", header=None, names=["a", "b", "lab"])
    trials["label"] = (trials.lab == "target").astype(int)
    clean = pd.read_csv(SRED / "lists/clean/public_trials_clean.txt", sep=r"\s+", header=None, names=["a", "b", "lab"])
    trials["clean"] = pd.MultiIndex.from_frame(trials[["a", "b"]]).isin(pd.MultiIndex.from_frame(clean[["a", "b"]]))
    files = sorted(set(trials.a) | set(trials.b))
    with Pool(32) as pool:
        res = pool.map(one, files, chunksize=32)
    usable = {f: u for f, u, _ in res}
    crops = pd.DataFrame([c for _, _, cs in res for c in cs], columns=["file", "N", "start", "length", "speech_frac"])
    crops.to_csv(out / "crops.csv", index=False)

    trials["shorter_speech"] = np.minimum(trials.a.map(usable), trials.b.map(usable)).round(3)
    trials["bucket"] = ""
    for lo, hi, name in BUCKETS:
        trials.loc[(trials.shorter_speech >= lo) & (trials.shorter_speech < hi), "bucket"] = name
    trials[["a", "b", "label", "clean", "shorter_speech", "bucket"]].to_csv(out / "trials.csv", index=False)

    have = {n: set(crops.file[crops.N == n]) for n in LENGTHS}
    conds, lines = {}, []
    for lo, hi, name in BUCKETS:
        idx = np.flatnonzero(trials.bucket == name)
        conds[f"natural {name}"] = {"kind": "natural", "enrol": None, "test": None, "trials": idx.tolist()}
    for name, e, t in CONTROLLED:
        ok = np.ones(len(trials), bool)
        if e is not None:
            ok &= trials.a.isin(have[e]).to_numpy()
        if t is not None:
            ok &= trials.b.isin(have[t]).to_numpy()
        conds[name] = {"kind": "controlled", "enrol": e, "test": t, "trials": np.flatnonzero(ok).tolist()}
    (out / "conditions.json").write_text(json.dumps(conds))
    n_all = len(trials)
    lines.append(f"public trials {n_all} ({int(trials.clean.sum())} in the cleaned list), files {len(files)}; "
                 f"crops from clean public audio, applied to silver audio")
    for name, c in conds.items():
        k = np.array(c["trials"], int)
        tgt = int(trials.label.to_numpy()[k].sum()) if len(k) else 0
        lines.append(f"  {name:18s} {len(k):6d} trials ({tgt} target, {len(k) - tgt} nontarget)"
                     f"{'' if c['kind'] == 'natural' else f'  excluded {n_all - len(k)} (speech < crop)'}")
    for n in LENGTHS:
        k = crops.N == n
        lines.append(f"  crop {n:4g}s: {int(k.sum())} files, speech frac median {crops.speech_frac[k].median():.2f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
