"""Embed ALL W-s chunks (hop W/2, no speech filter, at most 24 per clip, evenly spread) of the clips longer than
--min-dur seconds, for segment-gated chunk scoring (scripts/segment_fix.py). Shorter clips get n = 0.

    python scripts/chunk_embed.py --model exp/ft/S3_E2b_X2/avg.pt --splits traincohort5 public_rumble_v2 private --win 2 --min-dur 8

Cache: exp/emb/<model>/<split>_chunk<W>_gt<min-dur>.npz with E [files, 24, 192] and n [files], rows in the usual
file order of the split (unique_files of its trial list; cohort_subset order for traincohort5).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sred.data import TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from window_score import KMAX, embed_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def split_files(split):
    if split == "traincohort5":
        cf = [r for _, r in cohort_subset(read_train_list(), per_spk=5)]
        lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
        return cf, TRAIN_DIR, np.array([lengths[r] / 16000 for r in cf])
    if split == "private":
        files = unique_files(read_trials(ROOT / "submissions/final_private/trials_2col.txt")[0])
        return files, ROOT / "data/Vi-SRED26/private_test/data", np.load(ROOT / "lists/private_durations.npy")
    files = unique_files(read_trials()[0])
    root = ROOT / "data" / split
    dur = np.array([sf.info(str(root / f)).duration for f in files])
    return files, root, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--min-dur", type=float, default=8.0)
    a = ap.parse_args()
    for split in a.splits:
        files, root, dur = split_files(split)
        out = cache_key(a.model, split, files, mode="full").with_name(f"{split}_chunk{a.win:g}_gt{a.min_dur:g}.npz")
        if out.exists():
            print("cached", out)
            continue
        sel = np.where(dur > a.min_dur)[0]
        part = out.with_suffix(".part.npz")
        Es, ns = embed_windows(a.model, [files[i] for i in sel], root, a.win, part, min_speech=0.0)
        E = np.zeros((len(files), KMAX, Es.shape[2]), np.float32)
        n = np.zeros(len(files), np.int32)
        E[sel], n[sel] = Es, ns
        np.savez(out, E=E, n=n)
        part.unlink()
        print(f"{split}: {len(sel)} of {len(files)} clips > {a.min_dur:g} s, chunks/clip median {np.median(ns):.0f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
