"""Fixed crop bank (NEW_PLAN §0): every crop the student can see, so teacher embeddings can be cached offline.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/build_crop_bank.py

Short bins (speech-dense placement): 2 crops per utterance, the windows with the most active-interval speech,
non-overlapping where the clip allows. Long bins (random placement, pauses kept): 1 crop per utterance, a random
start among up to 10 tries whose window holds >= 40 % speech (else the best try). A bin gets crops only from
utterances at least that long (no tiling). Writes artifacts/crop_bank.npz:
  row (manifest row), view (0 short / 1 long), bin_sec, start, length (samples), speech_frac
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from dnd.vad import SR, dense_windows, speech_mask  # noqa: E402

SHORT = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
LONG = [2.5, 3.0, 4.0, 5.0, 6.0]
PER_SHORT, PER_LONG, LONG_MIN_SPEECH = 2, 1, 0.4


def main():
    rng = np.random.default_rng(0)
    man = [json.loads(ln) for ln in open(EXP / "artifacts/manifest.jsonl")]
    rows, views, bins, starts, lens, fracs = [], [], [], [], [], []

    def add(r, v, b, s, L, cover):
        rows.append(r); views.append(v); bins.append(b); starts.append(s); lens.append(L)  # noqa: E702
        fracs.append(cover / L)

    for m in man:
        n, iv, r = m["num_samples"], m["active_intervals"], m["row"]
        c = np.r_[0, np.cumsum(speech_mask(iv, n))]
        for b in SHORT:
            L = int(round(b * SR))
            if n < L:
                continue
            for s in dense_windows(iv, n, L, PER_SHORT):
                add(r, 0, b, s, L, c[s + L] - c[s])
        for b in LONG:
            L = int(round(b * SR))
            if n < L:
                continue
            tries = rng.integers(0, n - L + 1, 10)
            cover = c[tries + L] - c[tries]
            ok = np.flatnonzero(cover >= LONG_MIN_SPEECH * L)
            j = ok[0] if len(ok) else int(np.argmax(cover))
            add(r, 1, b, int(tries[j]), L, cover[j])

    out = EXP / "artifacts/crop_bank.npz"
    man_hash = (EXP / "artifacts/manifest.sha256").read_text().split()[0]
    np.savez(out, row=np.array(rows, np.int32), view=np.array(views, np.int8), bin_sec=np.array(bins, np.float32),
             start=np.array(starts, np.int64), length=np.array(lens, np.int32), speech_frac=np.array(fracs, np.float32),
             manifest_sha256=np.array(man_hash))
    lens_a, fr, bn = np.array(lens), np.array(fracs), np.array(bins)
    print(f"{len(rows)} crops, {lens_a.sum() / SR / 3600:.0f} h; manifest {man_hash[:12]}; "
          f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:12]}")
    for b in SHORT + LONG:
        k = bn == b
        print(f"  {b:4}s  {k.sum():7d} crops  speech frac median {np.median(fr[k]):.2f}  "
              f"<50% speech {np.mean(fr[k] < 0.5) * 100:4.1f}%")


if __name__ == "__main__":
    main()
