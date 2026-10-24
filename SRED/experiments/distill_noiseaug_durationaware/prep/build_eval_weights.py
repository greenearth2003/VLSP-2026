"""Freeze the per-trial silver metadata that evaluate.py needs, so any server can evaluate without the reference
server's private files (scripts/c2_eval.py reads the teammate's C2 package, hf_repo metadata and the private trial
list). Run once on the reference server.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/build_eval_weights.py

Writes artifacts/duration_eval/silver_weights.npz, rows in public-test.txt order (= trials.csv):
  private_like  float64 [49993]  weights that re-balance public non-targets to private's channel / gender mix
                                 (C2X report's "private-like" EER)
  in_cleaned    bool    [49993]  trial is in the cleaned list used for the C2X report's "cleaned" EER
  pairs_sha256                  hash of the "a b" trial lines, checked by evaluate.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(SRED / "scripts")]

from sred.data import read_trials  # noqa: E402


def pairs_hash(a, b) -> str:
    return hashlib.sha256("".join(f"{x} {y}\n" for x, y in zip(a, b)).encode()).hexdigest()


def main():
    import c2_eval as C  # reference-server only

    pairs, y = read_trials()
    tr = pd.read_csv(EXP / "artifacts/duration_eval/trials.csv")
    assert list(map(tuple, tr[["a", "b"]].to_numpy())) == [tuple(p) for p in pairs]
    w = C.private_like_weights(pairs, y)
    cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(pd.read_csv(C.META), how="left") \
        .in_cleaned.fillna(False).to_numpy(bool)
    out = EXP / "artifacts/duration_eval/silver_weights.npz"
    np.savez(out, private_like=np.asarray(w, np.float64), in_cleaned=cl, pairs_sha256=np.array(pairs_hash(tr.a, tr.b)))
    print(f"{out}: {len(w)} trials, {int(cl.sum())} in the cleaned list, private-like weight sum {w.sum():.1f}")


if __name__ == "__main__":
    main()
