"""Noisy copy of the train cohort (5 utts/speaker) with private-like SNRs, for noise-aware QMF fitting.

    python scripts/make_noisy_cohort.py --out data/traincohort_noisy
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_noisy_public as mnp  # noqa: E402
from sred.data import TRAIN_DIR, cohort_subset, read_train_list  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/traincohort_noisy")
    a = ap.parse_args()
    ci = cohort_subset(read_train_list(), per_spk=5)
    ft = pd.read_csv(ROOT / "exp/analysis/features_train.csv").set_index("rel")
    fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv")
    rng = np.random.default_rng(1)
    draws = rng.choice(fv.snr_db.values, len(ci))
    mnp.PUBLIC_DIR = TRAIN_DIR  # reuse the generator, reading from train audio
    jobs = [(r, float(min(draws[i], ft.loc[r].snr_db)) - 3.0, 10_000_000 + i, a.out) for i, (_, r) in enumerate(ci)]
    with Pool(48) as pool:
        pool.map(mnp.job, jobs, chunksize=64)
    print("wrote", len(jobs), "files to", a.out)


if __name__ == "__main__":
    main()
