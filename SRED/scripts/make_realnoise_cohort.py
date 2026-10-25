"""Real-background-noise copy of the train cohort (5 utts/speaker), same files as traincohort5 / traincohort_noisy:
the noise bank of scripts/make_realnoise_public.py (quiet frames of OTHER train recordings), private-like SNRs.

    python scripts/make_realnoise_cohort.py --out data/traincohort_realnoise
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_realnoise_public as mrp  # noqa: E402
from sred.data import TRAIN_DIR, cohort_subset, read_train_list  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/traincohort_realnoise")
    a = ap.parse_args()
    ci = cohort_subset(read_train_list(), per_spk=5)
    coh = {r for _, r in ci}
    ft = pd.read_csv(ROOT / "exp/analysis/features_train.csv")
    cand = ft[(ft.dur > 8) & (ft.speech_ratio < 0.8) & (~ft.rel.isin(coh))].sample(3000, random_state=1).rel.tolist()
    with Pool(32) as pool:
        bank = [b for b in pool.map(mrp.harvest, cand, chunksize=32) if b is not None and len(b) > 16000]
    print("noise bank:", len(bank), "segments (cohort files excluded)")
    fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv")
    ftr = ft.set_index("rel")
    draws = np.random.default_rng(12).choice(fv.snr_db.values, len(ci))
    mrp.PUBLIC_DIR = TRAIN_DIR
    jobs = [(r, float(min(draws[i], ftr.loc[r].snr_db)) - 3.0, 40_000_000 + i, a.out) for i, (_, r) in enumerate(ci)]
    with Pool(32, initializer=mrp.init, initargs=(bank,)) as pool:
        pool.map(mrp.job, jobs, chunksize=64)
    print("wrote", len(jobs), "files to", a.out)


if __name__ == "__main__":
    main()
