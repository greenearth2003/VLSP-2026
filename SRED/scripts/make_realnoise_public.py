"""Second labelled stand-in: public audio + REAL background noise (quiet frames harvested from train recordings),
mixed at private-like SNRs. Guards against models that only learned the synthetic noise generator.

    python scripts/make_realnoise_public.py --out data/public_realnoise
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import PUBLIC_DIR, TRAIN_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FL, HOP = 400, 160
BANK = None


def harvest(rel):
    x, _ = sf.read(str(TRAIN_DIR / rel), dtype="float32")
    if len(x) < FL * 50:
        return None
    n = 1 + (len(x) - FL) // HOP
    db = np.array([10 * np.log10(np.mean(x[i * HOP:i * HOP + FL] ** 2) + 1e-10) for i in range(n)])
    p10, p95 = np.percentile(db, [10, 95])
    q = db <= min(p10 + 3, p95 - 20)
    seg = np.concatenate([x[i * HOP:(i + 1) * HOP] for i in np.flatnonzero(q)]) if q.sum() > 100 else None
    if seg is None or np.sqrt(np.mean(seg ** 2)) < 1e-4:  # skip digital silence
        return None
    return seg


def job(args):
    rel, snr, seed, out = args
    rng = np.random.default_rng(seed)
    x, sr = sf.read(str(PUBLIC_DIR / rel), dtype="float32")
    noise = BANK[rng.integers(len(BANK))]
    reps = int(np.ceil(len(x) / len(noise)))
    st = rng.integers(len(noise))
    noise = np.roll(np.tile(noise, reps + 1), -st)[: len(x)]
    fr = x[: len(x) // FL * FL].reshape(-1, FL)
    p = np.mean(fr ** 2, 1)
    ps = np.mean(p[p >= np.quantile(p, 0.5)])
    noise = noise * np.sqrt(ps / (10 ** (snr / 10)) / (np.mean(noise ** 2) + 1e-12))
    y = x + noise
    y = y / max(1.0, np.abs(y).max() / 0.99)
    o = Path(out) / rel
    o.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(o), y.astype(np.float32), sr, subtype="PCM_16")


def init(bank):
    global BANK
    BANK = bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public_realnoise")
    a = ap.parse_args()
    ft = pd.read_csv(ROOT / "exp/analysis/features_train.csv")
    cand = ft[(ft.dur > 8) & (ft.speech_ratio < 0.8)].sample(3000, random_state=0).rel.tolist()
    with Pool(48) as pool:
        bank = [b for b in pool.map(harvest, cand, chunksize=32) if b is not None and len(b) > 16000]
    print("noise bank:", len(bank), "segments")
    fp = pd.read_csv(ROOT / "exp/analysis/features_public.csv")
    fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv")
    rng = np.random.default_rng(11)
    draws = rng.choice(fv.snr_db.values, len(fp))
    jobs = [(rel, float(min(draws[i], own)) - 3.0, 20_000_000 + i, a.out) for i, (rel, own) in enumerate(zip(fp.rel, fp.snr_db))]
    with Pool(48, initializer=init, initargs=(bank,)) as pool:
        pool.map(job, jobs, chunksize=64)
    print("wrote", len(jobs), "files to", a.out)


if __name__ == "__main__":
    main()
