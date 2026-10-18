"""Labelled stand-in for the private test: public audio + synthetic noise matched to private SNR statistics.

    python scripts/make_noisy_public.py --out data/public_noisy

Per public file, a target SNR is drawn from the empirical SNR distribution of the PRIVATE files (only the
per-file SNR statistics are used; no private labels). Noise = coloured noise (1/f^beta, beta~U(0,2)) plus a
high-pass "hiss" component (private files show much stronger >4 kHz energy), mixed to reach the target SNR
relative to the speech. Trials/labels stay those of public-test.txt.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import PUBLIC_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def coloured(n, beta, rng):
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(n // 2 + 1).clip(1)
    return np.fft.irfft(spec / f ** (beta / 2), n=n)


def hiss(n, rng, sr=16000):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    spec[f < 4000] *= 0.1
    return np.fft.irfft(spec, n=n)


def job(args):
    rel, snr, seed, out = args
    rng = np.random.default_rng(seed)
    x, sr = sf.read(str(PUBLIC_DIR / rel), dtype="float32")
    n = len(x)
    noise = coloured(n, rng.uniform(0, 2), rng)
    noise /= np.sqrt(np.mean(noise ** 2)) + 1e-9
    h = hiss(n, rng)
    noise = noise + rng.uniform(0.3, 1.5) * h / (np.sqrt(np.mean(h ** 2)) + 1e-9)
    # speech power from the loud frames (approximately what the SNR feature measures)
    fr = x[: n // 400 * 400].reshape(-1, 400)
    p = np.mean(fr ** 2, 1)
    ps = np.mean(p[p >= np.quantile(p, 0.5)]) if len(p) else np.mean(x ** 2)
    noise *= np.sqrt(ps / (10 ** (snr / 10)) / (np.mean(noise ** 2) + 1e-12))
    y = x + noise
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y / peak * 0.99
    o = Path(out) / rel
    o.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(o), y.astype(np.float32), sr, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public_noisy")
    a = ap.parse_args()
    fp = pd.read_csv(ROOT / "exp/analysis/features_public.csv")
    fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv")
    rng = np.random.default_rng(0)
    # the speech-to-noise target per file: sample from private SNRs, but never ABOVE the file's own SNR
    # (we can only add noise); files already noisier than the draw are left as they are
    draws = rng.choice(fv.snr_db.values, len(fp))
    jobs = []
    for i, (rel, own) in enumerate(zip(fp.rel, fp.snr_db)):
        jobs.append((rel, float(min(draws[i], own)) - 3.0, i, a.out))  # -3 dB: added noise is on top of existing noise
    with Pool(48) as pool:
        pool.map(job, jobs, chunksize=64)
    print("wrote", len(jobs), "files to", a.out)


if __name__ == "__main__":
    main()
