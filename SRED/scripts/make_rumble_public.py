"""Fourth labelled stand-in: the real-noise public stand-in + low-frequency rumble on 30% of files, matched to the
"wave" backgrounds found in 13.8% of private clips (vs 0.2% of train): low-passed noise (cutoff 80-300 Hz), a slow
0.1-2 Hz swell, a 40-120 Hz hum in 30% of cases, full-band SNR -3..8 dB. Labels unchanged.

    python scripts/make_rumble_public.py --base data/public_realnoise --out data/public_rumble --p 0.3
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import read_trials, unique_files  # noqa: E402

SR = 16000


def job(args):
    rel, p, seed, base, out = args
    rng = np.random.default_rng(seed)
    x, _ = sf.read(str(Path(base) / rel), dtype="float32")
    if rng.random() < p:
        T = len(x)
        f = np.fft.rfftfreq(T, 1 / SR)
        fc = rng.uniform(80, 300)
        spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(1 + (f / fc) ** 8)
        n = np.fft.irfft(spec, n=T)
        n /= np.sqrt(np.mean(n ** 2)) + 1e-9
        t = np.arange(T) / SR
        if rng.random() < 0.3:
            n += 0.7 * np.sin(2 * np.pi * rng.uniform(40, 120) * t)
        n *= 1 + rng.uniform(0.3, 0.9) * np.sin(2 * np.pi * rng.uniform(0.1, 2.0) * t + rng.uniform(0, 6.283))
        snr = rng.uniform(-3, 8)
        n *= np.sqrt(np.mean(x ** 2) / (10 ** (snr / 10)) / (np.mean(n ** 2) + 1e-12))
        x = x + n
        x = x / max(1.0, np.abs(x).max() / 0.99)
    o = Path(out) / rel
    o.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(o), x.astype(np.float32), SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/public_realnoise")
    ap.add_argument("--out", default="data/public_rumble")
    ap.add_argument("--p", type=float, default=0.3)
    a = ap.parse_args()
    files = unique_files(read_trials()[0])
    with Pool(32) as pool:
        pool.map(job, [(r, a.p, 50_000_000 + i, a.base, a.out) for i, r in enumerate(files)], chunksize=64)
    print("wrote", len(files), "files to", a.out)


if __name__ == "__main__":
    main()
