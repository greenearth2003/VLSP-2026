"""Third labelled stand-in: noisy public + a second speaker's TURN spliced in (sequential, not overlapping), as in
the private interview-style clips. Labels unchanged (the trial's speaker is still the host file's speaker).

    python scripts/make_multi_public.py --base data/public_noisy --out data/public_noisy_multi --p 0.35

For each file with prob p: take a 1.5-4 s stretch of another file from a different folder (public folders are
speakers), match its level to the host within +-3 dB, insert it at the start, the end or a random point.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import read_trials, unique_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def job(args):
    rel, other, p, seed, base, out = args
    rng = np.random.default_rng(seed)
    x, _ = sf.read(str(Path(base) / rel), dtype="float32")
    if rng.random() < p and other is not None:
        y, _ = sf.read(str(Path(base) / other), dtype="float32")
        L = int(rng.uniform(1.5, 4.0) * SR)
        if len(y) > L:
            st = rng.integers(0, len(y) - L)
            y = y[st:st + L]
        rms = lambda a: np.sqrt(np.mean(a ** 2) + 1e-10)  # noqa: E731
        y = y * rms(x) / rms(y) * 10 ** (rng.uniform(-3, 3) / 20)
        pos = rng.choice(["start", "end", "mid"])
        cut = 0 if pos == "start" else len(x) if pos == "end" else rng.integers(0, len(x))
        x = np.concatenate([x[:cut], y, x[cut:]])
        x = x / max(1.0, np.abs(x).max() / 0.99)
    o = Path(out) / rel
    o.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(o), x.astype(np.float32), SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/public_noisy")
    ap.add_argument("--out", default="data/public_noisy_multi")
    ap.add_argument("--p", type=float, default=0.35)
    a = ap.parse_args()
    pairs, _ = read_trials()
    files = unique_files(pairs)
    folder = np.array([f.split("/")[0] for f in files])
    rng = np.random.default_rng(23)
    jobs = []
    for i, rel in enumerate(files):
        other = None
        for _ in range(10):
            j = rng.integers(len(files))
            if folder[j] != folder[i]:
                other = files[j]
                break
        jobs.append((rel, other, a.p, 30_000_000 + i, a.base, a.out))
    with Pool(32) as pool:
        pool.map(job, jobs, chunksize=64)
    print("wrote", len(jobs), "files to", a.out)


if __name__ == "__main__":
    main()
