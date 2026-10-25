"""Listen to the real-noise bank: rebuild it exactly as make_realnoise_public.py does, then write N random segments
(first 4 s each, loudness-normalised to -26 dBFS RMS so the quiet backgrounds are audible) + one concatenated file
with a short tone between segments.

    python scripts/sample_noise_bank.py --n 20 --out exp/analysis/noise_samples
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_realnoise_public import harvest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="exp/analysis/noise_samples")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    ft = pd.read_csv(ROOT / "exp/analysis/features_train.csv")
    cand = ft[(ft.dur > 8) & (ft.speech_ratio < 0.8)].sample(3000, random_state=0).rel.tolist()  # same as the bank
    with Pool(32) as pool:
        segs = pool.map(harvest, cand, chunksize=32)
    bank = [(rel, s) for rel, s in zip(cand, segs) if s is not None and len(s) > 16000]
    print("noise bank:", len(bank), "segments")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    pick = np.random.default_rng(a.seed).choice(len(bank), a.n, replace=False)
    t = np.arange(int(0.15 * SR)) / SR
    tone = (0.1 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t))).astype(np.float32)
    gap = np.zeros(int(0.35 * SR), np.float32)
    parts, rows = [], []
    for k, i in enumerate(pick, 1):
        rel, s = bank[i]
        s = s[: 4 * SR]
        rms_db = 20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12)
        y = (s * 10 ** ((-26 - rms_db) / 20)).clip(-0.99, 0.99).astype(np.float32)
        name = f"{k:02d}_{rel.replace('/', '_')}"
        sf.write(out / name, y, SR, subtype="PCM_16")
        parts += [tone, gap, y, gap]
        rows.append(f"{k:02d}  {rel}  original level {rms_db:.1f} dBFS  ({len(s) / SR:.1f} s)")
    sf.write(out / "all_20_segments.wav", np.concatenate(parts), SR, subtype="PCM_16")
    (out / "index.txt").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
