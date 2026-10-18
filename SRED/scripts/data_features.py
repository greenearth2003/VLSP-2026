"""Per-file acoustic features for data analysis (train + public test).

    python scripts/data_features.py --out exp/analysis/features.csv [--workers 96]

One row per wav: format, duration, level/clipping, energy-VAD speech ratio, rough SNR,
bandwidth (highest bin within 50 dB of peak, 99% roll-off, HF energy ratio), median F0 (gender proxy), audio md5.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import PUBLIC_DIR, TRAIN_DIR, read_train_list, read_trials, unique_files

SR = 16000
FL, HOP = 400, 160  # 25 ms / 10 ms


def frame_db(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - FL) // HOP)
    idx = np.arange(FL)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[np.minimum(idx, len(x) - 1)]
    return 10 * np.log10(np.mean(fr**2, axis=1) + 1e-10), fr


def f0_median(frames: np.ndarray) -> float:
    """Autocorrelation F0 on (up to 300) loud frames; median over frames with clear periodicity."""
    if len(frames) == 0:
        return np.nan
    frames = frames[:: max(1, len(frames) // 300)]
    w = frames * np.hanning(FL)[None]
    spec = np.fft.rfft(w, n=2 * FL, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :FL]
    ac = ac / (ac[:, :1] + 1e-10)
    lo, hi = SR // 400, SR // 60  # 60–400 Hz
    seg = ac[:, lo:hi]
    lag = np.argmax(seg, axis=1) + lo
    peak = seg.max(axis=1)
    ok = peak > 0.5
    return float(np.median(SR / lag[ok])) if ok.sum() >= 5 else np.nan


def features(job: tuple[str, str, str]) -> dict:
    split, root, rel = job
    p = Path(root) / rel
    raw = p.read_bytes()
    info = sf.info(str(p))
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    d = dict(split=split, rel=rel, sr=sr, channels=info.channels, subtype=info.subtype,
             n=len(x), dur=len(x) / sr, md5=hashlib.md5(raw).hexdigest())
    if len(x) < FL:
        return d
    x = x - x.mean()
    peak = float(np.abs(x).max())
    db, fr = frame_db(x)
    p10, p50, p95 = np.percentile(db, [10, 50, 95])
    speech = db > max(p10 + min(12, 0.5 * (p95 - p10)), p95 - 35)
    sp_idx = np.flatnonzero(speech)
    # average magnitude spectrum over speech frames -> bandwidth
    sel = fr[speech][:: max(1, speech.sum() // 400)] if speech.any() else fr
    psd = np.mean(np.abs(np.fft.rfft(sel * np.hanning(FL)[None], axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(FL, 1 / SR)
    cum = np.cumsum(psd) / (psd.sum() + 1e-12)
    d.update(
        peak=peak,
        rms_db=float(10 * np.log10(np.mean(x**2) + 1e-10)),
        clip_frac=float(np.mean(np.abs(x) >= 0.99 * max(peak, 1e-6))) if peak > 0.95 else 0.0,
        speech_ratio=float(speech.mean()),
        speech_dur=float(speech.sum() * HOP / SR),
        snr_db=float(p95 - p10),
        noise_db=float(p10),
        lead_sil=float(sp_idx[0] * HOP / SR) if len(sp_idx) else np.nan,
        trail_sil=float((len(db) - 1 - sp_idx[-1]) * HOP / SR) if len(sp_idx) else np.nan,
        bw50=float(freqs[np.flatnonzero(10 * np.log10(psd / psd.max() + 1e-20) > -50).max()]),
        rolloff99=float(freqs[np.searchsorted(cum, 0.99)]),
        rolloff95=float(freqs[np.searchsorted(cum, 0.95)]),
        hf_ratio_db=float(10 * np.log10(psd[freqs >= 4000].sum() / (psd[freqs < 4000].sum() + 1e-12) + 1e-12)),
        f0=f0_median(fr[speech]),
    )
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp/analysis/features.csv")
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count()))
    ap.add_argument("--splits", nargs="+", default=["train", "public"])
    ap.add_argument("--limit", type=int, default=0, help="debug: first N files per split")
    a = ap.parse_args()

    jobs = []
    if "train" in a.splits:
        jobs += [("train", str(TRAIN_DIR), rel) for _, rel in read_train_list()]
    if "private" in a.splits:
        from sred.data import DATA_ROOT
        pt = DATA_ROOT / "private_test/private-test.txt"
        jobs += [("private", str(DATA_ROOT / "private_test/data"), f) for f in unique_files(read_trials(pt)[0])]
    if "public" in a.splits:
        jobs += [("public", str(PUBLIC_DIR), f) for f in unique_files(read_trials()[0])]
    if a.limit:
        jobs = [j for s in ("train", "public") for j in [j for j in jobs if j[0] == s][: a.limit]]
    print(f"{len(jobs)} files", flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(features, jobs, chunksize=32)):
            rows.append(r)
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
