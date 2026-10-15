"""Per-file acoustic statistics for train + public test (CPU, multiprocessing).

Output: eda/out/audio_stats.csv with one row per file:
  split, spk, rel, dur, sr, rms_dbfs, peak, clip_frac, dc, active_frac, lead_sil, trail_sil,
  noise_db, speech_db, snr_est, rolloff99, frac_above4k, frac_above7k, f0_med, f0_iqr, voiced_frac
"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sred.data import PUBLIC_DIR, PUBLIC_TRIALS, TRAIN_DIR, read_train_list, read_trials, unique_files  # noqa: E402

SR = 16000
FL, HOP = 400, 160  # 25 ms / 10 ms
OUT = Path(__file__).resolve().parent / "out"


def frames(x, fl=FL, hop=HOP):
    n = 1 + max(0, (len(x) - fl) // hop)
    idx = np.arange(fl)[None] + hop * np.arange(n)[:, None]
    return x[idx] if len(x) >= fl else x[None, :]


def f0_stats(x, active):
    """Autocorrelation F0 on up to 300 active 40 ms frames (50-400 Hz). Returns median, IQR, voiced fraction."""
    fr = frames(x, 640, 160)
    act = np.where(active[: len(fr)])[0]
    if len(act) == 0:
        return np.nan, np.nan, 0.0
    if len(act) > 300:
        act = act[np.linspace(0, len(act) - 1, 300).astype(int)]
    f = fr[act] * np.hanning(640)
    f = f - f.mean(1, keepdims=True)
    spec = np.fft.rfft(f, 1280)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, :640]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    lo, hi = SR // 400, SR // 50
    lag = lo + ac[:, lo:hi].argmax(1)
    peak = ac[np.arange(len(ac)), lag]
    v = peak > 0.45
    if v.sum() < 5:
        return np.nan, np.nan, float(v.mean())
    f0 = SR / lag[v]
    return float(np.median(f0)), float(np.subtract(*np.percentile(f0, [75, 25]))), float(v.mean())


def stats(args):
    split, spk, rel, root = args
    try:
        x, sr = sf.read(str(Path(root) / rel), dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001
        return dict(split=split, spk=spk, rel=rel, error=str(e))
    if x.ndim > 1:
        x = x.mean(1)
    dur = len(x) / sr
    peak = float(np.abs(x).max()) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    e = 10 * np.log10(np.mean(frames(x) ** 2, 1) + 1e-10)
    p10, p95 = np.percentile(e, [10, 95])
    active = e > max(p95 - 30, p10 + 6)
    a_idx = np.where(active)[0]
    lead = a_idx[0] * HOP / SR if len(a_idx) else dur
    trail = (len(e) - 1 - a_idx[-1]) * HOP / SR if len(a_idx) else dur
    # long-term spectrum (Welch-like, 512-pt)
    fr = frames(x, 512, 256) * np.hanning(512) if len(x) >= 512 else np.zeros((1, 512))
    ps = (np.abs(np.fft.rfft(fr, 512)) ** 2).mean(0)
    freqs = np.fft.rfftfreq(512, 1 / SR)
    cum = np.cumsum(ps) / max(ps.sum(), 1e-12)
    f0m, f0iqr, vfrac = f0_stats(x, np.repeat(active, 1)) if len(x) >= 640 else (np.nan, np.nan, 0.0)
    return dict(split=split, spk=spk, rel=rel, dur=round(dur, 3), sr=sr,
                rms_dbfs=round(20 * np.log10(rms), 2), peak=round(peak, 4),
                clip_frac=round(float(np.mean(np.abs(x) >= 0.999)), 6), dc=round(float(x.mean()), 5),
                active_frac=round(float(active.mean()), 3), lead_sil=round(lead, 2), trail_sil=round(trail, 2),
                noise_db=round(float(p10), 2), speech_db=round(float(p95), 2), snr_est=round(float(p95 - p10), 2),
                rolloff99=float(freqs[np.searchsorted(cum, 0.99)]),
                frac_above4k=round(float(ps[freqs >= 4000].sum() / max(ps.sum(), 1e-12)), 5),
                frac_above7k=round(float(ps[freqs >= 7000].sum() / max(ps.sum(), 1e-12)), 6),
                f0_med=round(f0m, 1) if f0m == f0m else "", f0_iqr=round(f0iqr, 1) if f0iqr == f0iqr else "",
                voiced_frac=round(vfrac, 3))


def main():
    jobs = [("train", s, r, str(TRAIN_DIR)) for s, r in read_train_list()]
    pairs, _ = read_trials(PUBLIC_TRIALS)
    jobs += [("public", f.split("/")[0], f, str(PUBLIC_DIR)) for f in unique_files(pairs)]
    OUT.mkdir(parents=True, exist_ok=True)
    keys = None
    with Pool(5) as pool, open(OUT / "audio_stats.csv", "w", newline="") as fh:
        for i, row in enumerate(pool.imap(stats, jobs, chunksize=64)):
            if keys is None:
                keys = list(row.keys()) + ["error"]
                w = csv.DictWriter(fh, keys, extrasaction="ignore")
                w.writeheader()
            w.writerow(row)
            if i % 5000 == 0:
                print(i, len(jobs), flush=True)
    print("done", len(jobs))


if __name__ == "__main__":
    main()
