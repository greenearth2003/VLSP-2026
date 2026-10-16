"""Label-free signal processing used at inference: energy VAD and per-file F0.

Both are computed from the waveform only (no speaker / trial information).
"""
from __future__ import annotations

import numpy as np

SR = 16000
FL, HOP = 400, 160  # 25 ms / 10 ms


def _frames(x: np.ndarray, fl: int = FL, hop: int = HOP) -> np.ndarray:
    if len(x) < fl:
        return x[None, :]
    n = 1 + (len(x) - fl) // hop
    return x[np.arange(fl)[None] + hop * np.arange(n)[:, None]]


def active_mask(x: np.ndarray) -> np.ndarray:
    """Frame-level speech activity: energy above max(p95 - 30 dB, p10 + 6 dB)."""
    e = 10 * np.log10(np.mean(_frames(x) ** 2, 1) + 1e-10)
    p10, p95 = np.percentile(e, [10, 95])
    return e > max(p95 - 30, p10 + 6)


def vad_trim(x: np.ndarray, hangover_s: float = 0.2, min_keep_s: float = 1.0) -> np.ndarray:
    """Drop silent stretches (> 2*hangover) anywhere in the file; keep original if too little remains."""
    m = active_mask(x)
    if m.all():
        return x
    h = int(hangover_s * SR / HOP)
    keep = np.convolve(m.astype(float), np.ones(2 * h + 1), mode="same") > 0  # dilate
    samp = np.repeat(keep, HOP)[: len(x)]
    if len(samp) < len(x):
        samp = np.r_[samp, np.repeat(keep[-1:], len(x) - len(samp))]
    y = x[samp]
    return y if len(y) >= min_keep_s * SR else x


def file_f0(x: np.ndarray) -> float:
    """Median F0 (Hz) by normalised autocorrelation on up to 300 active 40 ms frames (50-400 Hz)."""
    if len(x) < 640:
        return np.nan
    m = active_mask(x)
    fr = _frames(x, 640, 160)
    act = np.where(m[: len(fr)])[0]
    if len(act) == 0:
        return np.nan
    if len(act) > 300:
        act = act[np.linspace(0, len(act) - 1, 300).astype(int)]
    f = fr[act] * np.hanning(640)
    f = f - f.mean(1, keepdims=True)
    ac = np.fft.irfft(np.abs(np.fft.rfft(f, 1280)) ** 2)[:, :640]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    lo, hi = SR // 400, SR // 50
    lag = lo + ac[:, lo:hi].argmax(1)
    v = ac[np.arange(len(ac)), lag] > 0.45
    return float(np.median(SR / lag[v])) if v.sum() >= 5 else np.nan


def _info(path):
    from .data import load_wav
    x = load_wav(path)
    return len(vad_trim(x)) / SR, file_f0(x)


def file_info_cached(files: list[str], root, tag: str, workers: int = 3):
    """(VAD speech duration in s, F0 in Hz) per file, cached in lists/<tag>_vadf0.npy."""
    import hashlib
    from multiprocessing import Pool
    from pathlib import Path

    root_dir = Path(__file__).resolve().parent.parent / "lists"
    h = hashlib.md5("\n".join(files).encode()).hexdigest()[:8]
    path = root_dir / f"{tag}_vadf0_{h}.npy"
    if path.exists():
        return np.load(path)
    with Pool(workers) as p:
        out = np.array(p.map(_info, [str(Path(root) / f) for f in files], chunksize=64), dtype=float)
    np.save(path, out)
    return out
