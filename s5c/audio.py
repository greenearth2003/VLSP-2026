"""Audio loading and the two per-file quality measures used by Q3 (duration, SNR estimate).
They reproduce eda/audio_stats.py of the competition repo exactly, INCLUDING its rounding (duration to 3 decimals,
SNR to 2 decimals), so scores are bit-identical to the submitted private file."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR, FL, HOP = 16000, 400, 160  # 25 ms frames, 10 ms hop


def load_wav(path: str | Path) -> np.ndarray:
    """float32 mono in [-1, 1]; multi-channel files are averaged; the sample rate must be 16 kHz (no resampling)."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr} != 16000 (resample first)")
    return x


def duration_s(x: np.ndarray) -> float:
    return round(len(x) / SR, 3)


def snr_est(x: np.ndarray) -> float:
    """Frame-energy SNR estimate in dB: 95th minus 10th percentile of 25 ms / 10 ms frame energies
    (10 log10(mean(x^2) + 1e-10)), rounded to 2 decimals. Model-free; no VAD."""
    if len(x) >= FL:
        n = 1 + (len(x) - FL) // HOP
        fr = x[np.arange(FL)[None] + HOP * np.arange(n)[:, None]]
    else:
        fr = x[None, :]
    e = 10 * np.log10(np.mean(fr ** 2, 1) + 1e-10)
    p10, p95 = np.percentile(e, [10, 95])
    return round(float(p95 - p10), 2)
