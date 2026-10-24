"""Energy-based speech regions (no VAD model), same frame rule as SRED/scripts/data_features.py (speech_dur there):
a 25 ms / 10 ms frame is speech when its level exceeds max(p10 + min(12, (p95 - p10) / 2), p95 - 35) dB.
Frames are then joined into intervals: gaps < 0.2 s merged, intervals < 0.1 s dropped, 50 ms padding."""
from __future__ import annotations

import numpy as np

SR, FL, HOP = 16000, 400, 160


def frame_db(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - FL) // HOP)
    idx = np.arange(FL)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[np.minimum(idx, len(x) - 1)]
    return 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)


def speech_frames(db: np.ndarray) -> np.ndarray:
    p10, p95 = np.percentile(db, [10, 95])
    return db > max(p10 + min(12, 0.5 * (p95 - p10)), p95 - 35)


def active_intervals(x: np.ndarray, min_gap=0.2, min_seg=0.1, pad=0.05) -> list[tuple[float, float]]:
    """[(start_s, end_s)] of speech, sorted and non-overlapping."""
    sp = speech_frames(frame_db(x))
    if not sp.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, sp.astype(np.int8), 0]))
    segs = [[s * HOP / SR, (e * HOP + FL) / SR] for s, e in zip(edges[0::2], edges[1::2])]
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    dur = len(x) / SR
    return [(round(max(0.0, s - pad), 3), round(min(dur, e + pad), 3)) for s, e in merged if e - s >= min_seg]


def speech_mask(intervals, n: int) -> np.ndarray:
    """Per-sample boolean mask of length n from intervals in seconds."""
    m = np.zeros(n, bool)
    for s, e in intervals:
        m[int(s * SR):int(e * SR)] = True
    return m


def dense_windows(intervals, n: int, L: int, k: int, hop: int = 160) -> list[int]:
    """Starts of up to k length-L windows with the most speech, preferring non-overlapping ones
    (a later pick overlaps an earlier one only when no non-overlapping window is left)."""
    if n <= L:
        return [0]
    c = np.r_[0, np.cumsum(speech_mask(intervals, n))]
    starts = np.arange(0, n - L + 1, hop)
    cover = c[starts + L] - c[starts]
    order = starts[np.argsort(-cover, kind="stable")]
    picks = []
    for s in order:
        if all(abs(int(s) - p) >= L for p in picks):
            picks.append(int(s))
        if len(picks) == k:
            return picks
    for s in order:  # not enough room for non-overlapping windows: allow overlap, but at least L/2 apart
        if len(picks) == k:
            break
        if all(abs(int(s) - p) >= L // 2 for p in picks):
            picks.append(int(s))
    return picks
