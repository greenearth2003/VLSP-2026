"""Q3 score calibration (fitted for S5): a logistic regression on 9 trial-level features (cosine, durations, SNRs and their
interactions), fitted ONLY on train-cohort pairs (clean + re-recorded with train-half background noise). The fitted
parameters are frozen in assets/q3_qmf.json; no fitting happens at inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ASSET = Path(__file__).resolve().parent / "assets/q3_qmf.json"


class Q3:
    def __init__(self, path: Path = ASSET):
        p = json.loads(Path(path).read_text())
        self.mean, self.std = np.array(p["mean"]), np.array(p["std"])
        self.coef, self.intercept = np.array(p["coef"]), float(p["intercept"])

    @staticmethod
    def features(s, da, db, qa, qb) -> np.ndarray:
        a, b = np.log(np.minimum(da, db)), np.log(np.maximum(da, db))
        lo = np.clip(np.minimum(qa, qb), -5, 40) / 10
        hi = np.clip(np.maximum(qa, qb), -5, 40) / 10
        return np.stack([s, a, b, s * a, s * b, lo, hi, s * lo, s * hi], 1)

    def score(self, cos, dur_a, dur_b, snr_a, snr_b) -> np.ndarray:
        """All inputs are arrays over trials. Returns calibrated log-odds (higher = same speaker).
        The 9 features are computed in float32 exactly as in the competition pipeline (float32 embeddings, durations
        and SNRs), then standardised and combined in float64: this makes the output byte-identical to the submitted
        file at 6 decimals (float64 features differ only in the 6th decimal of ~6 % of the trials)."""
        f32 = np.float32
        X = self.features(np.asarray(cos, f32), np.asarray(dur_a, f32), np.asarray(dur_b, f32),
                          np.asarray(snr_a, f32), np.asarray(snr_b, f32))
        return ((X - self.mean) / self.std) @ self.coef + self.intercept
