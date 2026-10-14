from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER in percent."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    # linear interpolation between the two closest points
    if i + 1 < len(fpr) and (fnr[i] - fpr[i]) * (fnr[i + 1] - fpr[i + 1]) < 0:
        a = fnr[i] - fpr[i]
        b = fnr[i + 1] - fpr[i + 1]
        t = a / (a - b)
        return float(100 * (fpr[i] + t * (fpr[i + 1] - fpr[i])))
    return float(100 * (fpr[i] + fnr[i]) / 2)


def compute_min_dcf(scores, labels, p_target=0.01, c_miss=1.0, c_fa=1.0) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(dcf.min() / c_def)


def bootstrap_eer_ci(scores, labels, n=200, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(scores))
    vals = [compute_eer(scores[s], labels[s]) for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
