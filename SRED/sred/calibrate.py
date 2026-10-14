"""Duration-aware quality-measure-function (QMF) score calibration.

A logistic regression over [s, log dmin, log dmax, s*log dmin, s*log dmax (+ AS-norm score)]
is fitted on synthetic trials built from *train* cohort embeddings (no test labels used).
Because short-utterance pairs have compressed score distributions, a duration-dependent
re-scaling changes the pooled ranking and lowers the overall EER.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _topk_stats(e, c, k=300, spk=None, chunk=4096):
    """Top-k cohort score stats. If `spk` is given (e == c), same-speaker cohort entries are
    excluded so train-trial statistics match the test condition (unseen speakers)."""
    mu = np.empty(len(e), np.float32)
    sd = np.empty(len(e), np.float32)
    for s in range(0, len(e), chunk):
        S = e[s:s + chunk] @ c.T
        if spk is not None:
            S[spk[s:s + chunk, None] == spk[None, :]] = -9
        t = np.partition(S, -k, axis=1)[:, -k:]
        mu[s:s + chunk], sd[s:s + chunk] = t.mean(1), t.std(1)
    return mu, sd


def _features(s, d1, d2, asn=None):
    a = np.log(np.minimum(d1, d2))
    b = np.log(np.maximum(d1, d2))
    f = [s, a, b, s * a, s * b]
    if asn is not None:
        f += [asn, asn * a]
    return np.stack(f, 1)


class QMF:
    def __init__(self, use_asnorm: bool = True, top_k: int = 300, n_nontarget: int = 300000, seed: int = 0):
        self.use_asnorm, self.top_k, self.n_nontarget, self.seed = use_asnorm, top_k, n_nontarget, seed

    def fit(self, cohort: np.ndarray, spk, dur: np.ndarray):
        """cohort: normalised (and centred) train embeddings; spk: labels; dur: seconds."""
        self.cohort = cohort
        spk = np.asarray(spk)
        rng = np.random.default_rng(self.seed)
        n = len(cohort)
        I = rng.integers(0, n, self.n_nontarget)
        J = rng.integers(0, n, self.n_nontarget)
        ti, tj = [], []
        order = np.argsort(spk, kind="stable")
        for k in range(n - 1):  # all same-speaker pairs (cohort is speaker-sorted, few utts/spk)
            i = order[k]
            for m in range(k + 1, min(n, k + 10)):
                j = order[m]
                if spk[j] != spk[i]:
                    break
                ti.append(i)
                tj.append(j)
        I = np.concatenate([I, ti])
        J = np.concatenate([J, tj])
        keep = I != J
        I, J = I[keep], J[keep]
        y = (spk[I] == spk[J]).astype(int)
        s = np.sum(cohort[I] * cohort[J], 1)
        asn = None
        if self.use_asnorm:
            mu, sd = _topk_stats(cohort, cohort, self.top_k, spk=spk)
            asn = 0.5 * ((s - mu[I]) / sd[I] + (s - mu[J]) / sd[J])
        X = _features(s, dur[I], dur[J], asn)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((X - self.mu) / self.sd, y)
        return self

    def score(self, emb: np.ndarray, dur: np.ndarray, ia, ib) -> np.ndarray:
        s = np.sum(emb[ia] * emb[ib], 1)
        asn = None
        if self.use_asnorm:
            mu, sd = _topk_stats(emb, self.cohort, self.top_k)
            asn = 0.5 * ((s - mu[ia]) / sd[ia] + (s - mu[ib]) / sd[ib])
        X = _features(s, dur[ia], dur[ib], asn)
        return self.lr.decision_function((X - self.mu) / self.sd)
