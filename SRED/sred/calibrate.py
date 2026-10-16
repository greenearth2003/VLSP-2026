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


def _features(s, d1, d2, asn=None, df0=None, sd=None):
    a = np.log(np.minimum(d1, d2))
    b = np.log(np.maximum(d1, d2))
    f = [s, a, b, s * a, s * b]
    if asn is not None:
        f += [asn, asn * a]
    if df0 is not None:  # |log F0 ratio| between the two files (label-free, from audio)
        f += [df0, s * df0]
    if sd is not None:  # QMF 2.0: log speech duration (energy VAD) of the shorter / longer side
        f += [np.log(np.minimum(*sd)), np.log(np.maximum(*sd))]
    return np.stack(f, 1)


def _df0(f0, i, j, fill):
    d = np.abs(np.log(f0[i]) - np.log(f0[j]))
    return np.where(np.isfinite(d), d, fill)


class QMF:
    def __init__(self, use_asnorm: bool = True, top_k: int = 300, n_nontarget: int = 300000, seed: int = 0):
        self.use_asnorm, self.top_k, self.n_nontarget, self.seed = use_asnorm, top_k, n_nontarget, seed
        self.f0_fill = None

    def fit(self, cohort: np.ndarray, spk, dur: np.ndarray, f0: np.ndarray | None = None,
            sdur: np.ndarray | None = None, hard_neg: bool = False, fid: np.ndarray | None = None):
        """cohort: normalised (and centred) train embeddings; spk: labels; dur: seconds."""
        self.cohort = cohort
        spk = np.asarray(spk)
        rng = np.random.default_rng(self.seed)
        n = len(cohort)
        I = rng.integers(0, n, self.n_nontarget)
        J = rng.integers(0, n, self.n_nontarget)
        ti, tj = [], []
        order = np.argsort(spk, kind="stable")
        win = 10 if fid is None else 20
        for k in range(n - 1):  # all same-speaker pairs (cohort is speaker-sorted, few utts/spk)
            i = order[k]
            for m in range(k + 1, min(n, k + win)):
                j = order[m]
                if spk[j] != spk[i]:
                    break
                ti.append(i)
                tj.append(j)
        I = np.concatenate([I, ti])
        J = np.concatenate([J, tj])
        if hard_neg:  # add each cohort item's most similar other-speaker item as an extra non-target
            hi, hj = [], []
            for s0 in range(0, n, 4096):
                S = cohort[s0:s0 + 4096] @ cohort.T
                S[spk[s0:s0 + 4096, None] == spk[None, :]] = -9
                hi.append(np.arange(s0, s0 + len(S)))
                hj.append(S.argmax(1))
            I = np.concatenate([I, *hi])
            J = np.concatenate([J, *hj])
        keep = I != J
        if fid is not None:  # never pair an utterance with a crop of itself
            keep &= np.asarray(fid)[I] != np.asarray(fid)[J]
        I, J = I[keep], J[keep]
        y = (spk[I] == spk[J]).astype(int)
        s = np.sum(cohort[I] * cohort[J], 1)
        asn = None
        if self.use_asnorm:
            mu, sd = _topk_stats(cohort, cohort, self.top_k, spk=spk)
            asn = 0.5 * ((s - mu[I]) / sd[I] + (s - mu[J]) / sd[J])
        df0 = None
        if f0 is not None:
            d = np.abs(np.log(f0[I]) - np.log(f0[J]))
            self.f0_fill = float(np.nanmedian(d))
            df0 = np.where(np.isfinite(d), d, self.f0_fill)
        self.use_sdur = sdur is not None
        X = _features(s, dur[I], dur[J], asn, df0, (sdur[I], sdur[J]) if self.use_sdur else None)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((X - self.mu) / self.sd, y)
        return self

    def score(self, emb: np.ndarray, dur: np.ndarray, ia, ib, f0: np.ndarray | None = None,
              sdur: np.ndarray | None = None) -> np.ndarray:
        s = np.sum(emb[ia] * emb[ib], 1)
        asn = None
        if self.use_asnorm:
            mu, sd = _topk_stats(emb, self.cohort, self.top_k)
            asn = 0.5 * ((s - mu[ia]) / sd[ia] + (s - mu[ib]) / sd[ib])
        df0 = _df0(f0, ia, ib, self.f0_fill) if self.f0_fill is not None else None
        sd = (sdur[ia], sdur[ib]) if getattr(self, "use_sdur", False) else None
        X = _features(s, dur[ia], dur[ib], asn, df0, sd)
        return self.lr.decision_function((X - self.mu) / self.sd)
