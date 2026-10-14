"""Trainable linear back-ends on frozen embeddings: whitening, LDA, WCCN, two-covariance PLDA.

All are fitted on in-domain (VLSP train) embeddings with speaker labels.
They add only a DxD matrix (<0.04M params for D=192) at inference.
"""
from __future__ import annotations

import numpy as np

from .scoring import l2n


def _within_between(x: np.ndarray, spk: list[str]):
    spk = np.asarray(spk)
    mu = x.mean(0)
    sw = np.zeros((x.shape[1], x.shape[1]))
    sb = np.zeros_like(sw)
    for s in np.unique(spk):
        xs = x[spk == s]
        m = xs.mean(0)
        d = xs - m
        sw += d.T @ d
        sb += len(xs) * np.outer(m - mu, m - mu)
    return sw / len(x), sb / len(x), mu


class LinearBackend:
    """x -> l2n(W^T (l2n(x) - mean)); score = cosine."""

    def __init__(self, kind: str = "lda", dim: int | None = None, reg: float = 1e-3):
        self.kind, self.dim, self.reg = kind, dim, reg

    def fit(self, x: np.ndarray, spk: list[str]):
        x = l2n(x)
        self.mean = x.mean(0)
        x = x - self.mean
        D = x.shape[1]
        sw, sb, _ = _within_between(x, spk)
        sw += self.reg * np.eye(D)
        if self.kind == "whiten":
            cov = np.cov(x.T) + self.reg * np.eye(D)
            e, v = np.linalg.eigh(cov)
            self.W = v / np.sqrt(e)
        elif self.kind == "wccn":
            self.W = np.linalg.cholesky(np.linalg.inv(sw))
        elif self.kind in ("lda", "lda+wccn"):
            # generalised eigenproblem Sb v = l Sw v
            e_w, v_w = np.linalg.eigh(sw)
            wh = v_w / np.sqrt(e_w)
            e_b, v_b = np.linalg.eigh(wh.T @ sb @ wh)
            order = np.argsort(e_b)[::-1][: (self.dim or D)]
            self.W = wh @ v_b[:, order]
            if self.kind == "lda+wccn":
                y = x @ self.W
                sw2, _, _ = _within_between(y, spk)
                self.W = self.W @ np.linalg.cholesky(np.linalg.inv(sw2 + self.reg * np.eye(y.shape[1])))
        else:
            raise ValueError(self.kind)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return l2n((l2n(x) - self.mean) @ self.W)


class PLDA:
    """Two-covariance PLDA after centering + LDA(optional) + length norm."""

    def __init__(self, lda_dim: int | None = 150, n_iter: int = 10):
        self.lda_dim, self.n_iter = lda_dim, n_iter

    def fit(self, x, spk):
        self.pre = LinearBackend("lda", self.lda_dim).fit(x, spk)
        y = self.pre.transform(x) * np.sqrt(self.pre.W.shape[1])
        self.mu = y.mean(0)
        y = y - self.mu
        sw, sb, _ = _within_between(y, spk)
        self.B = np.linalg.inv(sb + 1e-6 * np.eye(len(sb)))  # between precision
        self.Wp = np.linalg.inv(sw + 1e-6 * np.eye(len(sw)))  # within precision
        self._prep(sb, sw)
        return self

    def _prep(self, sb, sw):
        # log-likelihood ratio for two single enrolment vectors (closed form)
        tot = sb + sw
        d = len(sb)
        inv_tot = np.linalg.inv(tot)
        big = np.block([[tot, sb], [sb, tot]])
        inv_big = np.linalg.inv(big)
        self.Q = inv_tot - inv_big[:d, :d]
        self.P = inv_big[:d, d:]
        self.P = -self.P

    def transform(self, x):
        return self.pre.transform(x) * np.sqrt(self.pre.W.shape[1]) - self.mu

    def score(self, ya, yb):
        return (0.5 * np.sum((ya @ self.Q) * ya, 1) + 0.5 * np.sum((yb @ self.Q) * yb, 1)
                + np.sum((ya @ self.P) * yb, 1))
