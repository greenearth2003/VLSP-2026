"""Transductive (test-set) embedding smoothing.

Each test embedding is pulled towards the similarity-weighted mean of its k nearest neighbours among the
OTHER test files (only neighbours above a cosine threshold tau count, so files without a true neighbour
are left unchanged). Uses unlabelled test audio only -- no labels, no file paths. Effective when the test
set contains several utterances per speaker; tau protects the low-redundancy case.

    e_i' = normalise( e_i + w * sum_j s_ij * e_j / sum_j s_ij ),  j in kNN(i), s_ij >= tau
applied only to files whose cluster (itself + accepted neighbours) has >= min_cluster points.
"""
from __future__ import annotations

import numpy as np

from .scoring import l2n


def knn_smooth(E: np.ndarray, k: int = 10, w: float = 0.5, tau: float = 0.5, iters: int = 1,
               mutual: bool = False, z: float = 0.0, z_k: int = 200, min_cluster: int = 1,
               chunk: int = 4096, return_mask: bool = False):
    """mutual: j counts for i only if i is also among j's k nearest. z: additionally require
    s_ij >= mu_i + z*sd_i where mu_i/sd_i are the mean/std of i's top-z_k similarities (AS-norm style),
    i.e. the neighbour must stand out from the file's own impostor background.
    min_cluster: dynamic gating -- file i is smoothed only if it forms a cluster of >= min_cluster points
    with its accepted neighbours (itself + >= min_cluster-1 neighbours above tau); otherwise it is left
    unchanged. Re-evaluated on every pass."""
    smoothed = np.zeros(len(E), bool)
    E = l2n(E.astype(np.float32))
    for _ in range(iters):
        S_full = E @ E.T
        np.fill_diagonal(S_full, -9)
        nn_all = np.argpartition(-S_full, k, axis=1)[:, :k]
        if z > 0:
            top = -np.partition(-S_full, z_k, axis=1)[:, :z_k]
            thr_z = top.mean(1) + z * top.std(1)
        out = np.empty_like(E)
        for s in range(0, len(E), chunk):
            S = S_full[s:s + chunk]
            nn = nn_all[s:s + chunk]
            sims = np.take_along_axis(S, nn, 1)
            ok = sims >= tau
            if z > 0:
                ok &= sims >= thr_z[s:s + chunk, None]
            if mutual:
                rows = np.arange(s, s + S.shape[0])[:, None]
                ok &= (nn_all[nn] == rows[..., None]).any(-1)
            if min_cluster > 1:
                gate = (ok.sum(1) + 1) >= min_cluster
                ok &= gate[:, None]
                smoothed[s:s + S.shape[0]] |= gate
            else:
                smoothed[s:s + S.shape[0]] |= ok.any(1)
            wts = np.where(ok, sims, 0.0)
            nb = (E[nn] * wts[..., None]).sum(1)
            denom = wts.sum(1, keepdims=True)
            nb = np.where(denom > 0, nb / np.maximum(denom, 1e-6), 0.0)
            out[s:s + chunk] = E[s:s + chunk] + w * nb
        E = l2n(out)
    return (E, smoothed) if return_mask else E
