"""Back-end scoring: cosine, mean-centering, adaptive symmetric score normalisation."""
from __future__ import annotations

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)


def cosine_scores(emb: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    e = l2n(emb)
    return np.sum(e[idx_a] * e[idx_b], axis=-1)


def center(emb: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return l2n(l2n(emb) - mean)


def asnorm(emb: np.ndarray, cohort: np.ndarray, idx_a, idx_b, top_k: int = 300,
           raw: np.ndarray | None = None) -> np.ndarray:
    """Adaptive S-norm. emb/cohort must already be in the same (normalised) space."""
    import torch

    e = torch.from_numpy(l2n(emb)).cuda() if torch.cuda.is_available() else torch.from_numpy(l2n(emb))
    c = torch.from_numpy(l2n(cohort)).to(e.device)
    mu = np.empty(len(e), dtype=np.float32)
    sd = np.empty(len(e), dtype=np.float32)
    for s in range(0, len(e), 4096):
        sc = e[s:s + 4096] @ c.T
        top = sc.topk(min(top_k, c.shape[0]), dim=1).values
        mu[s:s + 4096] = top.mean(1).cpu().numpy()
        sd[s:s + 4096] = top.std(1).cpu().numpy()
    if raw is None:
        raw = cosine_scores(emb, idx_a, idx_b)
    return 0.5 * ((raw - mu[idx_a]) / sd[idx_a] + (raw - mu[idx_b]) / sd[idx_b])


def speaker_means(emb: np.ndarray, spk: list[str]) -> np.ndarray:
    e = l2n(emb)
    spks = sorted(set(spk))
    pos = {s: i for i, s in enumerate(spks)}
    out = np.zeros((len(spks), e.shape[1]), dtype=np.float32)
    for v, s in zip(e, spk):
        out[pos[s]] += v
    return l2n(out)
