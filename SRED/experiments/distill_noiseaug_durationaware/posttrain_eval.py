"""Post-training study on silver from the cached pass of posttrain_embed.py (CPU only, seconds per variant).

    .venv/bin/python experiments/distill_noiseaug_durationaware/posttrain_eval.py --tag E4_step6000

Every calibrated variant is a logistic regression (our QMF2, sred/calibrate.py) fitted ONLY on train-cohort trials:
all same-speaker pairs of DIFFERENT files + 300k random pairs. Pairs of a file with its own noisy copy or own crop
are excluded (they are trivially easy targets).

Variants (one change at a time against the baseline, then the useful ones combined):
  cos             plain cosine of the whole-clip embedding
  qmf2            QMF2 on speech duration + SNR, cohort = whole clips, clean + noisy   (current best candidate)
  +short          cohort also contains 0.5-2 s speech-dense crops (clean + noisy): short trials in the fitting set
  +norm           raw embedding norm as a third quality measure
  +speech         embedding = normalised mean of the whole-clip and speech-only embeddings (cohort built the same way)
  +center         subtract the cohort mean embedding before cosine
  +asnorm         adaptive s-norm (top 300 cohort scores) as an extra score input
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

import evaluate as ev  # noqa: E402
from calibrate_eval import metrics  # noqa: E402
from sred.calibrate import QMF2, _topk_stats  # noqa: E402


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


class QMF2Files(QMF2):
    """QMF2 whose cohort trials never pair two versions of the same file."""

    def fit_files(self, cohort, spk, fid, quality, extra_fn=None):
        rng = np.random.default_rng(self.seed)
        n = len(spk)
        I, J = rng.integers(0, n, self.n_nontarget), rng.integers(0, n, self.n_nontarget)
        by = {}
        for i, s in enumerate(spk):
            by.setdefault(s, []).append(i)
        ti, tj = [], []
        for idx in by.values():
            for i, j in itertools.combinations(idx, 2):
                ti.append(i)
                tj.append(j)
        I, J = np.concatenate([I, ti]), np.concatenate([J, tj])
        keep = fid[I] != fid[J]
        I, J = I[keep], J[keep]
        y = (spk[I] == spk[J]).astype(int)
        s = np.sum(cohort[I] * cohort[J], 1)
        extra = extra_fn(I, J, s) if extra_fn is not None else None
        X = self._feats(s, {k: v[I] for k, v in quality.items()}, {k: v[J] for k, v in quality.items()}, extra)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((X - self.mu) / self.sd, y)
        self.n_trials, self.n_targets = len(y), int(y.sum())
        return self


def build(D, short=False, speech=False):
    """Cohort (emb, spk, file id, qualities) and silver (emb, qualities) for one variant."""
    spk0, n = D["coh_spk"], len(D["coh_spk"])
    E, S, F, Q = [], [], [], []
    for v in ("clean", "noisy"):
        w, sp = D[f"coh_{v}_whole"], D[f"coh_{v}_speech"]
        E.append(l2n(l2n(w) + l2n(sp)) if speech else l2n(w))
        Q.append(np.c_[D[f"q_coh_{v}_whole"], np.linalg.norm(w, axis=1)])
        S.append(spk0)
        F.append(np.arange(n))
        if short:
            e = D[f"coh_{v}_short"]
            ok = ~np.isnan(e).any(1)
            E.append(l2n(e[ok]))
            Q.append(np.c_[D[f"q_coh_{v}_short"][ok], np.linalg.norm(e[ok], axis=1)])
            S.append(spk0[ok])
            F.append(np.arange(n)[ok])
    ce, cq, cs, cf = np.concatenate(E), np.concatenate(Q), np.concatenate(S), np.concatenate(F)
    w, sp = D["silver_whole"], D["silver_speech"]
    se = l2n(l2n(w) + l2n(sp)) if speech else l2n(w)
    sq = np.c_[D["q_silver_whole"], np.linalg.norm(w, axis=1)]
    return ce, cs, cf, cq, se, sq


def run(D, tr, ia, ib, y, w, cl, short=False, speech=False, norm=False, center=False, asnorm=False, dup_ok=False):
    ce, cs, cf, cq, se, sq = build(D, short, speech)
    if center:
        mu = ce.mean(0)
        ce, se = l2n(ce - mu), l2n(se - mu)
    names = ["speech", "snr"] + (["norm"] if norm else [])
    qd = lambda q: {"speech": q[:, 0], "snr": q[:, 1], "norm": q[:, 2]}  # noqa: E731
    extra_fn = extra = None
    if asnorm:
        cmu, csd = _topk_stats(ce, ce, 300, spk=cs)
        extra_fn = lambda I, J, s: 0.5 * ((s - cmu[I]) / csd[I] + (s - cmu[J]) / csd[J])  # noqa: E731
        smu, ssd = _topk_stats(se, ce, 300)
        s0 = np.sum(se[ia] * se[ib], 1)
        extra = 0.5 * ((s0 - smu[ia]) / ssd[ia] + (s0 - smu[ib]) / ssd[ib])
    m = QMF2Files(names)
    fid = np.arange(len(cf)) if dup_ok else cf  # dup_ok: every row its own "file" = calibrate_eval.py's trials
    m.fit_files(ce, cs, fid, {k: v for k, v in qd(cq).items() if k in names}, extra_fn)
    s = m.score(se, {k: v for k, v in qd(sq).items() if k in names}, ia, ib, extra)
    r = metrics(s, tr, y, w, cl)
    r["fit_trials"], r["fit_targets"] = m.n_trials, m.n_targets
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    D = dict(np.load(EXP / "runs/_calib" / a.tag / "posttrain.npz"))
    tr = pd.read_csv(ev.DEV / "trials.csv")
    y = tr.label.to_numpy()
    sw = np.load(ev.DEV / "silver_weights.npz")
    w, cl = sw["private_like"], sw["in_cleaned"]
    pos = {f: i for i, f in enumerate(D["silver_files"])}
    ia, ib = np.array([pos[f] for f in tr.a]), np.array([pos[f] for f in tr.b])
    rows = {}
    se = l2n(D["silver_whole"])
    rows["cos"] = metrics(np.sum(se[ia] * se[ib], 1), tr, y, w, cl)
    rows["qmf2 (as before, same-file pairs kept)"] = run(D, tr, ia, ib, y, w, cl, dup_ok=True)
    rows["qmf2"] = run(D, tr, ia, ib, y, w, cl)
    for k in ("short", "norm", "speech", "center", "asnorm"):
        rows[f"qmf2 +{k}"] = run(D, tr, ia, ib, y, w, cl, **{k: True})
    base = rows["qmf2"]["silver_all"]
    good = [k for k in ("short", "norm", "speech", "center", "asnorm") if rows[f"qmf2 +{k}"]["silver_all"] < base]
    for r in range(2, len(good) + 1):
        for combo in itertools.combinations(good, r):
            rows["qmf2 +" + " +".join(combo)] = run(D, tr, ia, ib, y, w, cl, **{k: True for k in combo})
    (ev.OUT / f"posttrain__{a.tag}.json").write_text(json.dumps(rows, indent=1))
    cols = ["silver_all", "silver_cleaned", "silver_private_like", "silver_minDCF", "natural <0.75s", "natural 0.75-1s",
            "natural 1-2s", "natural >3s"]
    print(f"{a.tag}: silver EER % (best system S_C2X_X8 fv + Q3: 5.413 / 3.933 / 5.709)")
    print(f"  {'variant':42s}" + "".join(f"{c.replace('silver_', '').replace('natural ', ''):>11s}" for c in cols))
    for k, r in rows.items():
        print(f"  {k:42s}" + "".join(f"{r[c]:11.3f}" for c in cols))


if __name__ == "__main__":
    main()
