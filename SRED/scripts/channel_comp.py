"""Channel compensation at scoring time, learned on TRAIN only (no test labels, non-transductive).

Train cohort: 9,100 clips (5 per speaker), each embedded clean (traincohort5) and with private-like noise
(traincohort_noisy, same files). Variants, all followed by cosine + QMF(dur) fitted on the transformed cohort:
  base       E2b as submitted (QMF on the clean cohort)
  qmf-noisy  QMF fitted on clean + noisy cohort embeddings
  NAP-k      project out the top-k directions of the clean->noisy difference (nuisance attribute projection)
  WCCN       whiten by the within-speaker covariance of clean + noisy cohort embeddings (shrunk to identity)
  LDA-d      LDA on clean + noisy cohort embeddings (speakers as classes), keep d dimensions

    python scripts/channel_comp.py [--model exp/ft/W_E2b_sf2/avg.pt]
Reports EER on public_noisy and public_realnoise, plain and private-like (scripts/private_like_eer.py weights).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_like_eer import bins, gaps, weighted_eer  # noqa: E402
from sred.calibrate import QMF  # noqa: E402
from sred.data import cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.metrics import compute_eer  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def private_like_weights(pairs, y):
    fp = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel")
    fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv").set_index("rel")
    prv, _ = read_trials(ROOT / "submissions/final_private/trials_2col.txt")
    pn, ps, pg = gaps(pairs, fp)
    vn, vs, vg = gaps(prv, fv)
    en, es = np.quantile(vn, [0.2, 0.4, 0.6, 0.8]), np.quantile(vs, [0.25, 0.5, 0.75])
    bp, bv = bins(pn, ps, pg, en, es), bins(vn, vs, vg, en, es)
    nb = max(bp.max(), bv.max()) + 1
    nt = y == 0
    pv, pu = np.bincount(bv, minlength=nb) / len(bv), np.bincount(bp[nt], minlength=nb) / nt.sum()
    w = np.ones(len(y))
    w[nt] = np.minimum(np.where(pu[bp[nt]] > 0, pv[bp[nt]] / np.maximum(pu[bp[nt]], 1e-12), 0), 20)
    return w


def lda(X, lab, d):
    mu = X.mean(0)
    classes = np.unique(lab)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros_like(Sw)
    for c in classes:
        Xc = X[lab == c]
        m = Xc.mean(0)
        Sw += (Xc - m).T @ (Xc - m)
        Sb += len(Xc) * np.outer(m - mu, m - mu)
    Sw = Sw / len(X) + 1e-3 * np.eye(X.shape[1])
    ev, V = np.linalg.eigh(np.linalg.solve(Sw, Sb / len(X)) + 0)  # not symmetric in general; use generalized form below
    from scipy.linalg import eigh
    ev, V = eigh(Sb / len(X), Sw)
    return V[:, np.argsort(ev)[::-1][:d]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="exp/ft/W_E2b_sf2/avg.pt")
    ap.add_argument("--nap-only", action="store_true", help="only NAP variants, learned from synthetic / real / both noise")
    a = ap.parse_args()
    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[p] for p, _ in pairs])
    ib = np.array([pos[q] for _, q in pairs])
    dur = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel").loc[files].dur.values
    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    cdur = np.array([json.loads((ROOT / "lists/train_lengths.json").read_text())[r] / 16000 for r in cf])
    Cc = l2n(np.load(cache_key(a.model, "traincohort5", cf, mode="full")))
    Cn = l2n(np.load(cache_key(a.model, "traincohort_noisy", cf, mode="full")))
    Call, sall, dall = np.vstack([Cc, Cn]), np.concatenate([cs, cs]), np.concatenate([cdur, cdur])
    w = private_like_weights(pairs, y)
    mu = Call.mean(0)
    D = Cn - Cc
    U = np.linalg.svd(D - D.mean(0), full_matrices=False)[2]
    Sw = np.zeros((Call.shape[1],) * 2)
    for c in np.unique(sall):
        Xc = Call[sall == c]
        Sw += (Xc - Xc.mean(0)).T @ (Xc - Xc.mean(0))
    Sw /= len(Call)
    ev, V = np.linalg.eigh(0.7 * Sw + 0.3 * np.trace(Sw) / len(Sw) * np.eye(len(Sw)))
    Wc = V @ np.diag(ev ** -0.5) @ V.T
    variants = {"base": (lambda X: X, "clean"), "qmf-noisy": (lambda X: X, "both")}
    if a.nap_only:
        Cr = l2n(np.load(cache_key(a.model, "traincohort_realnoise", cf, mode="full")))
        variants = {"base": (lambda X: X, "clean")}
        for src, Dn in (("synth", Cn - Cc), ("real", Cr - Cc), ("both", np.vstack([Cn - Cc, Cr - Cc]))):
            Us = np.linalg.svd(Dn - Dn.mean(0), full_matrices=False)[2]
            for k in (5, 10, 20, 40):
                P = np.eye(len(Us)) - Us[:k].T @ Us[:k]
                variants[f"NAP-{src}-{k}"] = ((lambda P: lambda X: (X - mu) @ P)(P), "clean")
    else:
      for k in (5, 10, 20, 40):
        P = np.eye(len(U)) - U[:k].T @ U[:k]
        variants[f"NAP-{k}"] = ((lambda P: lambda X: (X - mu) @ P)(P), "both")
      variants["WCCN"] = (lambda X: (X - mu) @ Wc, "both")
      for d in (150, 120, 90):
        L = lda(Call - mu, sall, d)
        variants[f"LDA-{d}"] = ((lambda L: lambda X: (X - mu) @ L)(L), "both")
    rows = []
    for split in ["public_noisy", "public_realnoise"]:
        E = l2n(np.load(cache_key(a.model, split, files, mode="full")))
        for name, (f, coh) in variants.items():
            CE, CS, CD = (Cc, cs, cdur) if coh == "clean" else (Call, sall, dall)
            s = QMF(use_asnorm=False).fit(l2n(f(CE)), CS, CD).score(l2n(f(E)), dur, ia, ib)
            rows.append(dict(split=split, variant=name, plain=compute_eer(s, y), private_like=weighted_eer(s, y, w)))
            print(f"  {split:17s} {name:10s} plain {rows[-1]['plain']:6.3f}   private-like {rows[-1]['private_like']:6.3f}", flush=True)
    tag = "_nap" if a.nap_only else ""
    pd.DataFrame(rows).round(3).to_csv(ROOT / f"exp/analysis/channel_comp_{Path(a.model).parent.name}{tag}.csv", index=False)


if __name__ == "__main__":
    main()
