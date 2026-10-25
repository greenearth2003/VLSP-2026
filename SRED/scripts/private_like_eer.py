"""Private-like re-weighting of a public stand-in: non-target trials are re-weighted so that their channel and gender
make-up (noise-floor gap, SNR gap, same-gender share, measured on the ORIGINAL public audio) matches the private
trial list's; targets keep weight 1. Private's trials look like hard negatives sharing a channel
(exp/analysis/private_gap_trials.txt), which the plain stand-ins under-represent.

    python scripts/private_like_eer.py MODEL [MODEL ...] [--splits public_noisy public_realnoise]

Prints plain and re-weighted EER (QMF dur) per model and split. Weights depend only on audio features, never on scores.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF  # noqa: E402
from sred.data import cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.metrics import compute_eer  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def gaps(pairs, f):
    a = f.loc[[p for p, _ in pairs]]
    b = f.loc[[q for _, q in pairs]]
    male = lambda v: v < 165  # noqa: E731
    return (np.abs(a.noise_db.values - b.noise_db.values), np.abs(a.snr_db.values - b.snr_db.values),
            (male(a.f0.values) == male(b.f0.values)).astype(int))


def bins(noise_gap, snr_gap, same_g, edges_n, edges_s):
    return (np.digitize(noise_gap, edges_n) * (len(edges_s) + 1) + np.digitize(snr_gap, edges_s)) * 2 + same_g


def weighted_eer(s, y, w):
    o = np.argsort(s)
    s, y, w = s[o], y[o], w[o]
    wt, wn = (w * y).sum(), (w * (1 - y)).sum()
    frr = np.cumsum(w * y) / wt
    far = 1 - np.cumsum(w * (1 - y)) / wn
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["public_noisy", "public_realnoise"])
    a = ap.parse_args()
    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[p] for p, _ in pairs])
    ib = np.array([pos[q] for _, q in pairs])
    fp = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel")
    fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv").set_index("rel")
    prv, _ = read_trials(ROOT / "submissions/final_private/trials_2col.txt")
    pn, ps, pg = gaps(pairs, fp)
    vn, vs, vg = gaps(prv, fv)
    edges_n = np.quantile(vn, [0.2, 0.4, 0.6, 0.8])
    edges_s = np.quantile(vs, [0.25, 0.5, 0.75])
    bp, bv = bins(pn, ps, pg, edges_n, edges_s), bins(vn, vs, vg, edges_n, edges_s)
    nb = bins(np.array([1e9]), np.array([1e9]), np.array([1]), edges_n, edges_s)[0] + 1
    p_priv = np.bincount(bv, minlength=nb) / len(bv)
    nt = y == 0
    p_pub = np.bincount(bp[nt], minlength=nb) / nt.sum()
    w = np.ones(len(y))
    w[nt] = np.where(p_pub[bp[nt]] > 0, p_priv[bp[nt]] / np.maximum(p_pub[bp[nt]], 1e-12), 0)
    w[nt] = np.minimum(w[nt], 20)
    ess = w[nt].sum() ** 2 / (w[nt] ** 2).sum()
    print(f"re-weighted non-targets: effective sample size {ess:.0f} of {nt.sum()}; "
          f"noise-floor gap median {np.median(pn[nt]):.1f} -> private {np.median(vn):.1f} dB, same gender {pg[nt].mean():.2f} -> {vg.mean():.2f}")
    dur = fp.loc[files].dur.values
    ci = cohort_subset(read_train_list(), per_spk=5)
    cdur = np.array([json.loads((ROOT / "lists/train_lengths.json").read_text())[r] / 16000 for _, r in ci])
    for m in a.models:
        coh = l2n(np.load(cache_key(m, "traincohort5", [r for _, r in ci], mode="full")))
        q = QMF(use_asnorm=False).fit(coh, [c for c, _ in ci], cdur)
        for split in a.splits:
            if not cache_key(m, split, files, mode="full").exists():
                print(f"  {Path(m).parent.name:20s} {split:17s} (no embeddings)")
                continue
            s = q.score(l2n(np.load(cache_key(m, split, files, mode="full"))), dur, ia, ib)
            print(f"  {Path(m).parent.name:20s} {split:17s} plain {compute_eer(s, y):6.3f}   private-like {weighted_eer(s, y, w):6.3f}")


if __name__ == "__main__":
    main()
