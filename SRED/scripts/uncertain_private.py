"""Uncertain private trials: which trials sit near the decision threshold of the submitted system, what they have in
common, and which files keep appearing there. No labels; analysis only, nothing here feeds the submission.

    python scripts/uncertain_private.py [--scores submissions/final_private/results_result-private-S3-ms025-t0.txt]
                                        [--model exp/ft/S3_E2b_X2/avg.pt] [--band 0.05]

Threshold = the model's EER threshold on the synthetic-noise stand-in (QMF dur, same scale as the private scores).
Uncertain band = the `band` share of private trials closest to it. Compares band vs all trials on per-trial audio
features (exp/analysis/features_private.csv) and the two-voice flag (exp/emb/<model>/private_speakers.npz).
Writes exp/analysis/uncertain_private_{trials,files}.csv.
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
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def eer_threshold(model):
    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[a] for a, _ in pairs])
    ib = np.array([pos[b] for _, b in pairs])
    dur = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel").loc[files].dur.values
    ci = cohort_subset(read_train_list(), per_spk=5)
    coh = l2n(np.load(cache_key(model, "traincohort5", [r for _, r in ci], mode="full")))
    cdur = np.array([json.loads((ROOT / "lists/train_lengths.json").read_text())[r] / 16000 for _, r in ci])
    s = QMF(use_asnorm=False).fit(coh, [c for c, _ in ci], cdur).score(
        l2n(np.load(cache_key(model, "public_noisy", files, mode="full"))), dur, ia, ib)
    o = np.argsort(s)
    frr = np.cumsum(y[o]) / y.sum()
    far = 1 - np.cumsum(1 - y[o]) / (1 - y).sum()
    return float(s[o][np.argmin(np.abs(frr - far))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="submissions/final_private/results_result-private-S3-ms025-t0.txt")
    ap.add_argument("--model", default="exp/ft/S3_E2b_X2/avg.pt")
    ap.add_argument("--band", type=float, default=0.05)
    a = ap.parse_args()
    rows = [l.split() for l in open(ROOT / a.scores)]
    t = pd.DataFrame(rows, columns=["a", "b", "score"]).astype({"score": float})
    thr = eer_threshold(a.model)
    t["margin"] = t.score - thr
    k = int(a.band * len(t))
    band_idx = np.argsort(np.abs(t.margin.values))[:k]
    t["uncertain"] = False
    t.loc[band_idx, "uncertain"] = True
    f = pd.read_csv(ROOT / "exp/analysis/features_private.csv").set_index("rel")
    files = unique_files(list(zip(t.a, t.b)))
    z = np.load(cache_key(a.model, "private", files, mode="full").with_name("private_speakers.npz"))
    two = pd.Series(z["two"], index=files)
    A, B = f.loc[t.a], f.loc[t.b]
    male = lambda v: v < 165  # noqa: E731
    t["min_speech_s"] = np.minimum(A.speech_dur.values, B.speech_dur.values)
    t["min_snr_db"] = np.minimum(A.snr_db.values, B.snr_db.values)
    t["min_level_db"] = np.minimum(A.rms_db.values, B.rms_db.values)
    t["noise_gap_db"] = np.abs(A.noise_db.values - B.noise_db.values)
    t["same_gender"] = (male(A.f0.values) == male(B.f0.values)).astype(float)
    t["two_voice"] = (two.loc[t.a].values | two.loc[t.b].values).astype(float)
    print(f"threshold {thr:.3f} (EER point on the synthetic-noise stand-in); band = {k} trials closest to it "
          f"(|margin| < {np.abs(t.margin.values)[band_idx].max():.2f})")
    print(f"share of private trials above threshold (judged same speaker): {(t.margin > 0).mean():.1%}\n")
    cols = ["min_speech_s", "min_snr_db", "min_level_db", "noise_gap_db", "same_gender", "two_voice"]
    cmp = pd.DataFrame({"uncertain band": t.loc[t.uncertain, cols].median(), "all trials": t[cols].median()})
    for c in ["same_gender", "two_voice"]:
        cmp.loc[c] = [t.loc[t.uncertain, c].mean(), t[c].mean()]
    cmp["lift"] = cmp["uncertain band"] / cmp["all trials"].replace(0, np.nan)
    print("medians (shares for same_gender, two_voice):")
    print(cmp.round(3).to_string(), "\n")
    # conditions: how much more likely a trial is uncertain in each bin
    for c, edges in (("min_speech_s", [0, 1, 1.5, 2, 3, 5, 99]), ("min_snr_db", [-99, 5, 10, 15, 20, 99]),
                     ("min_level_db", [-99, -40, -35, -30, -25, 0])):
        g = t.groupby(pd.cut(t[c], edges), observed=True).uncertain.agg(["mean", "size"])
        g["mean"] = g["mean"] / a.band
        print(f"uncertain rate vs average, by {c}:\n{g.rename(columns={'mean': 'x average', 'size': 'trials'}).round(2).to_string()}\n")
    # problem files: over-represented in the band
    fa = pd.concat([t[["a", "uncertain"]].rename(columns={"a": "file"}), t[["b", "uncertain"]].rename(columns={"b": "file"})])
    pf = fa.groupby("file").uncertain.agg(["sum", "size"])
    pf["rate"] = pf["sum"] / pf["size"]
    pf = pf[pf["size"] >= 10].sort_values(["rate", "sum"], ascending=False)
    pf = pf.join(f[["dur", "speech_dur", "snr_db", "rms_db"]]).join(two.rename("two_voice"))
    print("files most often in the band (>= 10 trials):")
    print(pf.head(15).round(2).to_string())
    t.loc[t.uncertain].sort_values("margin").to_csv(ROOT / "exp/analysis/uncertain_private_trials.csv", index=False)
    pf.to_csv(ROOT / "exp/analysis/uncertain_private_files.csv")


if __name__ == "__main__":
    main()
