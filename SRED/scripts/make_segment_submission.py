"""Segment-gated energy-trim scoring (non-transductive), for a labelled stand-in (prints EER) or the private test (zip).

    python scripts/make_segment_submission.py --split public_rumble_v2 [--lam 0.25]
    python scripts/make_segment_submission.py --split private --lam 0.25 --zip result-private-S3-seg-ms025-t0.zip

Per trial, from label-free measurements of the two clips (exp/analysis/features_<split>.csv, energy based, no VAD model):
  gate = a clip is < 50% speech, or the shorter clip has < 1 s of speech, or the clips' levels differ by > 10 dB
  inside the gate  -> embedding = L2-normalised sum of the whole-clip and the energy-trimmed embeddings
                      (silence/noise-only stretches weigh less; sred.evaluate --mode vad caches the trimmed ones)
  outside the gate -> whole-clip embedding (unchanged baseline)
Each branch: raw cosine (+ lam * multi-speaker gain, as scripts/make_multi_submission.py) -> duration QMF fitted on
train-cohort trials embedded the same way (no test labels). Outputs are log-odds on one scale.
Selected on three noisy stand-ins (scripts/segment_fix.py): better on every set, all trials and cleaned list.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF2  # noqa: E402
from sred.data import cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.evaluate import write_submission  # noqa: E402
from sred.metrics import compute_eer, compute_min_dcf  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VAD = dict(mode="vad", crop_sec=3.0, hop_sec=1.5)


def qmf_feats(s, d1, d2):
    lo, hi = np.log(np.maximum(np.minimum(d1, d2), 0.05)), np.log(np.maximum(np.maximum(d1, d2), 0.05))
    return np.stack([s, lo, hi, s * lo, s * hi], 1)


def fit_qmf(C, spk, dur):
    I, J, y = QMF2(["dur"]).trials(spk)
    X = qmf_feats((C[I] * C[J]).sum(1), dur[I], dur[J])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((X - mu) / sd, y)
    return lambda s, d1, d2: lr.decision_function((qmf_feats(s, d1, d2) - mu) / sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="exp/ft/S3_E2b_X2/avg.pt")
    ap.add_argument("--split", default="public_rumble_v2", help="a labelled stand-in under data/, or 'private'")
    ap.add_argument("--lam", type=float, default=0.0, help="multi-speaker gain weight (0 = off)")
    ap.add_argument("--zip", default=None)
    ap.add_argument("--save", default=None, help="stand-ins: also write the gated scores here (fileA fileB score)")
    a = ap.parse_args()

    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
    cdur = np.array([lengths[r] / 16000 for r in cf])
    Cf = l2n(np.load(cache_key(a.model, "traincohort5", cf, mode="full")))
    q_full = fit_qmf(Cf, cs, cdur)
    q_fv = fit_qmf(l2n(Cf + l2n(np.load(cache_key(a.model, "traincohort5", cf, **VAD)))), cs, cdur)

    private = a.split == "private"
    pairs, y = read_trials(ROOT / "submissions/final_private/trials_2col.txt") if private else read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[p] for p, _ in pairs]); ib = np.array([pos[q] for _, q in pairs])
    feat = pd.read_csv(ROOT / f"exp/analysis/features_{a.split}.csv").set_index("rel").loc[files]
    dur, sp, lv = (feat[c].to_numpy() for c in ("dur", "speech_dur", "rms_db"))
    fk = cache_key(a.model, a.split, files, mode="full")
    full = l2n(np.load(fk))
    fv = l2n(full + l2n(np.load(cache_key(a.model, a.split, files, **VAD))))
    sh = feat.speech_ratio.to_numpy()

    gain = np.zeros(len(pairs))
    if a.lam > 0:
        z = np.load(fk.with_name(f"{a.split}_speakers.npz"))
        F_, c0, c1 = l2n(z["full"]), l2n(z["c0"]), l2n(z["c1"])
        s_f = (F_[ia] * F_[ib]).sum(1)
        gain = np.maximum(0, np.max([(A[ia] * B[ib]).sum(1) for A in (F_, c0, c1) for B in (F_, c0, c1)], 0) - s_f)

    gate = (np.minimum(sh[ia], sh[ib]) < 0.5) | (np.minimum(sp[ia], sp[ib]) < 1.0) | (np.abs(lv[ia] - lv[ib]) > 10)
    d1, d2 = dur[ia], dur[ib]
    s_base = q_full((full[ia] * full[ib]).sum(1) + a.lam * gain, d1, d2)
    s = np.where(gate, q_fv((fv[ia] * fv[ib]).sum(1) + a.lam * gain, d1, d2), s_base)
    print(f"{a.split}: {len(pairs):,} trials; gated (whole+trimmed embedding) {gate.mean():.1%}, unchanged {(~gate).mean():.1%}; lam {a.lam}")
    if not private:
        meta = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")
        cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
        if a.save:
            write_submission(ROOT / a.save, pairs, s)
        for name, v in (("baseline", s_base), ("segment-gated", s)):
            print(f"  {name:14s} EER {compute_eer(v, y):.3f}  minDCF {compute_min_dcf(v, y):.4f} | cleaned list EER "
                  f"{compute_eer(v[cl], y[cl]):.3f}  minDCF {compute_min_dcf(v[cl], y[cl]):.4f}")
        return
    out = ROOT / "submissions/final_private"
    res = out / "results.txt"
    write_submission(res, pairs, s)
    (out / f"results_{Path(a.zip).stem}.txt").write_text(res.read_text())
    lines = res.read_text().splitlines()
    bad = sum(1 for l in lines if len(l.split()) != 3 or not np.isfinite(float(l.split()[2])))
    order = sum(1 for l, (x, y_) in zip(lines, pairs) if l.split()[:2] != [x, y_])
    print(f"lines: {len(lines)} / {len(pairs)}   mismatched pairs: {order}   bad score lines: {bad}")
    assert len(lines) == len(pairs) and order == 0 and bad == 0, "checks failed"
    (out / a.zip).unlink(missing_ok=True)
    subprocess.run(["zip", "-q", "-j", str(out / a.zip), str(res)], check=True)
    print("ready:", out / a.zip)


if __name__ == "__main__":
    main()
