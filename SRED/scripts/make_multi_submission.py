"""Private zip with multi-speaker blended scoring on top of E2b (+ optional transductive smoothing).

    python scripts/make_multi_submission.py --lam 0.25 [--transductive] --zip result-private-E2b-ms-t0.zip

score = QMF_dur( cos(full_a, full_b) [smoothed if --transductive] + lam * max(0, best speaker-pair cosine - cos(full)) )
Speaker sets (whole-clip embedding + 2 cluster centroids for two-voice clips) come from scripts/multi_speaker_score.py's
cache exp/emb/<model>/private_speakers.npz. Same format checks as scripts/make_final_submission.sh.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF2  # noqa: E402
from sred.data import TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key, extract_cached  # noqa: E402
from sred.evaluate import write_submission  # noqa: E402
from sred.models import load_model  # noqa: E402
from sred.scoring import l2n  # noqa: E402
from sred.transductive import knn_smooth  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="exp/ft/W_E2b_sf2/avg.pt")
    ap.add_argument("--trials", default="submissions/final_private/trials_2col.txt")
    ap.add_argument("--audio-root", default="data/Vi-SRED26/private_test/data")
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--transductive", action="store_true")
    ap.add_argument("--zip", required=True)
    a = ap.parse_args()
    out = ROOT / "submissions/final_private"
    pairs, _ = read_trials(ROOT / a.trials)
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs])
    ib = np.array([pos[x] for _, x in pairs])
    z = np.load(cache_key(a.model, "private", files, mode="full").with_name("private_speakers.npz"))
    full, c0, c1, two = l2n(z["full"]), l2n(z["c0"]), l2n(z["c1"]), z["two"]
    s_full = (full[ia] * full[ib]).sum(1)
    gain = np.maximum(0, np.max([(A[ia] * B[ib]).sum(1) for A in (full, c0, c1) for B in (full, c0, c1)], 0) - s_full)
    print(f"private: two-voice clips {two.mean():.1%}; trials touching one {np.mean(two[ia] | two[ib]):.1%}; "
          f"trials with a positive pair gain {np.mean(gain > 0):.1%}")
    E = full
    if a.transductive:
        E, mask = knn_smooth(full, 20, 0.5, 0.6, 2, min_cluster=3, return_mask=True)
        print(f"transductive smoothing ON: {mask.mean():.1%} of files smoothed")
    s_base = (E[ia] * E[ib]).sum(1)
    mf = lambda: load_model(a.model)  # noqa: E731
    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    C = l2n(extract_cached(a.model, mf, "traincohort5", cf, TRAIN_DIR, mode="full"))
    dcache = ROOT / "lists/private_durations.npy"
    dur = np.load(dcache) if dcache.exists() and len(np.load(dcache)) == len(files) else \
        np.array([sf.info(str(ROOT / a.audio_root / f)).duration for f in files])
    np.save(dcache, dur)
    cdur = np.array([sf.info(str(TRAIN_DIR / f)).duration for f in cf])
    q = QMF2(["dur"]).fit(C, cs, {"dur": cdur})
    # QMF2.score computes the cosine itself; build the same features from the adjusted score instead
    X = q._feats(s_base + a.lam * gain, {"dur": dur[ia]}, {"dur": dur[ib]})
    s = q.lr.decision_function((X - q.mu) / q.sd)
    res = out / "results.txt"
    write_submission(res, pairs, s)
    (out / f"results_{Path(a.zip).stem}.txt").write_text(res.read_text())
    n = len(pairs)
    lines = res.read_text().splitlines()
    bad = sum(1 for l in lines if len(l.split()) != 3 or not np.isfinite(float(l.split()[2])))
    order = sum(1 for l, (x, y_) in zip(lines, pairs) if l.split()[:2] != [x, y_])
    print(f"lines: {len(lines)} / {n}   mismatched pairs: {order}   bad score lines: {bad}")
    assert len(lines) == n and order == 0 and bad == 0, "checks failed"
    (out / a.zip).unlink(missing_ok=True)
    subprocess.run(["zip", "-q", "-j", str(out / a.zip), str(res)], check=True)
    print("ready:", out / a.zip)


if __name__ == "__main__":
    main()
