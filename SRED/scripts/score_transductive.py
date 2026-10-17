"""Score a trial list with the final model + transductive kNN smoothing of the test embeddings + QMF.

    python scripts/score_transductive.py --trials T.txt --audio-root DIR --split NAME --out results.txt \
        [--model exp/ft/W_K21_M/avg.pt] [--k 20 --w 0.5 --tau 0.6 --iters 2]

Prints a redundancy diagnostic first: the share of test files that have >= 1 / >= 3 other files above
`tau` cosine. On the public test this is high (many utterances per speaker); if the private set shows a
much lower value, smoothing is expected to HURT and should not be used (see EXPERIMENTS.md, Round 3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF2
from sred.data import TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files
from sred.embed import extract_cached
from sred.evaluate import write_submission
from sred.metrics import compute_eer
from sred.models import load_model
from sred.scoring import l2n
from sred.transductive import knn_smooth

ROOT = Path(__file__).resolve().parents[1]


def redundancy(E, tau, chunk=4096):
    n1 = n3 = 0
    for s in range(0, len(E), chunk):
        S = E[s:s + chunk] @ E.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -9
        c = (S >= tau).sum(1)
        n1 += int((c >= 1).sum())
        n3 += int((c >= 3).sum())
    return n1 / len(E), n3 / len(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="exp/ft/W_K21_M/avg.pt")
    ap.add_argument("--trials", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--no-smooth", action="store_true")
    a = ap.parse_args()

    pairs, labels = read_trials(Path(a.trials))
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs])
    ib = np.array([pos[x] for _, x in pairs])
    mf = lambda: load_model(a.model)  # noqa: E731
    E = l2n(extract_cached(a.model, mf, a.split, files, Path(a.audio_root), mode="full"))
    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    C = l2n(extract_cached(a.model, mf, "traincohort5", cf, TRAIN_DIR, mode="full"))
    dur = np.array([sf.info(str(Path(a.audio_root) / f)).duration for f in files])
    cdur = np.array([sf.info(str(TRAIN_DIR / f)).duration for f in cf])

    r1, r3 = redundancy(E, a.tau)
    print(f"redundancy @tau={a.tau}: {r1:.1%} of files have >=1 neighbour, {r3:.1%} have >=3  "
          f"(public test @0.6: 97.5% / 94.2%; stress sets with <=3 / <=2 files per speaker: 87% / 3% and 75% / 0.2% -> use the >=3 share: below ~50% do NOT smooth)")
    if not a.no_smooth:
        E = knn_smooth(E, a.k, a.w, a.tau, a.iters)
    s = QMF2(["dur"]).fit(C, cs, {"dur": cdur}).score(E, {"dur": dur}, ia, ib)
    write_submission(Path(a.out), pairs, s)
    print(f"wrote {a.out}  ({len(pairs)} trials; smoothing={'off' if a.no_smooth else f'k={a.k} w={a.w} tau={a.tau} iters={a.iters}'})")
    if labels is not None:
        print(f"EER on this list: {compute_eer(s, labels):.3f}")


if __name__ == "__main__":
    main()
