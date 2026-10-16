"""Submission with QS calibration: QMF(dur) fitted on train-cohort full utterances + their 1.5 s crops.

  python scripts/make_submission_qs.py <trials> <audio_dir> <split_name> <out_results.txt> [model_ckpt]

Trial file: 'a.wav b.wav [label]' per line; output keeps the exact pair order ('a b score').
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sred.calibrate import QMF  # noqa: E402
from sred.data import TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import extract_cached  # noqa: E402
from sred.evaluate import write_submission  # noqa: E402
from sred.metrics import compute_eer  # noqa: E402
from sred.models import count_params, load_model  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("trials")
ap.add_argument("audio_dir")
ap.add_argument("split")
ap.add_argument("out")
ap.add_argument("model", nargs="?", default="releases/version_161026_S04qs/sred_S04qs_redimnetM_161026.pt")
a = ap.parse_args()

pairs, labels = read_trials(Path(a.trials))
files = unique_files(pairs)
pos = {f: i for i, f in enumerate(files)}
ia = np.array([pos[x] for x, _ in pairs])
ib = np.array([pos[y] for _, y in pairs])
fn = lambda: load_model(a.model)  # noqa: E731
print(f"model {a.model}: {count_params(fn()) / 1e6:.3f}M params, {len(pairs)} trials, {len(files)} files")

emb = l2n(extract_cached(a.model, fn, a.split, files, Path(a.audio_dir), mode="full"))
dur = np.array([sf.info(str(Path(a.audio_dir) / f)).duration for f in files])

coh = cohort_subset(read_train_list(), per_spk=5)
rels, spk = [r for _, r in coh], [s for s, _ in coh]
cf = l2n(extract_cached(a.model, fn, "traincohort5", rels, TRAIN_DIR, mode="full"))
cs = l2n(extract_cached(a.model, fn, "traincohort5", rels, TRAIN_DIR, mode="full", max_sec=1.5))
cd = np.array([sf.info(str(TRAIN_DIR / r)).duration for r in rels])
qmf = QMF(use_asnorm=False).fit(np.r_[cf, cs], spk * 2, np.r_[cd, np.minimum(cd, 1.5)],
                                fid=np.r_[np.arange(len(rels)), np.arange(len(rels))])
scores = qmf.score(emb, dur, ia, ib)
assert np.isfinite(scores).all() and len(scores) == len(pairs)
write_submission(Path(a.out), pairs, scores)
print("wrote", a.out)
if labels is not None:
    print(f"EER {compute_eer(scores, labels):.3f}")
