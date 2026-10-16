"""Evaluate a speaker model on a trial list with several scoring back-ends.

Example:
  python -m sred.evaluate --model redimnet_M_mix --mode full
  python -m sred.evaluate --model exp/ft/xxx/best.pt --submission out.txt
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import PUBLIC_DIR, PUBLIC_TRIALS, TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files
from .embed import extract_cached
from .metrics import compute_eer, compute_min_dcf
from .models import count_params, load_model
from .calibrate import QMF
from .scoring import asnorm, center, cosine_scores, l2n

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "exp/results.jsonl"


def write_submission(path: Path, pairs, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for (a, b), s in zip(pairs, scores):
            assert np.isfinite(s)
            f.write(f"{a} {b} {s:.6f}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trials", default=str(PUBLIC_TRIALS))
    ap.add_argument("--audio-root", default=str(PUBLIC_DIR))
    ap.add_argument("--split", default="public")
    ap.add_argument("--mode", default="full", choices=["full", "crop", "full+crop", "vad", "dom"])
    ap.add_argument("--crop-sec", type=float, default=3.0)
    ap.add_argument("--hop-sec", type=float, default=1.5)
    ap.add_argument("--cohort-per-spk", type=int, default=5)
    ap.add_argument("--topk", type=int, nargs="+", default=[100, 300, 1000])
    ap.add_argument("--submission", default=None, help="write scores of the best backend here")
    ap.add_argument("--submission-backend", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    pairs, labels = read_trials(Path(args.trials))
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[a] for a, _ in pairs])
    ib = np.array([pos[b] for _, b in pairs])

    kw = dict(mode=args.mode)
    if args.mode != "full":
        kw.update(crop_sec=args.crop_sec, hop_sec=args.hop_sec)
    model_fn = lambda: load_model(args.model)  # noqa: E731
    n_params = count_params(load_model(args.model))

    t0 = time.time()
    emb = extract_cached(args.model, model_fn, args.split, files, Path(args.audio_root), **kw)
    t_emb = time.time() - t0
    cohort_items = cohort_subset(read_train_list(), per_spk=args.cohort_per_spk)
    cohort = extract_cached(args.model, model_fn, f"traincohort{args.cohort_per_spk}",
                            [r for _, r in cohort_items], TRAIN_DIR, **kw)

    mean = l2n(cohort).mean(0)
    emb_c, coh_c = center(emb, mean), center(cohort, mean)
    raw = cosine_scores(emb, ia, ib)
    cen = cosine_scores(emb_c, ia, ib)
    backends = {"cosine": raw, "center": cen}
    for k in args.topk:
        backends[f"asnorm{k}"] = asnorm(emb, cohort, ia, ib, top_k=k, raw=raw)
        backends[f"center+asnorm{k}"] = asnorm(emb_c, coh_c, ia, ib, top_k=k, raw=cen)

    # duration-aware calibration fitted on train-cohort trials (no test labels)
    import json as _json

    import soundfile as sf
    lengths = _json.loads((ROOT / "lists/train_lengths.json").read_text())
    coh_dur = np.array([lengths[r] / 16000 for _, r in cohort_items])
    dur_cache = ROOT / "lists" / f"{args.split}_durations.npy"
    if dur_cache.exists() and len(np.load(dur_cache)) == len(files):
        dur = np.load(dur_cache)
    else:
        dur = np.array([sf.info(str(Path(args.audio_root) / f)).duration for f in files])
        np.save(dur_cache, dur)
    coh_spk = [s for s, _ in cohort_items]
    for nm, (e_, c_) in {"": (l2n(emb), l2n(cohort)), "center+": (emb_c, coh_c)}.items():
        backends[f"{nm}qmf_dur"] = QMF(use_asnorm=False).fit(c_, coh_spk, coh_dur).score(e_, dur, ia, ib)
        backends[f"{nm}qmf_dur_as"] = QMF(use_asnorm=True).fit(c_, coh_spk, coh_dur).score(e_, dur, ia, ib)

    rec = dict(time=datetime.now().isoformat(timespec="seconds"), model=args.model, split=args.split,
               params_M=round(n_params / 1e6, 3), embed_sec=round(t_emb, 1), tag=args.tag, **kw)
    if labels is not None:
        res = {}
        for name, s in backends.items():
            res[name] = dict(eer=round(compute_eer(s, labels), 3), mindcf01=round(compute_min_dcf(s, labels), 4))
        rec["results"] = res
        best = min(res, key=lambda k: res[k]["eer"])
        rec["best"] = best
        print(json.dumps(rec, ensure_ascii=False))
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if args.submission:
        b = args.submission_backend or rec.get("best", "center+asnorm300")
        write_submission(Path(args.submission), pairs, backends[b])
        print(f"wrote {args.submission} ({b})")


if __name__ == "__main__":
    main()
