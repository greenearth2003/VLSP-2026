"""Post-training score calibration on silver, fitted on TRAIN data only (our own QMF / QMF2 from sred/calibrate.py,
as in the earlier experiments; no teammate component).

    .venv/bin/python experiments/distill_noiseaug_durationaware/calibrate_eval.py MODEL --tag NAME

Calibration data: the 5-utterances-per-speaker train cohort (8,657 files, 1,823 speakers), embedded twice: the clean
full utterance and a noisy copy made with our own training augmentation (NoiseAugmenter, base.json settings, no shared
backgrounds, fixed seed per file). Per-file quality measures, computed on the audio actually embedded: speech duration
(energy speech frames, as features_*.csv speech_dur) and SNR estimate (frame-level p95 - p10 dB).

Variants (all logistic regressions with class balance, fitted on cohort trials: all same-speaker pairs + 300k random):
  cos            plain cosine
  qmf_dur        QMF on [s, log dmin, log dmax, s*log dmin, s*log dmax], clean cohort only (earlier recipe)
  qmf_dur_noisy  same, cohort clean + noisy copies
  qmf2_dur_snr   QMF2 on speech duration and SNR (min / max per trial and their products with s), clean + noisy
Reported on silver: all / cleaned / private-like / natural duration buckets / minDCF. Embeddings are cached in
runs/_calib/<tag>/ so variants can be added without re-embedding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

import evaluate as ev  # noqa: E402
from dnd.data import load_config  # noqa: E402
from dnd.noise import NoiseAugmenter  # noqa: E402
from dnd.vad import HOP, SR, frame_db, speech_frames  # noqa: E402
from sred.calibrate import QMF, QMF2  # noqa: E402
from sred.data import TRAIN_DIR, cohort_subset, read_train_list  # noqa: E402
from sred.metrics import compute_min_dcf  # noqa: E402

CAP = 30 * SR  # embed at most the first 30 s of a file (as the evaluator's whole-clip view of long files)


def quality(x):
    db = frame_db(x)
    p10, p95 = np.percentile(db, [10, 95])
    return float(speech_frames(db).sum() * HOP / SR), float(p95 - p10)


@torch.inference_mode()
def embed_one(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return F.normalize(model(torch.from_numpy(x[:CAP])[None].cuda()).float(), dim=1)[0].cpu().numpy()


def build(model, out: Path):
    """Embeddings + qualities for cohort clean, cohort noisy and silver files (cached)."""
    out.mkdir(parents=True, exist_ok=True)
    f = out / "embeddings.npz"
    if f.exists():
        return dict(np.load(f, allow_pickle=True))
    items = read_train_list(SRED / "lists/clean/train_clean.txt")
    coh = cohort_subset(items, per_spk=5)
    cfg = load_config(EXP / "configs/base.json")
    a = dict(cfg["aug"], p_shared=0.0)  # single files: no speaker pairs
    b = cfg["banks"]
    aug = NoiseAugmenter(a, SRED / b["rir"], SRED / b["noise"], SRED / b["train_bg"])
    E = {k: [] for k in ("coh_clean", "coh_noisy", "silver")}
    Q = {k: [] for k in E}
    for i, (_, rel) in enumerate(coh):
        x, _ = sf.read(str(TRAIN_DIR / rel), dtype="float32", always_2d=False)
        x = x if x.ndim == 1 else x.mean(1)
        y, _ = aug(x[None, :CAP].copy(), np.random.default_rng([7, i]))
        for k, z in (("coh_clean", x), ("coh_noisy", y[0])):
            E[k].append(embed_one(model, z))
            Q[k].append(quality(z[:CAP]))
        if i % 2000 == 0:
            print(f"  cohort {i}/{len(coh)}", flush=True)
    tr = pd.read_csv(ev.DEV / "trials.csv")
    files = sorted(set(tr.a) | set(tr.b))
    for i, fn in enumerate(files):
        x, _ = sf.read(str(ev.SILVER / fn), dtype="float32", always_2d=False)
        E["silver"].append(embed_one(model, x))
        Q["silver"].append(quality(x[:CAP]))
        if i % 5000 == 0:
            print(f"  silver {i}/{len(files)}", flush=True)
    res = {f"emb_{k}": np.stack(v) for k, v in E.items()}
    res.update({f"q_{k}": np.array(v, np.float32) for k, v in Q.items()})
    res["coh_spk"] = np.array([s for s, _ in coh])
    res["silver_files"] = np.array(files)
    np.savez(f, **res)
    return res


def metrics(s, tr, y, w, cl):
    r = {"silver_all": ev.eer(s, y), "silver_cleaned": ev.eer(s[cl], y[cl]), "silver_private_like": ev.eer(s, y, w),
         "silver_minDCF": float(compute_min_dcf(s, y))}
    for bk in ["<0.75s", "0.75-1s", "1-2s", "2-3s", ">3s"]:
        k = (tr.bucket == bk).to_numpy()
        r[f"natural {bk}"] = ev.eer(s[k], y[k])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    model, _ = ev.load_student(a.model)
    model = model.cuda().eval()
    D = build(model, EXP / "runs/_calib" / a.tag)
    tr = pd.read_csv(ev.DEV / "trials.csv")
    y = tr.label.to_numpy()
    sw = np.load(ev.DEV / "silver_weights.npz")
    w, cl = sw["private_like"], sw["in_cleaned"]
    pos = {f: i for i, f in enumerate(D["silver_files"])}
    ia, ib = np.array([pos[f] for f in tr.a]), np.array([pos[f] for f in tr.b])
    es, qs = D["emb_silver"], D["q_silver"]
    spk = D["coh_spk"]
    ec, en = D["emb_coh_clean"], D["emb_coh_noisy"]
    qc, qn = D["q_coh_clean"], D["q_coh_noisy"]
    e2, q2, spk2 = np.concatenate([ec, en]), np.concatenate([qc, qn]), np.concatenate([spk, spk])

    scores = {"cos": np.sum(es[ia] * es[ib], 1),
              "qmf_dur": QMF(use_asnorm=False).fit(ec, spk, qc[:, 0]).score(es, qs[:, 0], ia, ib),
              "qmf_dur_noisy": QMF(use_asnorm=False).fit(e2, spk2, q2[:, 0]).score(es, qs[:, 0], ia, ib)}
    qm = QMF2(["speech", "snr"]).fit(e2, spk2, {"speech": q2[:, 0], "snr": q2[:, 1]})
    scores["qmf2_dur_snr"] = qm.score(es, {"speech": qs[:, 0], "snr": qs[:, 1]}, ia, ib)

    rows = {k: metrics(s, tr, y, w, cl) for k, s in scores.items()}
    (ev.OUT / f"calib__{a.tag}.json").write_text(json.dumps(rows, indent=1))
    cols = list(next(iter(rows.values())))
    print(f"{a.tag} (silver EER %)")
    print(f"  {'variant':14s}" + "".join(f"{c.replace('silver_', '').replace('natural ', ''):>13s}" for c in cols))
    for k, r in rows.items():
        print(f"  {k:14s}" + "".join(f"{r[c]:13.3f}" for c in cols))


if __name__ == "__main__":
    main()
