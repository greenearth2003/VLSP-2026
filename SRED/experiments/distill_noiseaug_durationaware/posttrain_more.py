"""More post-training methods (after posttrain_embed.py / posttrain_eval.py), all fitted on train only:

  chunk      clips longer than 6 s are embedded as the mean of 6 s windows (3 s hop, last window flush with the end)
             instead of one pass over up to 30 s (E4 trained on crops of at most 6 s); cohort treated the same way
  nonlinear  calibrators that are not linear in the features: gradient-boosted trees (score kept monotone) and a
             per-duration-bin logistic regression, on the QMF2 features (cosine, speech duration, SNR, raw norm)

    .venv/bin/python experiments/distill_noiseaug_durationaware/posttrain_more.py chunk-embed MODEL --tag NAME
    .venv/bin/python experiments/distill_noiseaug_durationaware/posttrain_more.py eval --tag NAME
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

import evaluate as ev  # noqa: E402
from calibrate_eval import CAP, metrics  # noqa: E402
from posttrain_eval import QMF2Files, build, l2n  # noqa: E402

WIN, HOP = 6 * 16000, 3 * 16000


@torch.inference_mode()
def chunk_embed(model, x):
    x = x[:CAP]
    if len(x) <= WIN:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return l2n(model(torch.from_numpy(x)[None].cuda()).float().cpu().numpy())[0]
    starts = sorted(set(list(range(0, len(x) - WIN + 1, HOP)) + [len(x) - WIN]))
    w = np.stack([x[s:s + WIN] for s in starts])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e = l2n(model(torch.from_numpy(w).cuda()).float().cpu().numpy())
    return l2n(e.mean(0))


def cmd_chunk_embed(model_spec, tag):
    from dnd.data import load_config
    from dnd.noise import NoiseAugmenter
    from sred.data import TRAIN_DIR, cohort_subset, read_train_list
    model, _ = ev.load_student(model_spec)
    model = model.cuda().eval()
    D = dict(np.load(EXP / "runs/_calib" / tag / "posttrain.npz"))
    R = {"silver_chunk": [], "coh_clean_chunk": [], "coh_noisy_chunk": []}
    for i, fn in enumerate(D["silver_files"]):
        x, _ = sf.read(str(ev.SILVER / fn), dtype="float32", always_2d=False)
        R["silver_chunk"].append(chunk_embed(model, x if x.ndim == 1 else x.mean(1)))
        if i % 5000 == 0:
            print(f"  silver {i}/{len(D['silver_files'])}", flush=True)
    coh = cohort_subset(read_train_list(SRED / "lists/clean/train_clean.txt"), per_spk=5)
    cfg = load_config(EXP / "configs/base.json")
    b = cfg["banks"]
    aug = NoiseAugmenter(dict(cfg["aug"], p_shared=0.0), SRED / b["rir"], SRED / b["noise"], SRED / b["train_bg"])
    for i, (_, rel) in enumerate(coh):
        x, _ = sf.read(str(TRAIN_DIR / rel), dtype="float32", always_2d=False)
        x = (x if x.ndim == 1 else x.mean(1))[:CAP]
        y = aug(x[None].copy(), np.random.default_rng([7, i]))[0][0]  # the same noisy copy as posttrain_embed.py
        R["coh_clean_chunk"].append(chunk_embed(model, x))
        R["coh_noisy_chunk"].append(chunk_embed(model, y))
        if i % 2000 == 0:
            print(f"  cohort {i}/{len(coh)}", flush=True)
    np.savez(EXP / "runs/_calib" / tag / "chunk.npz", **{k: np.stack(v) for k, v in R.items()})
    print("->", EXP / "runs/_calib" / tag / "chunk.npz")


def cohort_trials(spk, fid, n_non=300000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(spk)
    I, J = rng.integers(0, n, n_non), rng.integers(0, n, n_non)
    by = {}
    for i, s in enumerate(spk):
        by.setdefault(s, []).append(i)
    pairs = [p for idx in by.values() for p in itertools.combinations(idx, 2)]
    I, J = np.concatenate([I, [p[0] for p in pairs]]), np.concatenate([J, [p[1] for p in pairs]])
    keep = fid[I] != fid[J]
    return I[keep], J[keep], (spk[I[keep]] == spk[J[keep]]).astype(int)


def feats(s, qa, qb):
    """cosine + per-trial min / max of log speech duration, SNR and raw norm (no products: trees find them)."""
    f = [s]
    for k, lg in ((0, True), (1, False), (2, False)):
        a, b = qa[:, k], qb[:, k]
        if lg:
            a, b = np.log(np.maximum(a, 0.05)), np.log(np.maximum(b, 0.05))
        f += [np.minimum(a, b), np.maximum(a, b)]
    return np.stack(f, 1)


def cmd_eval(tag):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    D = dict(np.load(EXP / "runs/_calib" / tag / "posttrain.npz"))
    tr = pd.read_csv(ev.DEV / "trials.csv")
    y = tr.label.to_numpy()
    sw = np.load(ev.DEV / "silver_weights.npz")
    w, cl = sw["private_like"], sw["in_cleaned"]
    pos = {f: i for i, f in enumerate(D["silver_files"])}
    ia, ib = np.array([pos[f] for f in tr.a]), np.array([pos[f] for f in tr.b])
    ce, cs, cf, cq, se, sq = build(D)
    rows = {}

    # reference: QMF2 duration + SNR (linear), same cohort trials
    qd = lambda q: {"speech": q[:, 0], "snr": q[:, 1]}  # noqa: E731
    m = QMF2Files(["speech", "snr"]).fit_files(ce, cs, cf, qd(cq))
    rows["qmf2 (reference)"] = metrics(m.score(se, qd(sq), ia, ib), tr, y, w, cl)

    # non-linear calibrators on the same trials
    I, J, yt = cohort_trials(cs, cf)
    Xc = feats(np.sum(ce[I] * ce[J], 1), cq[I], cq[J])
    Xs = feats(np.sum(se[ia] * se[ib], 1), sq[ia], sq[ib])
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, class_weight="balanced",
                                        monotonic_cst=[1] + [0] * (Xc.shape[1] - 1), random_state=0).fit(Xc, yt)
    rows["boosted trees"] = metrics(gb.decision_function(Xs), tr, y, w, cl)
    edges = [0, 1.0, 2.0, 4.0, 1e9]  # bins of the trial's shorter speech duration
    s_out = np.zeros(len(ia))
    for lo, hi in zip(edges[:-1], edges[1:]):
        kc, ks = (Xc[:, 1] >= np.log(lo + 1e-9)) & (Xc[:, 1] < np.log(hi)), (Xs[:, 1] >= np.log(lo + 1e-9)) & (Xs[:, 1] < np.log(hi))
        mu, sd = Xc[kc].mean(0), Xc[kc].std(0) + 1e-9
        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit((Xc[kc] - mu) / sd, yt[kc])
        s_out[ks] = lr.decision_function((Xs[ks] - mu) / sd)
    rows["per-duration-bin logistic"] = metrics(s_out, tr, y, w, cl)

    # chunked embeddings (if the chunk pass exists)
    cp = EXP / "runs/_calib" / tag / "chunk.npz"
    if cp.exists():
        C = dict(np.load(cp))
        n = len(D["coh_spk"])
        cc = np.concatenate([C["coh_clean_chunk"], C["coh_noisy_chunk"]])
        sc = C["silver_chunk"]
        rows["chunk: cosine"] = metrics(np.sum(sc[ia] * sc[ib], 1), tr, y, w, cl)
        m = QMF2Files(["speech", "snr"]).fit_files(cc, cs[:2 * n], cf[:2 * n], qd(cq[:2 * n]))
        rows["chunk: qmf2"] = metrics(m.score(sc, qd(sq), ia, ib), tr, y, w, cl)
        cw, sw2 = l2n(cc + ce[:2 * n]), l2n(sc + se)
        m = QMF2Files(["speech", "snr"]).fit_files(cw, cs[:2 * n], cf[:2 * n], qd(cq[:2 * n]))
        rows["chunk + whole mean: qmf2"] = metrics(m.score(sw2, qd(sq), ia, ib), tr, y, w, cl)

    (ev.OUT / f"posttrain_more__{tag}.json").write_text(json.dumps(rows, indent=1))
    cols = ["silver_all", "silver_cleaned", "silver_private_like", "silver_minDCF", "natural <0.75s", "natural 1-2s",
            "natural >3s"]
    print(f"{tag}: silver EER %")
    print(f"  {'variant':28s}" + "".join(f"{c.replace('silver_', '').replace('natural ', ''):>13s}" for c in cols))
    for k, r in rows.items():
        print(f"  {k:28s}" + "".join(f"{r[c]:13.3f}" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chunk-embed")
    c.add_argument("model")
    c.add_argument("--tag", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--tag", required=True)
    a = ap.parse_args()
    cmd_chunk_embed(a.model, a.tag) if a.cmd == "chunk-embed" else cmd_eval(a.tag)
