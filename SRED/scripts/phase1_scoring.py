"""Milestone M1 / Phase 1: score-side improvements, evaluated on the cleaned public trials.

    python scripts/phase1_scoring.py --model exp/ft/C01_M/step3000.pt [--emb-tag trim]

Variants (all fitted on train-cohort synthetic trials only, no test labels):
  V0  QMF(file duration)                         -- current submission back-end
  V1  QMF2(speech duration)                      -- energy-VAD speech seconds instead of file seconds
  V2  QMF2(file + speech duration)
  V3  QMF2(speech duration + SNR)
  V4  QMF2(speech) + AS-norm (whole cohort) as extra score
  V5  QMF2(speech) + gender-matched AS-norm      -- cohort restricted to the test file's F0-gender
Decision split: P-dev (md5 speaker halves on *merged identities*) intersected with cleaned trials.
Test-side gender uses per-FILE F0 only (folder/speaker IDs of the test set are never used by a system).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF, QMF2
from sred.data import cohort_subset, read_train_list, read_trials, unique_files
from sred.embed import cache_key
from sred.metrics import compute_eer, compute_min_dcf
from sred.protocol import _half
from sred.scoring import l2n

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "exp/analysis"


def topk_stats(e, eg, c, cg, k=300, spk_e=None, spk_c=None, gender=True, chunk=4096):
    """Mean/std of the top-k cohort scores; optionally only cohort of the same gender and never the same speaker."""
    mu, sd = np.empty(len(e), np.float32), np.empty(len(e), np.float32)
    for s in range(0, len(e), chunk):
        S = e[s:s + chunk] @ c.T
        if spk_e is not None:
            S[spk_e[s:s + chunk, None] == spk_c[None, :]] = -9
        if gender:
            g = eg[s:s + chunk, None]
            S[(g != "?") & (cg[None, :] != g)] = -9
        t = np.partition(S, -k, axis=1)[:, -k:]
        mu[s:s + chunk], sd[s:s + chunk] = t.mean(1), t.std(1) + 1e-6
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="exp/ft/C01_M/step3000.pt")
    ap.add_argument("--emb-split", default="public", help="embedding cache split name (e.g. public_trim)")
    ap.add_argument("--cohort-split", default="traincohort5")
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--mode", default="full", help="embedding mode used in the cache key (full | vad)")
    a = ap.parse_args()

    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs])
    ib = np.array([pos[x] for _, x in pairs])
    ci = cohort_subset(read_train_list(), per_spk=5)
    cfiles, cspk = [r for _, r in ci], np.array([s for s, _ in ci])

    emb = l2n(np.load(cache_key(a.model, a.emb_split, files, mode=a.mode)))
    coh = l2n(np.load(cache_key(a.model, a.cohort_split, cfiles, mode=a.mode)))

    fp = pd.read_csv(A / "features_public.csv").set_index("rel").loc[files]
    ft = pd.read_csv(A / "features_train.csv").set_index("rel").loc[cfiles]
    qp = {"dur": fp.dur.values, "speech": fp.speech_dur.values, "snr": fp.snr_db.values}
    qc = {"dur": ft.dur.values, "speech": ft.speech_dur.values, "snr": ft.snr_db.values}
    # gender: test = per-file F0; cohort = per-speaker median F0 (train labels are allowed)
    gp = np.where(np.isnan(fp.f0.values), "?", np.where(fp.f0.values < 165, "M", "F"))
    sf0 = ft.assign(spk=cspk).groupby("spk").f0.median()
    gc = np.array(["M" if sf0[s] < 165 else "F" for s in cspk])

    # evaluation subsets
    ident = json.loads((ROOT / "lists/clean/public_identity.json").read_text())
    keep = {tuple(l.split()[:2]) for l in open(ROOT / "lists/clean/public_trials_clean.txt")}
    clean = np.array([p in keep for p in pairs])
    ha = np.array([_half(ident[x.split("/")[0]]) for x, _ in pairs])
    hb = np.array([_half(ident[x.split("/")[0]]) for _, x in pairs])
    dev, hold = clean & (ha == 0) & (hb == 0), clean & (ha == 1) & (hb == 1)
    sa = np.array([ident[x.split("/")[0]] for x, _ in pairs])
    sb = np.array([ident[x.split("/")[0]] for _, x in pairs])

    V = {}
    lens = json.loads((ROOT / "lists/train_lengths.json").read_text())
    cdur = np.array([lens[r] / 16000 for r in cfiles])
    V["V0 QMF(file dur)"] = QMF(use_asnorm=False).fit(coh, cspk, cdur).score(emb, fp.dur.values, ia, ib)
    for name, q in (("V1 QMF2(speech)", ["speech"]), ("V2 QMF2(dur+speech)", ["dur", "speech"]),
                    ("V3 QMF2(speech+snr)", ["speech", "snr"])):
        V[name] = QMF2(q).fit(coh, cspk, {k: qc[k] for k in q}).score(emb, {k: qp[k] for k in q}, ia, ib)

    s_raw = np.sum(emb[ia] * emb[ib], 1)
    for name, gender in (("V4 QMF2(speech)+ASnorm", False), ("V5 QMF2(speech)+gender-ASnorm", True)):
        mu_t, sd_t = topk_stats(emb, gp, coh, gc, a.k, gender=gender)
        asn_t = 0.5 * ((s_raw - mu_t[ia]) / sd_t[ia] + (s_raw - mu_t[ib]) / sd_t[ib])
        mu_c, sd_c = topk_stats(coh, gc, coh, gc, a.k, spk_e=cspk, spk_c=cspk, gender=gender)
        fn = lambda I, J, s, mu_c=mu_c, sd_c=sd_c: 0.5 * ((s - mu_c[I]) / sd_c[I] + (s - mu_c[J]) / sd_c[J])  # noqa: E731
        V[name] = QMF2(["speech"]).fit(coh, cspk, {"speech": qc["speech"]}, extra_fn=fn).score(
            emb, {"speech": qp["speech"]}, ia, ib, extra=asn_t)

    # paired speaker bootstrap on P-dev (clean) vs V0
    spks = np.array(sorted(set(sa[dev]) | set(sb[dev])))
    rng = np.random.default_rng(0)
    W = []
    for _ in range(300):
        p = set(rng.choice(spks, len(spks), replace=True))
        W.append(np.array([(u in p) and (v in p) for u, v in zip(sa[dev], sb[dev])]))
    base = V["V0 QMF(file dur)"][dev]
    rows = []
    for name, s in V.items():
        d = np.array([compute_eer(s[dev][w], y[dev][w]) - compute_eer(base[w], y[dev][w]) for w in W if y[dev][w].sum() > 100])
        rows.append(dict(variant=name, Pdev_clean=compute_eer(s[dev], y[dev]), Phold_clean=compute_eer(s[hold], y[hold]),
                         all_clean=compute_eer(s[clean], y[clean]), full=compute_eer(s, y),
                         minDCF_clean=compute_min_dcf(s[clean], y[clean]),
                         dEER_Pdev=d.mean(), P_better=(d < 0).mean()))
    df = pd.DataFrame(rows)
    print(f"model {a.model}  emb {a.emb_split}  | P-dev clean trials {dev.sum()}, P-hold clean {hold.sum()}")
    print(df.round(3).to_string(index=False))
    out = A / "phase1" / f"{Path(a.model).parent.name}_{a.emb_split}_{a.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
