"""Ablation table of the final SRED system, all rows scored identically from cached embeddings.

    python scripts/ablation.py > exp/analysis/ablation.md

Metrics: EER % on the full public test (= leaderboard), on the cleaned trials, on P-dev / P-hold
(speaker-disjoint halves of the cleaned trials), minDCF(0.01) on the full test, and EER on cleaned trials
whose shorter side has < 1 s of speech. Rows whose checkpoint is missing are skipped with a note.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF2
from sred.data import cohort_subset, read_train_list, read_trials, unique_files
from sred.embed import cache_key
from sred.metrics import compute_eer, compute_min_dcf
from sred.protocol import _half
from sred.scoring import l2n
from sred.transductive import knn_smooth

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "exp/analysis"

pairs, y = read_trials()
files = unique_files(pairs)
pos = {f: i for i, f in enumerate(files)}
ia = np.array([pos[a] for a, _ in pairs])
ib = np.array([pos[b] for _, b in pairs])
ci = cohort_subset(read_train_list(), per_spk=5)
cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
fp = pd.read_csv(A / "features_public.csv").set_index("rel").loc[files]
ft = pd.read_csv(A / "features_train.csv").set_index("rel").loc[cf]
qp, qc = {"dur": fp.dur.values}, {"dur": ft.dur.values}
ident = json.loads((ROOT / "lists/clean/public_identity.json").read_text())
keep = {tuple(l.split()[:2]) for l in open(ROOT / "lists/clean/public_trials_clean.txt")}
clean = np.array([p in keep for p in pairs])
sa = np.array([ident[x.split("/")[0]] for x, _ in pairs])
sb = np.array([ident[x.split("/")[0]] for _, x in pairs])
ha, hb = np.array([_half(s) for s in sa]), np.array([_half(s) for s in sb])
dev, hold = clean & (ha == 0) & (hb == 0), clean & (ha == 1) & (hb == 1)
sp = fp.speech_dur.values
short = clean & (np.minimum(sp[ia], sp[ib]) < 1)


def score(model, backend="qmf", smooth=None):
    E = l2n(np.load(cache_key(model, "public", files, mode="full")))
    if smooth:
        E = knn_smooth(E, **smooth)
    if backend == "cosine":
        return np.sum(E[ia] * E[ib], 1)
    C = l2n(np.load(cache_key(model, "traincohort5", cf, mode="full")))
    return QMF2(["dur"]).fit(C, cs, qc).score(E, qp, ia, ib)


def row(name, model, backend="qmf", smooth=None, params=4.81):
    try:
        s = score(model, backend, smooth)
    except FileNotFoundError:
        return dict(system=name, params_M=params, note="(not run / no cache)")
    return dict(system=name, params_M=params, full=compute_eer(s, y), minDCF=compute_min_dcf(s, y),
                cleaned=compute_eer(s[clean], y[clean]), P_dev=compute_eer(s[dev], y[dev]),
                P_hold=compute_eer(s[hold], y[hold]), short_lt1s=compute_eer(s[short], y[short]))


AGGR = dict(k=20, w=0.5, tau=0.6, iters=2)
build = [
    ("A. pretrained ReDimNet-M, cosine", "redimnet_M_mix", "cosine"),
    ("B. A + QMF duration calibration", "redimnet_M_mix", "qmf"),
    ("C. B + fine-tune (AAM + aug, 3k steps, stem..stage2 frozen)", "exp/ft/C00_M/step3000.pt", "qmf"),
    ("D. C + cleaned train list", "exp/ft/C01_M/step3000.pt", "qmf"),
    ("E. D + long->short KD from fine-tuned B6, all layers, 18k steps", "exp/ft/K21_M/last.pt", "qmf"),
    ("F. E + WiSE-FT 0.5 (step 12k ⊕ pretrained)  = submitted K21", "exp/ft/W_K21_M/avg.pt", "qmf"),
    ("G. F + transductive kNN smoothing (k20 w0.5 tau0.6 x2)", "exp/ft/W_K21_M/avg.pt", "qmf", AGGR),
]
remove = [
    ("F without WiSE (= E)", "exp/ft/K21_M/last.pt", "qmf"),
    ("F without KD (same schedule, all layers, 18k, cleaned) + WiSE", "exp/ft/W_N01_noKD/avg.pt", "qmf"),
    ("F without cleaning (original train.txt) + WiSE", "exp/ft/W_N02_noclean/avg.pt", "qmf"),
    ("F with KD but early layers frozen, 3k steps (K10) + WiSE", "exp/ft/W_K10_M/avg.pt", "qmf"),
    ("F with weaker teacher C01_B6 (K11, 9k) + WiSE", "exp/ft/W_K11_M/avg.pt", "qmf"),
    ("F with KD weights x1 (K20, 9k) + WiSE", "exp/ft/W_K20_M/avg.pt", "qmf"),
    ("F with 9k steps (K22) + WiSE", "exp/ft/W_K22_M/avg.pt", "qmf"),
    ("F with 36k steps (K23) + WiSE", "exp/ft/W_K23_M/avg.pt", "qmf"),
    ("F + adaptive LMFT stage (L02) + WiSE", "exp/ft/W_L02_K21_almft/avg.pt", "qmf"),
    ("F + duration-adaptive margin (K26) + WiSE", "exp/ft/W_K26_M/avg.pt", "qmf"),
]
students = [
    ("ReDimNet-S, same recipe as D (C01_S)", "exp/ft/C01_S/step3000.pt", "qmf", None, 3.15),
    ("ReDimNet2-B4, KD recipe (K30) + WiSE", "exp/ft/W_K30_B4/avg.pt", "qmf", None, 6.66),
    ("B6 pruned to 6.33M + 18k recovery (P01)", "exp/ft/P01_PB6/last.pt", "qmf", None, 6.33),
    ("teacher: fine-tuned B6 + WiSE (not submittable)", "exp/ft/W_G05_B6/avg.pt", "qmf", None, 12.46),
]


def table(rows):
    d = pd.DataFrame(rows)
    cols = ["system", "params_M", "full", "cleaned", "P_dev", "P_hold", "minDCF", "short_lt1s", "note"]
    d = d.reindex(columns=[c for c in cols if c in d])
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) and not np.isnan(v) else ("" if (isinstance(v, float) and np.isnan(v)) else str(v))
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "|".join("---" for _ in d.columns) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r.values) + " |")
    return "\n".join(lines)


print("# Ablation — VLSP 2026 SRED final system (public test)\n")
print("EER % (lower is better). full = organisers' 49,993 trials (leaderboard metric); cleaned = 47,932 trials after "
      "removing label-noise trials; P-dev / P-hold = speaker-disjoint halves of cleaned; minDCF(0.01) on full; "
      "short = cleaned trials with < 1 s speech on the shorter side.\n")
print("## 1. Build-up (each row adds one component)\n")
print(table([row(*r) for r in build]))
print("\n## 2. Leave-one-out / variants of the submitted system F\n")
print(table([row(*r) for r in remove]))
print("\n## 3. Other students and the teacher\n")
print(table([row(n, m, b, s, p) for n, m, b, s, p in students]))
print("\nAll rows: cosine + QMF(duration) back-end fitted on train-cohort trials (no test labels); WiSE-FT = 0.5 fine-tuned "
      "+ 0.5 pretrained weights. Decisions were made on P-dev (paired speaker bootstrap, accept if P(better) > 0.95); "
      "P-hold is the check. Transductive row uses unlabelled test audio only.")
