"""Compare systems on the cleaned public split (P-dev decisions, P-hold check) vs a reference.

    python scripts/compare.py REF_CKPT CKPT [CKPT ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_analysis import qmf_scores  # noqa: E402
from sred.data import read_trials, unique_files  # noqa: E402
from sred.metrics import compute_eer, compute_min_dcf  # noqa: E402
from sred.protocol import _half  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
pairs, y = read_trials()
files = unique_files(pairs)
pos = {f: i for i, f in enumerate(files)}
ia = np.array([pos[a] for a, _ in pairs])
ib = np.array([pos[b] for _, b in pairs])
f = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel").loc[files]
dur, sp = f.dur.values, f.speech_dur.values
short_all = np.minimum(sp[ia], sp[ib]) < 1
ident = json.loads((ROOT / "lists/clean/public_identity.json").read_text())
keep = {tuple(l.split()[:2]) for l in open(ROOT / "lists/clean/public_trials_clean.txt")}
clean = np.array([p in keep for p in pairs])
sa = np.array([ident[x.split("/")[0]] for x, _ in pairs])
sb = np.array([ident[x.split("/")[0]] for _, x in pairs])
ha, hb = np.array([_half(s) for s in sa]), np.array([_half(s) for s in sb])
dev, hold, short = clean & (ha == 0) & (hb == 0), clean & (ha == 1) & (hb == 1), clean & short_all
spks = np.array(sorted(set(sa[dev]) | set(sb[dev])))
rng = np.random.default_rng(0)
W = [np.array([(u in p) and (v in p) for u, v in zip(sa[dev], sb[dev])])
     for p in [set(rng.choice(spks, len(spks), replace=True)) for _ in range(300)]]
ckpts = sys.argv[1:]
S = {c: qmf_scores(c, files, ia, ib, dur) for c in ckpts}
b, yd = S[ckpts[0]][dev], y[dev]
rows = []
for c, s in S.items():
    d = np.array([compute_eer(s[dev][w], yd[w]) - compute_eer(b[w], yd[w]) for w in W if yd[w].sum() > 100])
    rows.append(dict(system=c.replace("exp/ft/", ""), Pdev=compute_eer(s[dev], yd), Phold=compute_eer(s[hold], y[hold]),
                     clean=compute_eer(s[clean], y[clean]), full=compute_eer(s, y), short_lt1s=compute_eer(s[short], y[short]),
                     minDCF_clean=compute_min_dcf(s[clean], y[clean]), dPdev=d.mean(), P_better=(d < 0).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
