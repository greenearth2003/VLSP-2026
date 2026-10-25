"""Re-select the transductive smoothing parameters on the noisy stand-ins (the private set is ~10 dB noisier
than public, where k 20 / w 0.5 / tau 0.6 were chosen). Cached embeddings only (run sred.evaluate first).

    python scripts/transductive_grid_noisy.py exp/ft/W_E2b_sf2/avg.pt > exp/analysis/transductive_grid_noisy.csv

Selection split: P-dev of public_noisy; checks: P-hold, full, and the same on public_realnoise / clean public.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF2  # noqa: E402
from sred.data import cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.metrics import compute_eer  # noqa: E402
from sred.protocol import _half  # noqa: E402
from sred.scoring import l2n  # noqa: E402
from sred.transductive import knn_smooth  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
model = sys.argv[1]
pairs, y = read_trials()
files = unique_files(pairs)
pos = {f: i for i, f in enumerate(files)}
ia = np.array([pos[a] for a, _ in pairs])
ib = np.array([pos[b] for _, b in pairs])
dur = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel").loc[files].dur.values
ident = json.loads((ROOT / "lists/clean/public_identity.json").read_text())
keep = {tuple(l.split()[:2]) for l in open(ROOT / "lists/clean/public_trials_clean.txt")}
clean = np.array([p in keep for p in pairs])
ha = np.array([_half(ident[x.split("/")[0]]) for x, _ in pairs])
hb = np.array([_half(ident[x.split("/")[0]]) for _, x in pairs])
dev, hold = clean & (ha == 0) & (hb == 0), clean & (ha == 1) & (hb == 1)

ci = cohort_subset(read_train_list(), per_spk=5)
C = l2n(np.load(cache_key(model, "traincohort5", [r for _, r in ci], mode="full")))
lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
cdur = np.array([lengths[r] / 16000 for _, r in ci])
qmf = QMF2(["dur"]).fit(C, np.array([s for s, _ in ci]), {"dur": cdur})

grid = [None] + list(itertools.product([10, 20, 40], [0.3, 0.5, 1.0], [0.5, 0.55, 0.6, 0.65, 0.7]))
rows = []
for split in ["public_noisy", "public_realnoise", "public"]:
    E0 = l2n(np.load(cache_key(model, split, files, mode="full")))
    for g in grid:
        E = E0 if g is None else knn_smooth(E0, g[0], g[1], g[2], 2, min_cluster=3)
        s = qmf.score(E, {"dur": dur}, ia, ib)
        rows.append(dict(split=split, k=g and g[0], w=g and g[1], tau=g and g[2],
                         Pdev=compute_eer(s[dev], y[dev]), Phold=compute_eer(s[hold], y[hold]), full=compute_eer(s, y)))
        print(rows[-1], file=sys.stderr, flush=True)
pd.DataFrame(rows).round(3).to_csv(sys.stdout, index=False)
