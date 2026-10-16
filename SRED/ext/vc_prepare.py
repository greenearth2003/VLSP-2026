"""Prepare Vietnam-Celeb (VC) for training, with leakage + label-noise screening (no test labels used).

1. 10 utterances / VC speaker -> pretrained ReDimNet-M embeddings -> speaker centroids.
2. Drop VC speakers whose centroid is close (cos >= 0.80) to ANY VLSP train or public-test speaker centroid
   (protects against test-speaker leakage; same rule as the EDA duplicate-identity check).
3. Drop mixed-voice VC speakers (2-means on the 10 embeddings: centroid cos < 0.5, minor cluster >= 15 %).
4. Write lists/train_vlsp_vc.txt = lists/train_drop_bad.txt + remaining VC utterances (absolute paths, 'vc_' IDs).
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sred.data import read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key, extract_cached  # noqa: E402
from sred.models import load_model  # noqa: E402
from sred.scoring import l2n  # noqa: E402

VC = Path("/home/jovyan/data/ext/vietnam_celeb/wav")
rows = []
for split in ["train", "test"]:
    for line in (ROOT / f"lists/vc_{split}.txt").read_text().splitlines():
        if line.strip():
            s, rel = line.split()
            rows.append((s, str(VC / rel)))
by = defaultdict(list)
for s, p in rows:
    by[s].append(p)
rng = random.Random(0)
sample = []
for s in sorted(by):
    ps = sorted(by[s])
    rng.shuffle(ps)
    sample += [(s, p) for p in ps[:10]]
E = l2n(extract_cached("redimnet_M_mix", lambda: load_model("redimnet_M_mix"), "vc_sample10",
                       [p for _, p in sample], Path("/"), mode="full"))
sspk = np.array([s for s, _ in sample])
vu = sorted(by)
VCc = l2n(np.stack([E[sspk == s].mean(0) for s in vu]))


def centroids(X, spk):
    u = sorted(set(spk))
    spk = np.array(spk)
    return u, l2n(np.stack([X[spk == k].mean(0) for k in u]))


tr = read_train_list()
_, Ct = centroids(l2n(np.load(cache_key("redimnet_M_mix", "trainall", [r for _, r in tr], mode="full"))),
                  [s for s, _ in tr])
pairs, _ = read_trials()
pf = unique_files(pairs)
_, Cp = centroids(l2n(np.load(cache_key("redimnet_M_mix", "public", pf, mode="full"))), [f.split("/")[0] for f in pf])
max_tr, max_pu = (VCc @ Ct.T).max(1), (VCc @ Cp.T).max(1)
leak = {s for s, a, b in zip(vu, max_tr, max_pu) if max(a, b) >= 0.80}

mixed = set()
for s in vu:
    X = E[sspk == s]
    if len(X) < 8:
        continue
    km = KMeans(2, n_init=3, random_state=0).fit(X)
    c = l2n(km.cluster_centers_)
    if c[0] @ c[1] < 0.5 and np.bincount(km.labels_).min() / len(X) >= 0.15:
        mixed.add(s)

keep = [(s, p) for s, p in rows if s not in leak and s not in mixed]
base = (ROOT / "lists/train_drop_bad.txt").read_text().splitlines()
(ROOT / "lists/train_vlsp_vc.txt").write_text("\n".join(base + [f"{s} {p}" for s, p in keep]) + "\n")
stats = dict(vc_speakers=len(vu), vc_utts=len(rows), dropped_leak=len(leak), dropped_mixed=len(mixed),
             vc_kept_speakers=len({s for s, _ in keep}), vc_kept_utts=len(keep), vlsp_utts=len(base),
             total_utts=len(base) + len(keep),
             max_cos_to_vlsp_train_p50=float(np.median(max_tr)), max_cos_to_public_p50=float(np.median(max_pu)),
             leak_examples=sorted(leak)[:20])
(ROOT / "lists/train_vlsp_vc_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
