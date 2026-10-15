"""Build lists/train_clean.txt from TRAIN data only (pretrained ReDimNet-M embeddings, no test info).

Rules (thresholds fixed in the EDA report, not tuned):
 1. suspicious utterances: LOO own-centroid cos < 0.35 and another centroid > own + 0.1  -> drop
 2. near-duplicate recordings (cos > 0.97) across different IDs                         -> drop both
 3. bimodal IDs (>= 8 utts, 2-means centroid cos < 0.5, minor cluster >= 15 %)          -> minor cluster = new ID '<id>_b'
 4. same voice under two IDs (centroid cos >= 0.9)                                       -> merge (union-find)
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sred.data import read_train_list  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.scoring import l2n  # noqa: E402

tr = read_train_list()
rels = np.array([r for _, r in tr])
spk = np.array([s for s, _ in tr], dtype=object)
E = l2n(np.load(cache_key("redimnet_M_mix", "trainall", list(rels), mode="full")))
stats = {"n_utts_in": len(rels), "n_spk_in": len(set(spk))}

# 1. suspicious labels
u, inv = np.unique(spk.astype(str), return_inverse=True)
S = np.zeros((len(u), E.shape[1]))
np.add.at(S, inv, E)
n = np.bincount(inv)
C = l2n(S / n[:, None])
own = np.full(len(E), np.nan)
m = n[inv] > 1
own[m] = np.sum(l2n(S[inv[m]] - E[m]) * E[m], 1)
other = np.empty(len(E))
for s in range(0, len(E), 4096):
    sim = E[s:s + 4096] @ C.T
    sim[np.arange(len(sim)), inv[s:s + 4096]] = -9
    other[s:s + 4096] = sim.max(1)
drop = m & (other > own + 0.1) & (own < 0.35)
stats["drop_suspicious"] = int(drop.sum())

# 2. cross-ID near duplicates
nd = 0
for s in range(0, len(E), 4096):
    sim = E[s:s + 4096] @ E.T
    i, j = np.where(sim > 0.97)
    for a, b in zip(s + i, j):
        if a < b and spk[a] != spk[b]:
            drop[a] = drop[b] = True
            nd += 1
stats["cross_id_near_dup_pairs"] = nd
stats["drop_total_after_2"] = int(drop.sum())

# 3. split bimodal IDs
new_spk = spk.copy()
n_split = 0
for k, s_ in enumerate(u):
    idx = np.where((inv == k) & ~drop)[0]
    if len(idx) < 8:
        continue
    km = KMeans(2, n_init=3, random_state=0).fit(E[idx])
    c = l2n(km.cluster_centers_)
    cnt = np.bincount(km.labels_, minlength=2)
    if c[0] @ c[1] < 0.5 and cnt.min() / len(idx) >= 0.15:
        minor = int(cnt.argmin())
        new_spk[idx[km.labels_ == minor]] = f"{s_}_b"
        n_split += 1
stats["split_bimodal_ids"] = n_split

# 4. merge IDs with near-identical centroids (computed on the cleaned/split labels)
keep = ~drop
u2, inv2 = np.unique(new_spk[keep].astype(str), return_inverse=True)
S2 = np.zeros((len(u2), E.shape[1]))
np.add.at(S2, inv2, E[keep])
C2 = l2n(S2)
parent = list(range(len(u2)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


sim = C2 @ C2.T
ii, jj = np.where(np.triu(sim, 1) >= 0.9)
for a, b in zip(ii, jj):
    parent[find(a)] = find(b)
root_name = {}
for k in range(len(u2)):
    r = find(k)
    root_name.setdefault(r, u2[r])
merged = np.array([root_name[find(k)] for k in inv2], dtype=object)
stats["merge_pairs_cos_ge_0.9"] = int(len(ii))
stats["n_utts_out"] = int(keep.sum())
stats["n_spk_out"] = int(len(set(merged)))

out = ROOT / "lists/train_clean.txt"
with open(out, "w") as f:
    for s_, r in zip(merged, rels[keep]):
        f.write(f"{s_} {r}\n")
(ROOT / "lists/train_clean_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
