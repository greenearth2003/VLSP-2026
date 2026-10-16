"""Rebuild lists/train_drop_bad.txt (train list used by S04 / S04+QS) from TRAIN data only.

Uses pretrained ReDimNet-M embeddings of every train utterance (eda/extract_train_emb.py) and removes:
 1. all utterances of mixed-voice IDs: >= 8 utts, 2-means (n_init=3, seed 0) centroid cos < 0.5, minor >= 15 %
 2. suspicious labels: leave-one-out own-centroid cos < 0.35 and best other centroid > own + 0.1
 3. near-duplicate recordings (cos > 0.97) shared by two different IDs (both files)
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
spk = np.array([s for s, _ in tr])
E = l2n(np.load(cache_key("redimnet_M_mix", "trainall", list(rels), mode="full")))
u, inv = np.unique(spk, return_inverse=True)
n = np.bincount(inv)

# 1. mixed-voice IDs (KEEP_DOMINANT=1: keep the dominant 2-means cluster if it holds >= 60 %, else drop the ID)
import os
KEEP_DOMINANT = os.environ.get("KEEP_DOMINANT") == "1"
mixed, minor_utts = set(), set()
for k, s in enumerate(u):
    idx = np.where(inv == k)[0]
    if len(idx) < 8:
        continue
    km = KMeans(2, n_init=3, random_state=0).fit(E[idx])
    c = l2n(km.cluster_centers_)
    cnt = np.bincount(km.labels_)
    if c[0] @ c[1] < 0.5 and cnt.min() / len(idx) >= 0.15:
        if KEEP_DOMINANT and cnt.max() / len(idx) >= 0.6:
            minor_utts.update(rels[idx[km.labels_ != cnt.argmax()]])
        else:
            mixed.add(s)

# 2. suspicious labels (leave-one-out)
S = np.zeros((len(u), E.shape[1]))
np.add.at(S, inv, E)
C = l2n(S / n[:, None])
own = np.full(len(E), np.nan)
m = n[inv] > 1
own[m] = np.sum(l2n(S[inv[m]] - E[m]) * E[m], 1)
other = np.empty(len(E))
for s0 in range(0, len(E), 4096):
    sim = E[s0:s0 + 4096] @ C.T
    sim[np.arange(len(sim)), inv[s0:s0 + 4096]] = -9
    other[s0:s0 + 4096] = sim.max(1)
susp = set(rels[m & (other > own + 0.1) & (own < 0.35)])

# 3. cross-ID near duplicates
dup = set()
for s0 in range(0, len(E), 4096):
    sim = E[s0:s0 + 4096] @ E.T
    i, j = np.where(sim > 0.97)
    for a, b in zip(s0 + i, j):
        if a < b and spk[a] != spk[b]:
            dup.update([rels[a], rels[b]])

out, st = [], dict(mixed_voice_ids=len(mixed), utts_mixed=0, utts_suspicious=0, utts_crossid_dup=0)
for s, r in tr:
    if s in mixed or r in minor_utts:
        st["utts_mixed"] += 1
    elif r in susp:
        st["utts_suspicious"] += 1
    elif r in dup:
        st["utts_crossid_dup"] += 1
    else:
        out.append(f"{s} {r}")
st.update(n_in=len(tr), n_out=len(out), spk_out=len({x.split()[0] for x in out}))
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "lists/train_drop_bad.txt"
dst.write_text("\n".join(out) + "\n")
print(json.dumps(st, indent=1))
