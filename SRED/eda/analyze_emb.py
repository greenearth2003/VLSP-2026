"""EDA part 2: embedding-space analysis with the *pretrained* ReDimNet-M (never trained on VLSP labels).

- speaker separability (train vs public), intra-speaker spread
- suspicious labels in train (utterance closer to another speaker than to its own)
- likely duplicate identities (two speaker IDs, same voice) within train, within public, train<->public
- near-duplicate recordings (cos > 0.97 between different files)
- are W01 false alarms explained by duplicate identities in public?
Writes figures to docs/eda_figs/ and numbers to eda/out/emb_summary.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sred.data import read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.metrics import compute_eer  # noqa: E402
from sred.scoring import l2n  # noqa: E402

FIG = ROOT / "docs/eda_figs"
OUT = ROOT / "eda/out"
DUP_THR = 0.75  # centroid cosine above which two IDs are flagged as the same voice
res = {}

tr = read_train_list()
tr_files = [r for _, r in tr]
tr_spk = np.array([s for s, _ in tr])
E = l2n(np.load(cache_key("redimnet_M_mix", "trainall", tr_files, mode="full")))
pairs, lab = read_trials()
pfiles = unique_files(pairs)
Pm = l2n(np.load(cache_key("redimnet_M_mix", "public", pfiles, mode="full")))
p_spk = np.array([f.split("/")[0] for f in pfiles])
dur_tr = np.array([json.loads((ROOT / "lists/train_lengths.json").read_text())[r] / 16000 for r in tr_files])


def centroids(X, spk):
    u, inv = np.unique(spk, return_inverse=True)
    S = np.zeros((len(u), X.shape[1]))
    np.add.at(S, inv, X)
    n = np.bincount(inv)
    return u, inv, S, n


# ---------------------------------------------------------------- suspicious labels (leave-one-out)
u, inv, S, n = centroids(E, tr_spk)
C = l2n(S / n[:, None])
own = np.full(len(E), np.nan)
m = n[inv] > 1
loo = S[inv[m]] - E[m]
own[m] = np.sum(l2n(loo) * E[m], 1)
other = np.empty(len(E))
best_other = np.empty(len(E), dtype=int)
for s in range(0, len(E), 4096):
    sim = E[s:s + 4096] @ C.T
    sim[np.arange(len(sim)), inv[s:s + 4096]] = -9
    other[s:s + 4096] = sim.max(1)
    best_other[s:s + 4096] = sim.argmax(1)
susp = m & (other > own + 0.1) & (own < 0.35)
res["train_label_noise"] = dict(
    n_utts_scored=int(m.sum()), suspicious=int(susp.sum()), suspicious_frac=float(susp.sum() / m.sum()),
    own_cos_p5=float(np.nanpercentile(own, 5)), own_cos_p50=float(np.nanmedian(own)),
    spk_with_ge20pct_suspicious=int(sum(1 for k in range(len(u)) if n[k] >= 5 and susp[inv == k].mean() >= 0.2)),
    suspicious_dur_p50=float(np.median(dur_tr[susp])) if susp.any() else None, all_dur_p50=float(np.median(dur_tr)))
pd.DataFrame(dict(rel=np.array(tr_files)[susp], spk=tr_spk[susp], own_cos=own[susp], best_other_spk=u[best_other[susp]],
                  other_cos=other[susp], dur=dur_tr[susp])).sort_values("own_cos").to_csv(OUT / "train_suspicious_labels.csv", index=False)

# ---------------------------------------------------------------- separability + spread
rng = np.random.default_rng(0)


def synthetic_eer(X, spk, n_non=200000):
    uu, ii = np.unique(spk, return_inverse=True)
    order = np.argsort(ii, kind="stable")
    tgt = []
    for k in range(len(order) - 1):
        a, b = order[k], order[k + 1]
        if ii[a] == ii[b]:
            tgt.append((a, b))
    tgt = np.array(tgt)
    I, J = rng.integers(0, len(X), n_non), rng.integers(0, len(X), n_non)
    keep = ii[I] != ii[J]
    st = np.sum(X[tgt[:, 0]] * X[tgt[:, 1]], 1)
    sn = np.sum(X[I[keep]] * X[J[keep]], 1)
    return compute_eer(np.r_[st, sn], np.r_[np.ones(len(st)), np.zeros(len(sn))]), st, sn


eer_tr, st_tr, sn_tr = synthetic_eer(E, tr_spk)
pos = {f: i for i, f in enumerate(pfiles)}
ia = np.array([pos[a] for a, _ in pairs])
ib = np.array([pos[b] for _, b in pairs])
s_pub = np.sum(Pm[ia] * Pm[ib], 1)
res["separability_pretrainedM"] = dict(
    train_synthetic_eer=eer_tr, public_trials_eer=compute_eer(s_pub, lab),
    train_target_cos_p50=float(np.median(st_tr)), train_nontarget_cos_p50=float(np.median(sn_tr)),
    public_target_cos_p50=float(np.median(s_pub[lab == 1])), public_nontarget_cos_p50=float(np.median(s_pub[lab == 0])))
fig, ax = plt.subplots(figsize=(8, 4))
b = np.linspace(-0.4, 1, 100)
ax.hist(st_tr, b, density=True, histtype="step", lw=1.5, label="train target")
ax.hist(sn_tr, b, density=True, histtype="step", lw=1.5, label="train nontarget")
ax.hist(s_pub[lab == 1], b, density=True, alpha=0.4, label="public target")
ax.hist(s_pub[lab == 0], b, density=True, alpha=0.4, label="public nontarget")
ax.set_xlabel("cosine (ReDimNet-M pretrained)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "05_cosine_train_vs_public.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- duplicate identities
pu_u, pu_inv, PS, pn = centroids(Pm, p_spk)
PC = l2n(PS / pn[:, None])


def dup_pairs(A, ua, B=None, ub=None, min_n=None):
    same = B is None
    B, ub = (A, ua) if same else (B, ub)
    sim = A @ B.T
    if same:
        sim[np.tril_indices(len(A))] = -9
    i, j = np.where(sim > DUP_THR)
    return [(ua[a], ub[b], round(float(sim[a, b]), 3)) for a, b in zip(i, j)], sim


tt, sim_tt = dup_pairs(C, u)
pp, sim_pp = dup_pairs(PC, pu_u)
tp, sim_tp = dup_pairs(PC, pu_u, C, u)
nn_tt = np.where(np.eye(len(C), dtype=bool), -9, C @ C.T).max(1)
res["duplicate_identities"] = dict(threshold=DUP_THR, train_train=len(tt), public_public=len(pp), public_train=len(tp),
                                   public_spk_with_train_match=len({a for a, _, _ in tp}),
                                   train_nn_centroid_cos_p50=float(np.median(nn_tt)), train_nn_centroid_cos_p99=float(np.percentile(nn_tt, 99)),
                                   public_to_train_nn_cos_p50=float(np.median(sim_tp.max(1))),
                                   examples_train=tt[:10], examples_public=pp[:10], examples_public_train=tp[:10])
pd.DataFrame(tt + pp + tp, columns=["spk1", "spk2", "centroid_cos"]).assign(
    kind=["train-train"] * len(tt) + ["public-public"] * len(pp) + ["public-train"] * len(tp)).to_csv(OUT / "duplicate_identities.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(nn_tt, np.linspace(0, 1, 80), alpha=0.6, density=True, label="train: speaker gần nhất (khác ID)")
ax.hist(sim_tp.max(1), np.linspace(0, 1, 80), alpha=0.6, density=True, label="public → train gần nhất")
ax.axvline(DUP_THR, color="k", ls="--")
ax.set_xlabel("cosine giữa centroid speaker")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "06_nearest_speaker_centroid.png", dpi=110)
plt.close(fig)

# do duplicate public identities explain false alarms of W01?
t = pd.read_csv(OUT / "public_trials_annotated.csv")
dup_set = {(a, b) for a, b, _ in pp} | {(b, a) for a, b, _ in pp}
non = t[t.y == 0]
is_dup = np.array([(a, b) in dup_set for a, b in zip(non.spk_a, non.spk_b)])
res["fa_vs_duplicate_public_ids"] = dict(nontarget_trials_between_dup_ids=int(is_dup.sum()),
                                         fa_rate_dup=float(non[is_dup].err.mean()) if is_dup.any() else None,
                                         fa_rate_rest=float(non[~is_dup].err.mean()),
                                         share_of_all_fa=float(non[is_dup].err.sum() / non.err.sum()))

# ---------------------------------------------------------------- near-duplicate recordings
nd = []
for s in range(0, len(E), 4096):
    sim = E[s:s + 4096] @ E.T
    sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -9
    i, j = np.where(sim > 0.97)
    nd += [(s + a, b) for a, b in zip(i, j) if s + a < b]
nd = np.array(nd) if nd else np.zeros((0, 2), int)
res["near_duplicate_train_utts"] = dict(pairs=int(len(nd)), cross_speaker=int((tr_spk[nd[:, 0]] != tr_spk[nd[:, 1]]).sum()) if len(nd) else 0)
pd.DataFrame(dict(a=np.array(tr_files)[nd[:, 0]], b=np.array(tr_files)[nd[:, 1]],
                  same_spk=tr_spk[nd[:, 0]] == tr_spk[nd[:, 1]])).to_csv(OUT / "train_near_duplicates.csv", index=False)

# ---------------------------------------------------------------- spread vs data per speaker
spread = 1 - np.array([np.mean(E[inv == k] @ C[k]) for k in range(len(u))])
res["intra_speaker_spread"] = dict(corr_with_n_utts=float(pd.Series(spread).corr(pd.Series(n), method="spearman")),
                                   spread_p50=float(np.median(spread)))

(OUT / "emb_summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
