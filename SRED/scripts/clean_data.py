"""Model-assisted label cleaning of train and public test.

    python scripts/clean_data.py --models ext_a ext_b exp/ft/W_G05_B6/avg.pt [--extract-only]

Uses an ensemble of speaker models (incl. EXTERNAL, non-approved ones -- see sred/models.py) to flag:
  1. files with (almost) no speech
  2. byte-identical duplicates (within train, train<->public leakage)
  3. outlier files that do not match their own speaker folder
  4. folders that contain two (or more) distinct voices
  5. different folders that are the same person (duplicate identities)
Outputs in lists/clean/: train_clean.txt (same format as train.txt, merged IDs), public_trials_clean.txt
(questionable trials removed, labels untouched), public_identity.json (folder -> identity), report.json.
Thresholds are calibrated from the score distributions of the data itself and printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import PUBLIC_DIR, PUBLIC_TRIALS, TRAIN_DIR, read_train_list, read_trials, unique_files
from sred.embed import extract_cached
from sred.models import load_model
from sred.scoring import l2n

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lists/clean"
A = ROOT / "exp/analysis"


def ensemble(models, split, files, root):
    """Per-model: L2-norm, centre on the split mean, L2-norm again; concatenate / sqrt(n) so that
    the dot product is the mean of per-model cosines."""
    embs = []
    for m in models:
        e = l2n(extract_cached(m, lambda m=m: load_model(m), split, files, root, mode="full"))
        e = l2n(e - e.mean(0))
        embs.append(e)
    return np.concatenate(embs, 1) / np.sqrt(len(embs))


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def analyse(E, spk, files, feats, name, args):
    """Returns dict with outlier files, split folders, identity merges for one split."""
    spk = np.asarray(spk)
    U = sorted(set(spk))
    idx = {u: np.flatnonzero(spk == u) for u in U}
    C = l2n(np.stack([E[idx[u]].mean(0) for u in U]))
    ui = {u: i for i, u in enumerate(U)}

    # --- 5 first. duplicate identities: centroid sim above the extreme tail of cross-folder sims
    iu = np.triu_indices(len(U), 1)
    cc = (C @ C.T)[iu]
    th_merge = float(np.quantile(cc, args.merge_q))
    big = np.array([len(idx[u]) for u in U])
    merges, uf = [], UF()
    for k in np.flatnonzero(cc > th_merge):
        a, b = iu[0][k], iu[1][k]
        merges.append(dict(a=U[a], b=U[b], sim=float(cc[k]), na=int(big[a]), nb=int(big[b])))
        uf.union(U[a], U[b])
    print(f"[{name}] cross-folder centroid sim q50 {np.median(cc):.3f} q99.9 {np.quantile(cc, .999):.3f}; "
          f"merge threshold (q{args.merge_q}) {th_merge:.3f} -> {len(merges)} folder pairs")

    # --- 3. outlier files: leave-one-out similarity to own folder vs best OTHER identity (after merging)
    own = np.full(len(files), np.nan)
    for u in U:
        ii = idx[u]
        if len(ii) < 3:
            continue
        s = E[ii].sum(0)
        own[ii] = np.sum(E[ii] * l2n(s[None] - E[ii]), 1)
    ident = np.array([uf.find(s) for s in spk])
    IU = sorted(set(ident))
    iix = {u: i for i, u in enumerate(IU)}
    CI = l2n(np.stack([E[ident == u].mean(0) for u in IU]))
    other = E @ CI.T
    other[np.arange(len(files)), [iix[s] for s in ident]] = -9
    best_other = other.max(1)
    margin = own - best_other
    outlier = (own < args.own_min) | ((own < args.own_weak) & (margin < -args.margin))
    print(f"[{name}] file->own-folder sim median {np.nanmedian(own):.3f} (q01 {np.nanquantile(own, .01):.3f}); "
          f"best-other-identity median {np.median(best_other):.3f}; outlier = own<{args.own_min} or "
          f"(own<{args.own_weak} and margin<-{args.margin}) -> {int(outlier.sum())} outlier files")

    # --- 4. multi-voice folders
    split_rows = []
    for u in U:
        ii = idx[u]
        if len(ii) < 4:
            continue
        cl = AgglomerativeClustering(2, metric="cosine", linkage="average").fit_predict(E[ii])
        n0, n1 = np.bincount(cl, minlength=2)
        if min(n0, n1) < 2:
            continue
        c0, c1 = l2n(E[ii][cl == 0].mean(0, keepdims=True)), l2n(E[ii][cl == 1].mean(0, keepdims=True))
        split_rows.append(dict(spk=u, n=len(ii), n_minor=int(min(n0, n1)), between=float(c0 @ c1.T),
                               minor_files=[files[i] for i in ii[cl == (0 if n0 < n1 else 1)]]))
    sp = pd.DataFrame(split_rows)
    # same-person centroid pairs score far above impostor centroid pairs; a folder whose two halves are
    # less similar than a typical impostor *file* pair margin is two people
    th_split = float(np.quantile(cc, args.split_q))
    multi = sp[sp.between < th_split] if len(sp) else sp
    print(f"[{name}] folder 2-cluster between-sim median {sp.between.median() if len(sp) else float('nan'):.3f}; "
          f"split threshold (q{args.split_q} of cross-folder centroid sim) {th_split:.3f} -> {len(multi)} multi-voice folders")

    return dict(own=own, best_other=best_other, outlier=outlier, multi=multi, merges=merges,
                th=dict(split=th_split, merge=th_merge))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="models for the public test")
    ap.add_argument("--train-models", nargs="+", default=None,
                    help="models for train (default --models); use models NOT fine-tuned on train, so noisy "
                         "train labels are not memorised")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--own-min", type=float, default=0.2, help="flag file if sim to own folder (LOO) below this")
    ap.add_argument("--own-weak", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.1, help="... or own<own-weak and another identity is closer by this")
    ap.add_argument("--split-q", type=float, default=0.99, help="quantile of cross-folder centroid sim")
    ap.add_argument("--merge-q", type=float, default=0.9999, help="quantile of cross-folder centroid sim")
    ap.add_argument("--min-speech", type=float, default=0.3)
    args = ap.parse_args()

    items = read_train_list()
    tr_files = [r for _, r in items]
    tr_spk = [s for s, _ in items]
    pairs, labels = read_trials()
    pu_files = unique_files(pairs)
    pu_spk = [f.split("/")[0] for f in pu_files]

    args.train_models = args.train_models or args.models
    for m in args.train_models:  # extraction (cached)
        extract_cached(m, lambda m=m: load_model(m), "trainfull", tr_files, TRAIN_DIR, mode="full")
    for m in args.models:
        extract_cached(m, lambda m=m: load_model(m), "public", pu_files, PUBLIC_DIR, mode="full")
    if args.extract_only:
        return
    Etr = ensemble(args.train_models, "trainfull", tr_files, TRAIN_DIR)
    Epu = ensemble(args.models, "public", pu_files, PUBLIC_DIR)

    feats = pd.concat([pd.read_csv(A / "features_train.csv"), pd.read_csv(A / "features_public.csv")])
    sp_dur = dict(zip(feats.split + "/" + feats.rel, feats.speech_dur))
    md5 = feats.groupby("md5").agg(rels=("rel", list), splits=("split", list))

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"public_models": args.models, "train_models": args.train_models, "args": vars(args)}

    # ======================= TRAIN
    tr = analyse(Etr, tr_spk, tr_files, feats, "train", args)
    drop = {}
    for i, f in enumerate(tr_files):
        if sp_dur.get("train/" + f, 1) < args.min_speech:
            drop[f] = "no_speech"
    for _, r in md5.iterrows():  # byte-identical duplicates
        if len(r.rels) < 2:
            continue
        tr_rels = [x for x, s in zip(r.rels, r.splits) if s == "train"]
        if "public" in r.splits:
            for x in tr_rels:
                drop[x] = "identical_to_public"
        else:
            for x in tr_rels[1:]:
                drop.setdefault(x, "duplicate_file")
    for i in np.flatnonzero(tr["outlier"]):
        drop.setdefault(tr_files[i], "outlier")
    for _, r in tr["multi"].iterrows():
        for f in r.minor_files:  # keep the dominant voice, drop the minority cluster
            drop.setdefault(f, "minor_voice_in_folder")
    uf = UF()
    for m in tr["merges"]:
        uf.union(m["a"], m["b"])
    kept = [(uf.find(s), r) for s, r in items if r not in drop]
    (OUT / "train_clean.txt").write_text("".join(f"{s}\t{r}\n" for s, r in kept))
    n_ids = len({s for s, _ in kept})
    reasons = pd.Series(drop).value_counts().to_dict()
    print(f"[train] dropped {len(drop)} / {len(items)} files {reasons}; speakers {len(set(tr_spk))} -> {n_ids} "
          f"after merging {len(tr['merges'])} duplicate-identity pairs")
    report["train"] = dict(dropped=len(drop), reasons=reasons, n_files=len(kept), n_speakers=n_ids,
                           thresholds=tr["th"], merges=tr["merges"],
                           multi_voice=tr["multi"].drop(columns="minor_files").to_dict("records"))
    pd.Series(drop, name="reason").rename_axis("file").to_csv(OUT / "train_dropped.csv")

    # ======================= PUBLIC (labels untouched; questionable trials removed for local eval)
    pu = analyse(Epu, pu_spk, pu_files, feats, "public", args)
    bad = {f for i, f in enumerate(pu_files) if pu["outlier"][i] or sp_dur.get("public/" + f, 1) < args.min_speech}
    for _, r in pu["multi"].iterrows():
        bad |= set(r.minor_files)
    multi_spk = set(pu["multi"].spk) if len(pu["multi"]) else set()
    uf = UF()
    for m in pu["merges"]:
        uf.union(m["a"], m["b"])
    ident = {s: uf.find(s) for s in sorted(set(pu_spk))}
    keep, why = [], defaultdict(int)
    for (a, b), y in zip(pairs, labels):
        sa, sb = a.split("/")[0], b.split("/")[0]
        if a in bad or b in bad:
            why["file_flagged"] += 1
        elif y == 0 and ident[sa] == ident[sb]:
            why["nontarget_but_same_identity"] += 1
        elif y == 0 and sa == sb:
            why["nontarget_same_folder"] += 1
        else:
            keep.append(f"{a} {b} {'target' if y else 'nontarget'}\n")
    (OUT / "public_trials_clean.txt").write_text("".join(keep))
    (OUT / "public_identity.json").write_text(json.dumps(ident, indent=0))
    print(f"[public] flagged files {len(bad)}; trials kept {len(keep)} / {len(pairs)} removed {dict(why)}")
    report["public"] = dict(flagged_files=len(bad), trials_kept=len(keep), removed=dict(why), thresholds=pu["th"],
                            merges=pu["merges"], multi_voice=pu["multi"].drop(columns="minor_files").to_dict("records"))
    pd.Series(sorted(bad)).to_csv(OUT / "public_flagged_files.csv", index=False, header=["file"])
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
