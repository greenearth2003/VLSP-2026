"""Speaker-disjoint validation set V carved out of TRAIN (round v4).

V = 300 train speakers (seed 0, speakers with >= 6 utterances), at most 20 utterances each. Models evaluated
on V must be trained with `--exclude-val` (train list minus V speakers). Public test is not used here.

Each V file exists in two versions: L = full utterance, S = first 1.5 s (short-utterance condition).
Trials (fixed, seed 0): per speaker up to 30 target pairs (preferring distant file indices = other session),
same number of non-targets (60 % same-gender by audio F0, like the public list), each trial assigned
uniformly to SS / SL / LL.

  python -m sred.valset <model or ckpt>[::wise=A] ...   (first system = reference; paired speaker bootstrap)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .calibrate import QMF
from .data import TRAIN_DIR, cohort_subset, read_train_list
from .metrics import compute_eer, compute_min_dcf
from .scoring import l2n

ROOT = Path(__file__).resolve().parent.parent
N_SPK, MAX_UTT, SHORT_S = 300, 20, 1.5
VAL_FILE = ROOT / "lists/val_v4.json"


def build(seed: int = 0) -> dict:
    if VAL_FILE.exists():
        return json.loads(VAL_FILE.read_text())
    from .vad import file_info_cached
    rng = np.random.default_rng(seed)
    items = read_train_list()
    by = defaultdict(list)
    for s, r in items:
        by[s].append(r)
    cand = sorted(s for s, rs in by.items() if len(rs) >= 6)
    spk = sorted(rng.choice(cand, N_SPK, replace=False).tolist())
    files, fspk = [], []
    for s in spk:
        rs = sorted(by[s])
        if len(rs) > MAX_UTT:
            rs = sorted(rng.choice(rs, MAX_UTT, replace=False).tolist())
        files += rs
        fspk += [s] * len(rs)
    fspk = np.array(fspk)
    f0 = file_info_cached(files, TRAIN_DIR, "valv4")[:, 1]
    spk_f0 = {s: np.nanmedian(f0[fspk == s]) for s in spk}
    gender = {s: ("M" if spk_f0[s] < 165 else "F") for s in spk}
    idx_of = {f: i for i, f in enumerate(files)}
    num = lambda f: int(f.split("/")[1][:-4])  # noqa: E731
    tgt = []
    for s in spk:
        fs = [f for f in files if f.startswith(s + "/")]
        pairs = [(a, b) for i, a in enumerate(fs) for b in fs[i + 1:]]
        pairs.sort(key=lambda p: -abs(num(p[0]) - num(p[1])))  # distant indices first (cross-session proxy)
        tgt += pairs[:30]
    non = []
    by_g = {g: [f for f in files if gender[f.split("/")[0]] == g] for g in "MF"}
    while len(non) < len(tgt):
        a = files[rng.integers(len(files))]
        sa = a.split("/")[0]
        pool = by_g[gender[sa]] if rng.random() < 0.6 else files
        b = pool[rng.integers(len(pool))]
        if b.split("/")[0] != sa:
            non.append((a, b))
    trials = []
    for (a, b), y in [(p, 1) for p in tgt] + [(p, 0) for p in non]:
        cond = ["SS", "SL", "LL"][rng.integers(3)]
        va, vb = ("S" if cond != "LL" else "L"), ("S" if cond == "SS" else "L")
        if rng.random() < 0.5:
            a, b = b, a
        trials.append([idx_of[a], va, idx_of[b], vb, y, cond])
    out = dict(speakers=spk, files=files, trials=trials, short_s=SHORT_S)
    VAL_FILE.write_text(json.dumps(out))
    return out


def val_speakers() -> set:
    return set(build()["speakers"])


def train_items_minus_val(items):
    vs = val_speakers()
    return [(s, r) for s, r in items if r.split("/")[0] not in vs]


class ValSet:
    def __init__(self):
        v = build()
        self.files, self.spk = v["files"], v["speakers"]
        t = v["trials"]
        self.n = len(self.files)
        import pandas as pd  # drop trials of mixed-voice speakers (train EDA flags; evaluation-only)
        tb = pd.read_csv(ROOT / "eda/out/train_spk_bimodality.csv")
        mixed = set(tb[(tb.cos_between_2clusters < 0.5) & (tb.minor_frac >= 0.15)].spk)
        sp = lambda i: self.files[i].split("/")[0]  # noqa: E731
        t = [x for x in t if sp(x[0]) not in mixed and sp(x[2]) not in mixed]
        # embedding index: L versions are 0..n-1, S versions n..2n-1
        self.ia = np.array([x[0] + (self.n if x[1] == "S" else 0) for x in t])
        self.ib = np.array([x[2] + (self.n if x[3] == "S" else 0) for x in t])
        self.y = np.array([x[4] for x in t])
        self.cond = np.array([x[5] for x in t])
        lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
        dL = np.array([lengths[f] / 16000 for f in self.files])
        self.dur = np.r_[dL, np.minimum(dL, SHORT_S)]
        g = np.array([self.files[x[0]].split("/")[0] for x in t])
        _, inv = np.unique(g, return_inverse=True)
        self.groups = [np.where(inv == k)[0] for k in range(inv.max() + 1)]
        coh = cohort_subset(train_items_minus_val(read_train_list()), per_spk=5)
        self.coh_rels, self.coh_spk = [r for _, r in coh], [s for s, _ in coh]
        self.coh_dur = np.array([lengths[r] / 16000 for r in self.coh_rels])

    def embeddings(self, spec: str):
        from .embed import extract_cached
        from .models import load_model
        model, *opts = spec.split("::")
        fn = lambda: load_model(model)  # noqa: E731
        kwL = dict(mode="full+crop", crop_sec=3.0, hop_sec=1.5) if "mc" in opts else dict(mode="full")
        eL = extract_cached(model, fn, "valv4", self.files, TRAIN_DIR, **kwL)
        eS = extract_cached(model, fn, "valv4", self.files, TRAIN_DIR, mode="full", max_sec=SHORT_S)
        coh = extract_cached(model, fn, "traincohort5_minusval", self.coh_rels, TRAIN_DIR, mode="full")
        return l2n(np.r_[eL, eS]), l2n(coh)

    def scores(self, model: str) -> dict:
        e, c = self.embeddings(model)
        cos = np.sum(e[self.ia] * e[self.ib], 1)
        qmf = QMF(use_asnorm=False).fit(c, self.coh_spk, self.coh_dur).score(e, self.dur, self.ia, self.ib)
        return {"cos": cos, "qmf": qmf}

    def report(self, s) -> dict:
        r = {"eer": round(compute_eer(s, self.y), 3), "mindcf": round(compute_min_dcf(s, self.y), 4)}
        for c in ["SS", "SL", "LL"]:
            m = self.cond == c
            r[c] = round(compute_eer(s[m], self.y[m]), 2)
        return r

    def paired(self, sa, sb, n=300, seed=0) -> dict:
        rng = np.random.default_rng(seed)
        d = []
        for _ in range(n):
            idx = np.concatenate([self.groups[k] for k in rng.integers(0, len(self.groups), len(self.groups))])
            d.append(compute_eer(sb[idx], self.y[idx]) - compute_eer(sa[idx], self.y[idx]))
        d = np.array(d)
        return dict(diff=round(compute_eer(sb, self.y) - compute_eer(sa, self.y), 3),
                    ci=[round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                    p_better=round(float((d < 0).mean()), 3))


def _name(spec: str) -> str:
    m, *opts = spec.split("::")
    base = f"{Path(m).parent.name}_{Path(m).stem}" if "/" in m else m
    return base + "".join(f"+{o}" for o in opts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    V = ValSet()
    print(f"V: {len(V.spk)} spk, {V.n} files, {len(V.y)} trials ({V.y.mean():.2f} target)")
    S = {_name(m): V.scores(m) for m in args.systems}
    ref = _name(args.systems[0])
    lines = [{"val": "v4", "clean": True, "sys": k, "backend": b, **V.report(s)} for k, sc in S.items() for b, s in sc.items()]
    lines += [{"val": "v4", "clean": True, "cmp": f"{k} vs {ref}", "backend": b, **V.paired(S[ref][b], s, n=args.n_boot)}
              for k, sc in S.items() if k != ref for b, s in sc.items()]
    for ln in lines:
        print(json.dumps(ln))
    if args.out:
        with open(args.out, "a") as f:
            f.writelines(json.dumps(ln) + "\n" for ln in lines)


if __name__ == "__main__":
    main()
