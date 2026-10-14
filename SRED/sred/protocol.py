"""Anti-overfitting evaluation protocol on the public test.

Public speakers are split (deterministic md5 hash) into two disjoint halves:
  P-dev  - used for every decision (recipe, back-end, comparisons)
  P-hold - looked at once, for the final confirmation (--reveal)
Trials whose two files come from different halves are dropped. The speaker IDs are taken from
the public directory names ONLY to define this split / bootstrap groups; systems never see them.

Decision rule: accept B over A only if the paired speaker-bootstrap P(B better) > 0.95 on P-dev.

  python -m sred.protocol exp/emb/redimnet_M_mix exp/emb/F03_M_s3_aug_bnfz_var_best
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .calibrate import QMF
from .data import cohort_subset, read_train_list, read_trials, unique_files
from .metrics import compute_eer, compute_min_dcf
from .scoring import l2n

ROOT = Path(__file__).resolve().parent.parent
DUR_BINS = [(0, 2), (2, 4), (4, 8), (8, 1e9)]


def _spk(f: str) -> str:
    return f.split("/")[0]


def _half(spk: str) -> int:
    return int(hashlib.md5(spk.encode()).hexdigest(), 16) % 2


class Protocol:
    def __init__(self, cohort_per_spk: int = 5):
        self.pairs, self.labels = read_trials()
        self.files = unique_files(self.pairs)
        pos = {f: i for i, f in enumerate(self.files)}
        self.ia = np.array([pos[a] for a, _ in self.pairs])
        self.ib = np.array([pos[b] for _, b in self.pairs])
        self.dur = np.load(ROOT / "lists/public_durations.npy")
        assert len(self.dur) == len(self.files)
        ha = np.array([_half(_spk(a)) for a, _ in self.pairs])
        hb = np.array([_half(_spk(b)) for _, b in self.pairs])
        self.mask = {"dev": (ha == 0) & (hb == 0), "hold": (ha == 1) & (hb == 1)}
        g = np.array([_spk(a) for a, _ in self.pairs])
        self.groups = {}
        for split, m in self.mask.items():
            idx = np.where(m)[0]
            _, inv = np.unique(g[idx], return_inverse=True)
            self.groups[split] = [idx[inv == k] for k in range(inv.max() + 1)]
        items = cohort_subset(read_train_list(), per_spk=cohort_per_spk)
        lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
        self.coh_spk = [s for s, _ in items]
        self.coh_dur = np.array([lengths[r] / 16000 for _, r in items])
        self.mind = np.minimum(self.dur[self.ia], self.dur[self.ib])

    def scores(self, emb_dir: Path) -> dict[str, np.ndarray]:
        emb = np.load(next(Path(emb_dir).glob("public_modefull_*.npy")))
        coh = np.load(next(Path(emb_dir).glob("traincohort5_modefull_*.npy")))
        e, c = l2n(emb), l2n(coh)
        cos = np.sum(e[self.ia] * e[self.ib], 1)
        qmf = QMF(use_asnorm=False).fit(c, self.coh_spk, self.coh_dur).score(e, self.dur, self.ia, self.ib)
        return {"cos": cos, "qmf": qmf}

    def eer(self, s, split, idx=None):
        idx = np.where(self.mask[split])[0] if idx is None else idx
        return compute_eer(s[idx], self.labels[idx])

    def report(self, s, split):
        idx = np.where(self.mask[split])[0]
        r = {"eer": round(self.eer(s, split), 3),
             "mindcf": round(compute_min_dcf(s[idx], self.labels[idx]), 4)}
        for lo, hi in DUR_BINS:
            j = idx[(self.mind[idx] >= lo) & (self.mind[idx] < hi)]
            r[f"d{lo}-{hi if hi < 1e9 else 'inf'}"] = round(compute_eer(s[j], self.labels[j]), 2)
        return r

    def paired(self, sa, sb, split="dev", n=300, seed=0):
        """Speaker-bootstrap of EER(b) - EER(a). Returns mean diff, 95% CI, P(b better)."""
        rng = np.random.default_rng(seed)
        G = self.groups[split]
        d = []
        for _ in range(n):
            idx = np.concatenate([G[k] for k in rng.integers(0, len(G), len(G))])
            d.append(compute_eer(sb[idx], self.labels[idx]) - compute_eer(sa[idx], self.labels[idx]))
        d = np.array(d)
        full = self.eer(sb, split) - self.eer(sa, split)
        return dict(diff=round(full, 3), ci=[round(float(np.percentile(d, 2.5)), 3),
                    round(float(np.percentile(d, 97.5)), 3)], p_better=round(float((d < 0).mean()), 3))


def extract_system(model: str, P: "Protocol") -> Path:
    from .data import PUBLIC_DIR, TRAIN_DIR
    from .embed import cache_key, extract_cached
    from .models import load_model
    fn = lambda: load_model(model)  # noqa: E731
    extract_cached(model, fn, "public", P.files, PUBLIC_DIR, mode="full")
    items = cohort_subset(read_train_list(), per_spk=5)
    rels = [r for _, r in items]
    extract_cached(model, fn, "traincohort5", rels, TRAIN_DIR, mode="full")
    return cache_key(model, "public", P.files, mode="full").parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dirs", nargs="+", help="exp/emb/<system> dirs; first one is the reference")
    ap.add_argument("--reveal", action="store_true", help="also report P-hold (use once, at the end)")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--extract", action="store_true",
                    help="emb_dirs are model names / ckpt paths: extract (cached) embeddings first")
    args = ap.parse_args(argv)
    P = Protocol()
    if args.extract:
        args.emb_dirs = [str(extract_system(m, P)) for m in args.emb_dirs]
    print(f"P-dev trials {P.mask['dev'].sum()} ({len(P.groups['dev'])} spk), "
          f"P-hold trials {P.mask['hold'].sum()} ({len(P.groups['hold'])} spk)")
    splits = ["dev", "hold"] if args.reveal else ["dev"]
    S = {Path(d).name: P.scores(Path(d)) for d in args.emb_dirs}
    ref = Path(args.emb_dirs[0]).name
    for name, sc in S.items():
        for b, s in sc.items():
            for sp in splits:
                print(json.dumps({"sys": name, "backend": b, "split": sp, **P.report(s, sp)}))
    for name, sc in S.items():
        if name == ref:
            continue
        for b in sc:
            for sp in splits:
                print(json.dumps({"cmp": f"{name} vs {ref}", "backend": b, "split": sp,
                                  **P.paired(S[ref][b], sc[b], sp, n=args.n_boot)}))


if __name__ == "__main__":
    main()
