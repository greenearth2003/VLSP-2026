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


def label_noise_sets():
    """Public IDs flagged in the EDA (evaluation-only): mixed-voice IDs and ID pairs that are the same voice."""
    import pandas as pd
    out = ROOT / "eda/out"
    b = pd.read_csv(out / "public_spk_bimodality.csv")
    mixed = set(b[(b.cos_between_2clusters < 0.5) & (b.minor_frac >= 0.15)].spk)
    d = pd.read_csv(out / "duplicate_identities.csv")
    d = d[d.kind == "public-public"]
    same = {(a, c) for a, c in zip(d.spk1, d.spk2)} | {(c, a) for a, c in zip(d.spk1, d.spk2)}
    return mixed, same


class Protocol:
    def __init__(self, cohort_per_spk: int = 5, clean: bool = True):
        self.clean = clean
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
        if clean:  # drop trials touching label-noise IDs (EDA): mixed-voice IDs, same-voice ID pairs
            mixed, same = label_noise_sets()
            ok = np.array([_spk(a) not in mixed and _spk(b) not in mixed and (_spk(a), _spk(b)) not in same
                           for a, b in self.pairs])
            self.mask = {k: m & ok for k, m in self.mask.items()}
        g = np.array([_spk(a) for a, _ in self.pairs])
        self.groups = {}
        for split, m in self.mask.items():
            idx = np.where(m)[0]
            _, inv = np.unique(g[idx], return_inverse=True)
            self.groups[split] = [idx[inv == k] for k in range(inv.max() + 1)]
        items = cohort_subset(read_train_list(), per_spk=cohort_per_spk)
        lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
        self.coh_spk = [s for s, _ in items]
        self.coh_rels = [r for _, r in items]
        self.coh_dur = np.array([lengths[r] / 16000 for _, r in items])
        self.mind = np.minimum(self.dur[self.ia], self.dur[self.ib])

    def scores(self, spec: str) -> dict[str, np.ndarray]:
        """spec = '<model name or ckpt>[::vad][::f0]'. Embeddings must be cached (see extract_system)."""
        from .data import PUBLIC_DIR, TRAIN_DIR
        from .embed import cache_key
        from .vad import file_info_cached
        model, opts = parse_spec(spec)
        kw = dict(mode="full", **({"vad": True} if "vad" in opts else {}))
        e = l2n(np.load(cache_key(model, "public", self.files, **kw)))
        c = l2n(np.load(cache_key(model, "traincohort5", self.coh_rels, **kw)))
        cos = np.sum(e[self.ia] * e[self.ib], 1)
        dur, coh_dur, f0p, f0c, sdp, sdc = self.dur, self.coh_dur, None, None, None, None
        coh_spk, fid = self.coh_spk, None
        if "qs" in opts:  # QMF fitted on full + 1.5 s crops of the train cohort (short-duration calibration)
            cs = l2n(np.load(cache_key(model, "traincohort5", self.coh_rels, mode="full", max_sec=1.5)))
            c = np.r_[c, cs]
            coh_dur = np.r_[coh_dur, np.minimum(coh_dur, 1.5)]
            coh_spk = list(self.coh_spk) * 2
            fid = np.r_[np.arange(len(self.coh_rels)), np.arange(len(self.coh_rels))]
        if opts - {"hn", "qs"}:
            ip = file_info_cached(self.files, PUBLIC_DIR, "public")
            ic = file_info_cached(self.coh_rels, TRAIN_DIR, "traincohort5")
            if "vad" in opts:
                dur, coh_dur = ip[:, 0], ic[:, 0]
            if "f0" in opts:
                f0p, f0c = ip[:, 1], ic[:, 1]
            if "sdur" in opts:
                sdp, sdc = np.maximum(ip[:, 0], 0.1), np.maximum(ic[:, 0], 0.1)
        qmf = QMF(use_asnorm=False).fit(c, coh_spk, coh_dur, f0=f0c, sdur=sdc, hard_neg="hn" in opts, fid=fid) \
            .score(e, dur, self.ia, self.ib, f0=f0p, sdur=sdp)
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


def parse_spec(spec: str) -> tuple[str, set]:
    model, *opts = spec.split("::")
    assert set(opts) <= {"vad", "f0", "sdur", "hn", "qs"}, opts
    return model, set(opts)


def system_name(spec: str) -> str:
    model, opts = parse_spec(spec)
    name = f"{Path(model).parent.name}_{Path(model).stem}" if "/" in model else model
    return name + "".join(f"+{o}" for o in sorted(opts))


def extract_system(spec: str, P: "Protocol") -> None:
    from .data import PUBLIC_DIR, TRAIN_DIR
    from .embed import extract_cached
    from .models import load_model
    model, opts = parse_spec(spec)
    kw = dict(mode="full", **({"vad": True} if "vad" in opts else {}))
    fn = lambda: load_model(model)  # noqa: E731
    extract_cached(model, fn, "public", P.files, PUBLIC_DIR, **kw)
    extract_cached(model, fn, "traincohort5", P.coh_rels, TRAIN_DIR, **kw)
    if "qs" in opts:
        extract_cached(model, fn, "traincohort5", P.coh_rels, TRAIN_DIR, mode="full", max_sec=1.5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="'<model or ckpt>[::vad][::f0]'; the first one is the reference")
    ap.add_argument("--reveal", action="store_true", help="also report P-hold (use once, at the end)")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--extract", action="store_true", help="extract (cached) embeddings first")
    ap.add_argument("--out", default=None, help="append JSON lines here")
    ap.add_argument("--noisy", action="store_true", help="keep trials of label-noise IDs (old behaviour)")
    args = ap.parse_args(argv)
    P = Protocol(clean=not args.noisy)
    if args.extract:
        for sp in args.systems:
            extract_system(sp, P)
    print(f"P-dev trials {P.mask['dev'].sum()} ({len(P.groups['dev'])} spk), "
          f"P-hold trials {P.mask['hold'].sum()} ({len(P.groups['hold'])} spk)")
    splits = ["dev", "hold"] if args.reveal else ["dev"]
    S = {system_name(sp): P.scores(sp) for sp in args.systems}
    ref = system_name(args.systems[0])
    lines = []
    for name, sc in S.items():
        for b, s in sc.items():
            for sp in splits:
                lines.append({"sys": name, "backend": b, "split": sp, "clean": P.clean, **P.report(s, sp)})
    for name, sc in S.items():
        if name == ref:
            continue
        for b in sc:
            for sp in splits:
                lines.append({"cmp": f"{name} vs {ref}", "backend": b, "split": sp, "clean": P.clean,
                              **P.paired(S[ref][b], sc[b], sp, n=args.n_boot)})
    for ln in lines:
        print(json.dumps(ln))
    if args.out:
        with open(args.out, "a") as f:
            for ln in lines:
                f.write(json.dumps(ln) + "\n")


if __name__ == "__main__":
    main()
