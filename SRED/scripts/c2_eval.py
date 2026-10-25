"""Scoring and calibration variants for the teammate's C2 (embeddings from scripts/c2_embed_all.py), judged on the
noisy stand-ins: silver (+ private-like impostor weighting), real noise, synthetic. EER all trials / cleaned list.

Raw scores: whole (their system) · tta (mean over TTA view pairs) · fv (whole + energy-trimmed average) ·
gated (fv inside our label-free gate, whole outside) · tta_fv (tta + trimmed).
Calibrations: Q3 = their frozen weights (whole only) · R = the same 9-feature logistic regression refitted here on
train-cohort trials over clean + synthetic-noise + real-noise copies (a clip never paired with its own copy) ·
R+n = R plus the min / max raw embedding norm (a quality measure).

    python scripts/c2_eval.py [--splits silver public_realnoise public_noisy] [--private VARIANT --zip NAME]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/mnt/tsharp/thucth/dataset/vlsp2026-sred/incoming/c2pkg")
import c2q3  # noqa: E402
from sred.data import cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402

E = ROOT / "exp/emb/ext_C2"  # overridden by --emb-dir
META = "/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv"


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def load(split):
    z = np.load(E / f"{split}_all.npz")
    return {k: z[k] for k in z.files}


def raw_scores(z, ia, ib, gate=None):
    W, T, V = l2n(z["whole"]), z["tta"], l2n(z["vad"])
    FV = l2n(W + V)
    TV = l2n(l2n(T) + V)
    s = {"whole": (W[ia] * W[ib]).sum(1), "tta": (T[ia] * T[ib]).sum(1), "fv": (FV[ia] * FV[ib]).sum(1),
         "tta_fv": (TV[ia] * TV[ib]).sum(1)}
    if gate is not None:
        s["gated"] = np.where(gate, s["fv"], s["whole"])
    return s


def feats(s, da, db, qa, qb, na=None, nb=None):
    a, b = np.log(np.minimum(da, db)), np.log(np.maximum(da, db))
    lo, hi = np.clip(np.minimum(qa, qb), -5, 40) / 10, np.clip(np.maximum(qa, qb), -5, 40) / 10
    f = [s, a, b, s * a, s * b, lo, hi, s * lo, s * hi]
    if na is not None:
        nlo, nhi = np.log(np.minimum(na, nb)), np.log(np.maximum(na, nb))
        f += [nlo, nhi, s * nlo, s * nhi]
    return np.stack(f, 1)


class Cal:
    def __init__(self, s, y, da, db, qa, qb, na=None, nb=None):
        X = feats(s, da, db, qa, qb, na, nb)
        self.norms = na is not None
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((X - self.mu) / self.sd, y)

    def __call__(self, s, da, db, qa, qb, na=None, nb=None):
        return self.lr.decision_function((feats(s, da, db, qa, qb, *((na, nb) if self.norms else ())) - self.mu) / self.sd)


def cohort_trials(n_non=500000, seed=0):
    """items = clean, synthetic-noise and real-noise copies of the cohort; same-speaker pairs from different source
    clips (any copies), random different-speaker pairs."""
    ci = cohort_subset(read_train_list(), per_spk=5)
    spk = np.array([s for s, _ in ci])
    n = len(spk)
    cop = [load(s) for s in ("traincohort5", "traincohort_noisy", "traincohort_realnoise")]
    Z = {k: np.concatenate([c[k] for c in cop]) for k in ("whole", "tta", "vad", "dur", "snr", "whole_norm")}
    src = np.tile(np.arange(n), 3)
    S = np.tile(spk, 3)
    rng = np.random.default_rng(seed)
    I, J = rng.integers(0, 3 * n, n_non), rng.integers(0, 3 * n, n_non)
    keep = S[I] != S[J]
    I, J = I[keep], J[keep]
    by = pd.Series(np.arange(3 * n)).groupby(S).apply(np.array)
    ti, tj = [], []
    for idx in by:
        for x in range(len(idx)):
            for w in range(x + 1, len(idx)):
                if src[idx[x]] != src[idx[w]]:
                    ti.append(idx[x]); tj.append(idx[w])
    I, J = np.concatenate([I, ti]), np.concatenate([J, tj])
    return Z, I, J, (S[I] == S[J]).astype(int), S


ASN = {}  # cohort embeddings for adaptive s-norm (train data only)


def topk_stats(e, k=300, spk=None, chunk=4096):
    C = ASN["C"]
    mu, sd = np.empty(len(e), np.float32), np.empty(len(e), np.float32)
    for s in range(0, len(e), chunk):
        S = e[s:s + chunk] @ C.T
        if spk is not None:
            S[spk[s:s + chunk, None] == ASN["spk"][None, :]] = -9
        top = np.partition(S, -k, axis=1)[:, -k:]
        mu[s:s + chunk], sd[s:s + chunk] = top.mean(1), top.std(1)
    return mu, sd


def asnorm(s, stats, ia, ib):
    mu, sd = stats
    return 0.5 * ((s - mu[ia]) / sd[ia] + (s - mu[ib]) / sd[ib])


def eer(s, y, w=None):
    o = np.argsort(s)
    yo = y[o]
    wo = np.ones(len(y)) if w is None else w[o]
    frr = np.cumsum(wo * yo) / (wo * yo).sum()
    far = 1 - np.cumsum(wo * (1 - yo)) / (wo * (1 - yo)).sum()
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2


def private_like_weights(pairs, y):
    import private_like_eer as PL
    fp = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel")
    fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv").set_index("rel")
    prv, _ = read_trials(ROOT / "submissions/final_private/trials_2col.txt")
    pn, ps, pg = PL.gaps(pairs, fp)
    vn, vs, vg = PL.gaps(prv, fv)
    en, es = np.quantile(vn, [0.2, 0.4, 0.6, 0.8]), np.quantile(vs, [0.25, 0.5, 0.75])
    bp, bv = PL.bins(pn, ps, pg, en, es), PL.bins(vn, vs, vg, en, es)
    nbins = PL.bins(np.array([1e9]), np.array([1e9]), np.array([1]), en, es)[0] + 1
    p_priv = np.bincount(bv, minlength=nbins) / len(bv)
    nt = y == 0
    p_pub = np.bincount(bp[nt], minlength=nbins) / nt.sum()
    w = np.ones(len(y))
    w[nt] = np.minimum(np.where(p_pub[bp[nt]] > 0, p_priv[bp[nt]] / np.maximum(p_pub[bp[nt]], 1e-12), 0), 20)
    return w


def gate_of(split, files, ia, ib):
    name = {"silver": "public_rumble_v2"}.get(split, split)
    f = pd.read_csv(ROOT / f"exp/analysis/features_{name}.csv").set_index("rel").loc[files]
    sh, sp, lv = f.speech_ratio.to_numpy(), f.speech_dur.to_numpy(), f.rms_db.to_numpy()
    return (np.minimum(sh[ia], sh[ib]) < 0.5) | (np.minimum(sp[ia], sp[ib]) < 1.0) | (np.abs(lv[ia] - lv[ib]) > 10)


def fit_all():
    Z, I, J, y, S = cohort_trials()
    ASN["C"], ASN["spk"] = l2n(Z["whole"]), S
    s = raw_scores(Z, I, J)
    s["whole_as"] = asnorm(s["whole"], topk_stats(l2n(Z["whole"]), spk=S), I, J)
    args = (Z["dur"][I], Z["dur"][J], Z["snr"][I], Z["snr"][J])
    nrm = (Z["whole_norm"][I], Z["whole_norm"][J])
    cals = {}
    for k in ("whole", "tta", "fv", "tta_fv", "whole_as"):
        cals[(k, "R")] = Cal(s[k], y, *args)
        cals[(k, "R+n")] = Cal(s[k], y, *args, *nrm)
    print(f"calibration trials: {len(y):,} ({y.sum():,} same speaker)", flush=True)
    return cals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["silver", "public_realnoise", "public_noisy"])
    ap.add_argument("--private", default=None, help="'<raw variant>:<calibration>' to score private, e.g. gated:R")
    ap.add_argument("--zip", default=None)
    ap.add_argument("--emb-dir", default="ext_C2", help="exp/emb/<dir> holding the <split>_all.npz files")
    ap.add_argument("--out", default="exp/analysis/c2_eval.csv")
    a = ap.parse_args()
    global E
    E = ROOT / "exp/emb" / a.emb_dir
    cals = fit_all()
    q3 = c2q3.Q3()

    def score(z, raw, key, ia, ib):
        v, c = key
        args = (z["dur"][ia], z["dur"][ib], z["snr"][ia], z["snr"][ib])
        if c == "Q3":
            return q3.score(raw[v], *args)
        base = "fv" if v == "gated" else v
        if v == "gated":  # each branch with its own calibration
            g = raw["gated_mask"]
            nrm = (z["whole_norm"][ia], z["whole_norm"][ib]) if c == "R+n" else ()
            return np.where(g, cals[("fv", c)](raw["fv"], *args, *nrm), cals[("whole", c)](raw["whole"], *args, *nrm))
        nrm = (z["whole_norm"][ia], z["whole_norm"][ib]) if c == "R+n" else ()
        return cals[(base, c)](raw[v], *args, *nrm)

    if a.private:
        pairs, _ = read_trials(ROOT / "submissions/final_private/trials_2col.txt")
        files = unique_files(pairs)
        pos = {f: i for i, f in enumerate(files)}
        ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
        z = load("private")
        assert list(z["files"]) == files
        g = gate_of("private", files, ia, ib)
        raw = raw_scores(z, ia, ib, g)
        raw["gated_mask"] = g
        raw["whole_as"] = asnorm(raw["whole"], topk_stats(l2n(z["whole"])), ia, ib)
        v, c = a.private.split(":")
        s = score(z, raw, (v, c), ia, ib)
        out = ROOT / "submissions/final_private"
        res = out / "results.txt"
        res.write_text("".join(f"{x} {w} {val:.6f}\n" for (x, w), val in zip(pairs, s)))
        (out / f"results_{Path(a.zip).stem}.txt").write_text(res.read_text())
        lines = res.read_text().splitlines()
        bad = sum(1 for l in lines if len(l.split()) != 3 or not np.isfinite(float(l.split()[2])))
        order = sum(1 for l, (x, w) in zip(lines, pairs) if l.split()[:2] != [x, w])
        assert len(lines) == len(pairs) and order == 0 and bad == 0, (len(lines), order, bad)
        (out / a.zip).unlink(missing_ok=True)
        subprocess.run(["zip", "-q", "-j", str(out / a.zip), str(res)], check=True)
        print(f"private {a.private}: {len(lines):,} lines OK -> {out / a.zip}")
        return

    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
    meta = pd.read_csv(META)
    cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
    w = private_like_weights(pairs, y)
    keys = [("whole", "Q3"), ("whole", "R"), ("whole", "R+n"), ("tta", "R"), ("tta", "R+n"), ("fv", "R"), ("fv", "R+n"),
            ("gated", "R"), ("gated", "R+n"), ("tta_fv", "R"), ("tta_fv", "R+n"), ("whole_as", "R"), ("whole_as", "R+n")]
    rows = []
    for split in a.splits:
        z = load(split)
        assert list(z["files"]) == files
        g = gate_of(split, files, ia, ib)
        raw = raw_scores(z, ia, ib, g)
        raw["gated_mask"] = g
        raw["whole_as"] = asnorm(raw["whole"], topk_stats(l2n(z["whole"])), ia, ib)
        for key in keys:
            s = score(z, raw, key, ia, ib)
            r = dict(split=split, variant=f"{key[0]}+{key[1]}", all=round(eer(s, y), 3), cleaned=round(eer(s[cl], y[cl]), 3))
            if split == "silver":
                r["private_like"] = round(eer(s, y, w), 3)
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="variant", columns="split", values=["all", "cleaned"], sort=False)
    pd.set_option("display.width", 220)
    print(piv.to_string())
    if "silver" in a.splits:
        print(df[df.split == "silver"][["variant", "private_like"]].to_string(index=False))
    df.to_csv(ROOT / a.out, index=False)


if __name__ == "__main__":
    main()
