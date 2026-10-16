"""Failure analysis on the public test: which trial conditions make the models fail?

    python scripts/error_analysis.py            # dumps scores (cached) + prints the report

Per system: QMF(dur) scores (the fixed submission back-end), errors at the system's own EER
threshold. Trial attributes come from exp/analysis/features_*.csv (scripts/data_features.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF
from sred.data import PUBLIC_DIR, TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files
from sred.embed import cache_key
from sred.metrics import compute_eer
from sred.scoring import l2n

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "exp/analysis"
SYSTEMS = {  # short name -> model name used for the embedding cache
    "M_zs": "redimnet_M_mix",
    "M_wise": "exp/ft/W_G01_M/avg.pt",
    "S_ft": "exp/ft/G02_S/step3000.pt",
    "B4_ft": "exp/ft/G03_B4/step3000.pt",
    "R34_wise": "exp/ft/W_G04_R34/avg.pt",
    "B6_wise": "exp/ft/W_G05_B6/avg.pt",
}


def h(t):
    print(f"\n{'=' * 6} {t} {'=' * max(4, 84 - len(t))}")


def qmf_scores(model, files, ia, ib, dur):
    out = A / "scores" / (Path(model).parent.name + "_" + Path(model).stem if "/" in model else model)
    out = out.parent / (out.name + ".npy")  # not with_suffix: names like a0.4 contain dots
    if out.exists():
        return np.load(out)
    ci = cohort_subset(read_train_list(), per_spk=5)
    emb = l2n(np.load(cache_key(model, "public", files, mode="full")))
    coh = l2n(np.load(cache_key(model, "traincohort5", [r for _, r in ci], mode="full")))
    lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
    cdur = np.array([lengths[r] / 16000 for _, r in ci])
    s = QMF(use_asnorm=False).fit(coh, [s for s, _ in ci], cdur).score(emb, dur, ia, ib)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, s)
    return s


def eer_threshold(s, y):
    order = np.argsort(s)
    ss, yy = s[order], y[order]
    frr = np.cumsum(yy) / yy.sum()  # targets at/below threshold
    far = 1 - np.cumsum(1 - yy) / (1 - yy).sum()
    k = np.argmin(np.abs(frr - far))
    return ss[k]


def rate_table(t, col, systems, label):
    """Error rate (FRR for targets / FAR for non-targets) per bin of `col`, per system."""
    d = t[t.y == label]
    g = d.groupby(col, observed=True)
    tab = pd.DataFrame({"n": g.size(), "share": g.size() / len(d)})
    for s in systems:
        tab[s] = g[f"err_{s}"].mean() * 100
    tab["share_of_errors(M_wise)"] = g["err_M_wise"].sum() / d["err_M_wise"].sum()
    return tab.round(3)


def main():
    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[a] for a, _ in pairs])
    ib = np.array([pos[b] for _, b in pairs])

    f = pd.read_csv(A / "features_public.csv").set_index("rel").loc[files]
    f["spk"] = [x.split("/")[0] for x in files]
    dur = f.dur.values
    spk_f0 = f.groupby("spk").f0.median()
    f["spk_gender"] = f.spk.map(lambda s: "?" if pd.isna(spk_f0[s]) else ("M" if spk_f0[s] < 165 else "F"))

    t = pd.DataFrame({"a": [a for a, _ in pairs], "b": [b for _, b in pairs], "y": y})
    A_, B_ = f.iloc[ia].reset_index(drop=True), f.iloc[ib].reset_index(drop=True)
    t["sa"], t["sb"] = A_.spk.values, B_.spk.values
    t["dmin"], t["dmax"] = np.minimum(A_.dur, B_.dur), np.maximum(A_.dur, B_.dur)
    t["spmin"] = np.minimum(A_.speech_dur, B_.speech_dur)
    t["snr_min"] = np.minimum(A_.snr_db, B_.snr_db)
    t["snr_gap"] = (A_.snr_db - B_.snr_db).abs()
    t["bw_min"] = np.minimum(A_.bw50, B_.bw50)
    t["bw_gap"] = (A_.bw50 - B_.bw50).abs()
    t["hf_gap"] = (A_.hf_ratio_db - B_.hf_ratio_db).abs()
    t["rms_gap"] = (A_.rms_db - B_.rms_db).abs()
    t["clip_any"] = (A_.clip_frac > 1e-3) | (B_.clip_frac > 1e-3)
    t["sratio_min"] = np.minimum(A_.speech_ratio, B_.speech_ratio)
    t["f0_gap"] = (np.log(A_.f0) - np.log(B_.f0)).abs() * 12 / np.log(2)  # semitones
    t["f0_mean"] = np.sqrt(A_.f0 * B_.f0)
    t["gpair"] = np.where(A_.spk_gender.values == B_.spk_gender.values, A_.spk_gender.values + A_.spk_gender.values,
                          "MF")
    t["dom_spk"] = (t.sa == "id03289") | (t.sb == "id03289")

    systems = list(SYSTEMS)
    for s, m in SYSTEMS.items():
        sc = qmf_scores(m, files, ia, ib, dur)
        th = eer_threshold(sc, y)
        t[f"s_{s}"] = sc
        t[f"err_{s}"] = np.where(y == 1, sc < th, sc >= th)
    show = ["M_zs", "M_wise", "S_ft", "B4_ft", "R34_wise", "B6_wise"]

    h("OVERALL (QMF scores, errors at each system's EER threshold)")
    print(pd.DataFrame({s: dict(EER=compute_eer(t[f"s_{s}"].values, y),
                                FRR=t[t.y == 1][f"err_{s}"].mean() * 100,
                                FAR=t[t.y == 0][f"err_{s}"].mean() * 100) for s in systems}).round(2).to_string())

    # ------------------------------------------------------------------ binned rates
    B = {
        "dmin": [0, 1.5, 2, 3, 4, 6, 8, 99],
        "spmin": [0, 0.75, 1, 1.5, 2, 3, 5, 99],
        "snr_min": [0, 10, 15, 20, 25, 30, 40, 200],
        "bw_min": [0, 4100, 5600, 7000, 7900, 8001],
        "bw_gap": [0, 300, 800, 1500, 3000, 9000],
        "snr_gap": [0, 3, 6, 10, 20, 200],
        "rms_gap": [0, 2, 4, 6, 10, 100],
        "f0_gap": [0, 1, 2, 3, 5, 8, 40],
        "f0_mean": [0, 140, 165, 200, 240, 500],
    }
    for col, edges in B.items():
        t[col + "_bin"] = pd.cut(t[col], edges, right=False)
    for col in ["dmin", "spmin", "snr_min", "bw_min", "bw_gap", "snr_gap", "rms_gap", "f0_gap", "f0_mean"]:
        for lab, nm in ((1, "TARGET trials -> false-reject rate %"), (0, "NON-TARGET trials -> false-accept rate %")):
            h(f"{col}: {nm}")
            print(rate_table(t, col + "_bin", show, lab).to_string())
    for col in ["gpair", "clip_any", "dom_spk"]:
        for lab, nm in ((1, "FRR %"), (0, "FAR %")):
            h(f"{col}: {nm}")
            print(rate_table(t, col, show, lab).to_string())

    # ------------------------------------------------------------------ multivariate
    h("LOGISTIC REGRESSION of error (M_wise) on standardized trial attributes")
    feats = ["dmin", "dmax", "snr_min", "snr_gap", "bw_min", "bw_gap", "rms_gap", "f0_gap", "f0_mean", "sratio_min"]
    for lab, nm in ((1, "target / false reject"), (0, "non-target / false accept")):
        d = t[(t.y == lab)].dropna(subset=feats)
        X = d[feats].copy()
        X["dmin"], X["dmax"] = np.log(X.dmin), np.log(X.dmax)
        X["cross_gender"] = (d.gpair == "MF").astype(float)
        X["dom_spk"] = d.dom_spk.astype(float)
        X = (X - X.mean()) / (X.std() + 1e-9)
        for s in ["M_wise", "B6_wise"]:
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(X, d[f"err_{s}"])
            co = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False)
            print(f"[{nm}] {s}: " + "  ".join(f"{k}={v:+.2f}" for k, v in co.items()))

    # ------------------------------------------------------------------ agreement across models
    h("ERROR OVERLAP ACROSS SUBMITTABLE MODELS (M_wise, S_ft, B4_ft, R34_wise)")
    sub = ["M_wise", "S_ft", "B4_ft", "R34_wise"]
    t["n_fail"] = t[[f"err_{s}" for s in sub]].sum(1)
    print(pd.crosstab(t.n_fail, t.y.map({1: "target", 0: "nontarget"})).to_string())
    print(f"errors of M_wise also made by ALL other 3: "
          f"{(t[t.err_M_wise].n_fail == 4).mean():.1%};  by B6_wise too: "
          f"{t[t.err_M_wise & (t.n_fail == 4)].err_B6_wise.mean():.1%}")
    hard = t[t.n_fail == 4]
    easy = t[t.n_fail == 0]
    cmp_cols = ["dmin", "dmax", "spmin", "snr_min", "bw_min", "bw_gap", "snr_gap", "rms_gap", "f0_gap", "f0_mean"]
    for lab, nm in ((1, "target"), (0, "nontarget")):
        print(f"\n[{nm}] median attributes: all-4-fail vs all-4-correct")
        print(pd.DataFrame({"all_fail": hard[hard.y == lab][cmp_cols].median(),
                            "all_correct": easy[easy.y == lab][cmp_cols].median()}).T.round(2).to_string())
        print("   gender pair share:", {k: round(v, 3) for k, v in hard[hard.y == lab].gpair.value_counts(normalize=True).items()},
              " vs correct:", {k: round(v, 3) for k, v in easy[easy.y == lab].gpair.value_counts(normalize=True).items()})

    # ------------------------------------------------------------------ speaker concentration
    h("SPEAKER CONCENTRATION OF ERRORS (M_wise)")
    tg = t[t.y == 1].groupby("sa").agg(n=("y", "size"), fr=("err_M_wise", "sum"))
    tg = tg.sort_values("fr", ascending=False)
    cum = tg.fr.cumsum() / tg.fr.sum()
    print(f"target speakers with >=1 FR: {(tg.fr > 0).sum()} / {len(tg)}; "
          f"top-5 speakers hold {cum.iloc[4]:.1%} of FRs, top-20 {cum.iloc[19]:.1%}")
    tg["frr%"] = (tg.fr / tg.n * 100).round(1)
    tg["f0"] = spk_f0.reindex(tg.index).round(0)
    tg["spk_bw_std"] = f.groupby("spk").bw50.std().reindex(tg.index).round(0)
    tg["spk_snr_std"] = f.groupby("spk").snr_db.std().reindex(tg.index).round(1)
    print(tg.head(12).to_string())
    big = tg[tg.n >= 20]
    print(f"\nspeakers with >=20 target trials: {len(big)}; corr(FRR, within-spk bw std)="
          f"{big['frr%'].corr(big.spk_bw_std, method='spearman'):.2f}, corr(FRR, within-spk snr std)="
          f"{big['frr%'].corr(big.spk_snr_std, method='spearman'):.2f}, corr(FRR, F0)={big['frr%'].corr(big.f0, method='spearman'):.2f}")
    nt = t[t.y == 0].copy()
    nt["pair"] = [tuple(sorted(p)) for p in zip(nt.sa, nt.sb)]
    fa = nt[nt.err_M_wise]
    ps = fa.pair.value_counts()
    print(f"false accepts: {len(fa)} over {len(ps)} speaker pairs; pairs with >=3 FAs hold {ps[ps >= 3].sum() / len(fa):.1%}")
    fs = pd.concat([fa.sa, fa.sb]).value_counts()
    print("speakers most involved in FAs:", fs.head(8).to_dict())

    # worst trials to listen to
    cols = ["a", "b", "y", "dmin", "snr_min", "bw_gap", "f0_gap", "gpair", "s_M_wise", "s_B6_wise", "n_fail"]
    hard.sort_values("s_M_wise", ascending=True)[hard.y == 1][cols].head(40).to_csv(A / "worst_false_rejects.csv", index=False)
    hard.sort_values("s_M_wise", ascending=False)[hard.y == 0][cols].head(40).to_csv(A / "worst_false_accepts.csv", index=False)
    t.drop(columns=[c for c in t if c.endswith("_bin")]).to_csv(A / "trial_errors.csv", index=False)
    print("\nwrote worst_false_rejects.csv / worst_false_accepts.csv / trial_errors.csv in", A)


if __name__ == "__main__":
    main()
