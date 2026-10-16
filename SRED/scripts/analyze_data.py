"""Dataset analysis report (train + public test), from exp/analysis/features_*.csv.

    python scripts/analyze_data.py > exp/analysis/report.txt
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import PUBLIC_DIR, PUBLIC_TRIALS, TRAIN_LIST, read_train_list, read_trials

A = Path("exp/analysis")
Q = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]


def h(t):
    print(f"\n{'=' * 8} {t} {'=' * (70 - len(t))}")


def desc(df: pd.DataFrame, cols, by="split"):
    print(df.groupby(by)[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.round(2).to_string())


def bins(x, edges, labels=None):
    return pd.cut(x, edges, right=False, labels=labels)


def main():
    feats = pd.concat([pd.read_csv(p) for p in sorted(A.glob("features_*.csv"))], ignore_index=True)
    feats["spk"] = feats.rel.str.split("/").str[0]
    feats["gender"] = np.where(feats.f0 < 165, "M", np.where(feats.f0 >= 165, "F", "?"))
    splits = sorted(feats.split.unique())

    # ------------------------------------------------------------------ integrity
    h("FORMAT / INTEGRITY")
    print(feats.groupby("split")[["sr", "channels", "subtype"]].agg(lambda s: dict(Counter(s))).to_string())
    print("files <0.5 s:", feats.groupby("split").apply(lambda d: int((d.dur < 0.5).sum())).to_dict())
    print("files with ~no speech (speech_dur<0.3 s):",
          feats.groupby("split").apply(lambda d: int((d.speech_dur < 0.3).sum())).to_dict())
    dup = feats[feats.duplicated("md5", keep=False)].sort_values("md5")
    print(f"byte-identical files: {len(dup)} rows in {dup.md5.nunique()} groups")
    if len(dup):
        g = dup.groupby("md5").agg(n=("rel", "size"), splits=("split", lambda s: "+".join(sorted(set(s)))),
                                   spks=("spk", lambda s: len(set(s))))
        print(g.value_counts(["splits", "spks"]).to_string())
        print(dup.head(12)[["split", "rel", "dur"]].to_string(index=False))

    # ------------------------------------------------------------------ speakers
    h("SPEAKERS")
    for s in splits:
        d = feats[feats.split == s]
        per = d.groupby("spk").agg(n=("rel", "size"), dur=("dur", "sum"))
        print(f"[{s}] speakers={len(per)} files={len(d)} hours={d.dur.sum() / 3600:.1f}")
        print("  utts/spk quantiles", per.n.quantile(Q).round(1).tolist())
        print("  minutes/spk quantiles", (per.dur / 60).quantile(Q).round(2).tolist())
    if {"train", "public"} <= set(splits):
        tr, pu = set(feats[feats.split == "train"].spk), set(feats[feats.split == "public"].spk)
        print(f"speaker-ID overlap train∩public: {len(tr & pu)} (train {len(tr)}, public {len(pu)})")
        ids = lambda S: sorted(int(x[2:]) for x in S if x[2:].isdigit())  # noqa: E731
        print(f"  id range train {ids(tr)[0]}..{ids(tr)[-1]}, public {ids(pu)[0]}..{ids(pu)[-1]}")

    # ------------------------------------------------------------------ acoustics
    h("DURATION (s)")
    desc(feats, ["dur", "speech_dur", "speech_ratio"])
    for s in splits:
        d = feats[feats.split == s]
        print(f"[{s}] dur bins", bins(d.dur, [0, 1, 2, 3, 4, 6, 8, 12, 20, 1e9]).value_counts(sort=False, normalize=True)
              .round(3).to_dict())
    h("LEVEL / NOISE / CHANNEL")
    desc(feats, ["rms_db", "peak", "snr_db", "noise_db", "bw50", "rolloff95", "hf_ratio_db"])
    print("clipped files (clip_frac>0.1%):", feats.groupby("split").apply(lambda d: round((d.clip_frac > 1e-3).mean(), 4)).to_dict())
    print("digital-silence floor (noise_db<-80):", feats.groupby("split").apply(lambda d: round((d.noise_db < -80).mean(), 4)).to_dict())
    for s in splits:
        d = feats[feats.split == s]
        print(f"[{s}] bw50 bins", bins(d.bw50, [0, 3500, 4100, 5600, 7000, 7900, 8001],
                                       ["<3.5k", "~4k(nb)", "4-5.5k", "5.6-7k", "7-7.9k", "8k(full)"])
              .value_counts(sort=False, normalize=True).round(3).to_dict())
        print(f"[{s}] snr bins", bins(d.snr_db, [0, 10, 15, 20, 30, 40, 1e9]).value_counts(sort=False, normalize=True).round(3).to_dict())

    h("GENDER PROXY (median F0 per speaker, <165 Hz = M)")
    for s in splits:
        d = feats[feats.split == s]
        f0s = d.groupby("spk").f0.median()
        print(f"[{s}] speakers M={int((f0s < 165).sum())} F={int((f0s >= 165).sum())} unknown={int(f0s.isna().sum())}",
              " | F0 quantiles", f0s.quantile([.05, .25, .5, .75, .95]).round(0).tolist())

    # within-speaker variability (session/channel heterogeneity)
    h("WITHIN-SPEAKER CHANNEL VARIABILITY (std across a speaker's files)")
    for s in splits:
        d = feats[feats.split == s]
        g = d.groupby("spk")[["snr_db", "bw50", "rms_db", "f0"]].std()
        print(f"[{s}] median per-speaker std:", g.median().round(2).to_dict())

    # ------------------------------------------------------------------ trials
    if "public" not in splits:
        return
    h("PUBLIC TRIALS")
    pairs, labels = read_trials(PUBLIC_TRIALS)
    pf = feats[feats.split == "public"].set_index("rel")
    t = pd.DataFrame(pairs, columns=["a", "b"])
    t["y"] = labels
    t["sa"], t["sb"] = t.a.str.split("/").str[0], t.b.str.split("/").str[0]
    print(f"trials={len(t)} target={int(t.y.sum())} ({t.y.mean():.3f})")
    lab_ok = ((t.sa == t.sb) == (t.y == 1)).mean()
    print(f"label == (same speaker dir): {lab_ok:.4f}")
    print(f"self-trials (a==b): {(t.a == t.b).sum()}   duplicate pairs: {t.duplicated(['a', 'b']).sum()}"
          f"   reversed duplicates: {len(set(zip(t.a, t.b)) & set(zip(t.b, t.a)))}")
    on_disk = {p.relative_to(PUBLIC_DIR).as_posix() for p in PUBLIC_DIR.rglob("*.wav")}
    used = set(t.a) | set(t.b)
    print(f"wav files on disk={len(on_disk)} used in trials={len(used)} unused={len(on_disk - used)}"
          f" missing={len(used - on_disk)}")
    unused_spk = {f.split('/')[0] for f in on_disk - used}
    print(f"  unused files span {len(unused_spk)} speakers; of which not in any trial: "
          f"{len(unused_spk - set(t.sa) - set(t.sb))}")
    use = Counter(list(t.a) + list(t.b))
    print("file reuse count quantiles", np.quantile(list(use.values()), Q).tolist())
    spk_tr = Counter(list(t.sa) + list(t.sb))
    print(f"speakers in trials={len(spk_tr)}; trial-appearances/spk quantiles",
          np.quantile(list(spk_tr.values()), Q).tolist())
    tg = t[t.y == 1].groupby("sa").size()
    print(f"target trials by speaker: speakers={len(tg)} quantiles", tg.quantile(Q).tolist())

    for c, col in (("a", "da"), ("b", "db")):
        t[col] = pf.loc[t[c], "dur"].values
    t["dmin"], t["dmax"] = t[["da", "db"]].min(1), t[["da", "db"]].max(1)
    print("\nd_min distribution by label (fraction):")
    t["dmin_bin"] = bins(t.dmin, [0, 1, 2, 3, 4, 6, 8, 1e9])
    print(pd.crosstab(t.dmin_bin, t.y, normalize="columns").round(3).to_string())
    print("mean d_min target/nontarget:", t.groupby("y").dmin.mean().round(2).to_dict(),
          " mean d_max:", t.groupby("y").dmax.mean().round(2).to_dict())

    g = pf.groupby(pf.index.str.split("/").str[0]).f0.median()
    gend = lambda s: "?" if pd.isna(g.get(s)) else ("M" if g[s] < 165 else "F")  # noqa: E731
    t["ga"], t["gb"] = t.sa.map(gend), t.sb.map(gend)
    nt = t[t.y == 0]
    print(f"\nnontarget same-gender (F0 proxy): {(nt.ga == nt.gb).mean():.3f}   "
          f"(random pairing of trial speakers would give ~{((np.array(list(map(gend, spk_tr))) == 'M').mean() ** 2 + (np.array(list(map(gend, spk_tr))) == 'F').mean() ** 2):.3f})")

    # do nontarget speakers come from a structured pool? (e.g. nearest neighbours)
    print("nontarget: unique speaker pairs", nt[["sa", "sb"]].apply(lambda r: tuple(sorted(r)), axis=1).nunique(),
          "of", len(nt))

    # trial-level acoustic covariates
    for c, col in (("a", "snra"), ("b", "snrb")):
        t[col] = pf.loc[t[c], "snr_db"].values
    t["snr_min"] = t[["snra", "snrb"]].min(1)
    print("\nsnr_min by label (mean):", t.groupby("y").snr_min.mean().round(2).to_dict())
    for c, col in (("a", "bwa"), ("b", "bwb")):
        t[col] = pf.loc[t[c], "bw50"].values
    t["bw_mismatch"] = (t.bwa - t.bwb).abs() > 1500
    print("bandwidth-mismatched trials (|Δbw50|>1.5k):", t.groupby("y").bw_mismatch.mean().round(3).to_dict())
    t.to_csv(A / "public_trials_annot.csv", index=False)


if __name__ == "__main__":
    main()
