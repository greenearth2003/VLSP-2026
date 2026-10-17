"""Identity / sample distribution of train, P-dev, P-hold and the full public test.

    /opt/conda/bin/python scripts/identity_stats.py   -> exp/analysis/identity_stats.{md,png}

Train: raw folders (train.txt) and cleaned identities (lists/clean/train_clean.txt, duplicate folders merged,
flagged files dropped). Public: raw folders, merged identities (lists/clean/public_identity.json), and the
P-dev / P-hold halves of the cleaned trials (only files that occur in those trials are counted).
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "exp/analysis"
DATA = ROOT / "data/Vi-SRED26"


def half(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2


dur = pd.concat([pd.read_csv(A / "features_train.csv"), pd.read_csv(A / "features_public.csv")])
dmap = {(r.split, r.rel): (r.dur, r.speech_dur) for r in dur.itertuples()}

tr_raw = pd.read_csv(DATA / "train/train.txt", sep=r"\s+", header=None, names=["spk", "rel"])
tr_cln = pd.read_csv(ROOT / "lists/clean/train_clean.txt", sep=r"\s+", header=None, names=["spk", "rel"])
for d in (tr_raw, tr_cln):
    d["dur"] = [dmap[("train", r)][0] for r in d.rel]

pairs = [l.split() for l in open(DATA / "public_test/public-test.txt")]
ident = json.loads((ROOT / "lists/clean/public_identity.json").read_text())
keep = {tuple(l.split()[:2]) for l in open(ROOT / "lists/clean/public_trials_clean.txt")}
tr = pd.DataFrame(pairs, columns=["a", "b", "lab"])
tr["y"] = (tr.lab == "target").astype(int)
tr["sa"] = [ident[x.split("/")[0]] for x in tr.a]
tr["sb"] = [ident[x.split("/")[0]] for x in tr.b]
tr["clean"] = [(a, b) in keep for a, b in zip(tr.a, tr.b)]
tr["ha"], tr["hb"] = tr.sa.map(half), tr.sb.map(half)
subsets = {
    "public (full)": tr,
    "P-dev (cleaned)": tr[tr.clean & (tr.ha == 0) & (tr.hb == 0)],
    "P-hold (cleaned)": tr[tr.clean & (tr.ha == 1) & (tr.hb == 1)],
}


def files_of(t, merged=True):
    f = pd.DataFrame({"rel": pd.concat([t.a, t.b]).unique()})
    f["spk"] = [ident[x.split("/")[0]] if merged else x.split("/")[0] for x in f.rel]
    f["dur"] = [dmap[("public", r)][0] for r in f.rel]
    return f


def summarize(name, f, t=None):
    g = f.groupby("spk").agg(n=("rel", "size"), dur=("dur", "sum"))
    r = dict(set=name, identities=len(g), files=len(f), hours=f.dur.sum() / 3600,
             files_per_id_min=g.n.min(), files_per_id_p25=g.n.quantile(.25), files_per_id_median=g.n.median(),
             files_per_id_mean=g.n.mean(), files_per_id_p75=g.n.quantile(.75), files_per_id_p95=g.n.quantile(.95),
             files_per_id_max=g.n.max(), ids_with_1_file=int((g.n == 1).sum()), ids_with_le3=int((g.n <= 3).sum()),
             min_per_id_median=g.dur.median() / 60, top1_id_share_files=g.n.max() / len(f),
             top10_ids_share_files=g.n.nlargest(10).sum() / len(f), file_dur_median=f.dur.median())
    if t is not None:
        tg = t[t.y == 1].groupby("sa").size()
        r.update(trials=len(t), targets=int(t.y.sum()), nontargets=int((1 - t.y).sum()),
                 ids_with_target_trials=len(tg), top1_id_share_targets=tg.max() / tg.sum() if len(tg) else np.nan)
    return r, g


rows, groups = [], {}
for name, f in [("train raw folders", tr_raw), ("train cleaned identities", tr_cln)]:
    r, g = summarize(name, f)
    rows.append(r), groups.__setitem__(name, g)
r, g = summarize("public raw folders", files_of(tr, merged=False), tr)
rows.append(r), groups.__setitem__("public raw folders", g)
for name, t in subsets.items():
    r, g = summarize(name, files_of(t), t)
    rows.append(r), groups.__setitem__(name, g)
S = pd.DataFrame(rows).set_index("set")

# overlap of identities between P-dev and P-hold must be empty
dev_ids = set(subsets["P-dev (cleaned)"].sa) | set(subsets["P-dev (cleaned)"].sb)
hold_ids = set(subsets["P-hold (cleaned)"].sa) | set(subsets["P-hold (cleaned)"].sb)

# ---------------------------------------------------------------- markdown
lines = ["# Identity / sample distribution\n",
         f"P-dev ∩ P-hold identities: {len(dev_ids & hold_ids)} (must be 0). Train and public identity namespaces are "
         "independent (folder IDs are reused for different people).\n"]
show = S.T.copy()
fmt = lambda v: f"{v:,.0f}" if float(v).is_integer() else (f"{v:.1%}" if abs(v) < 1 and v > 0 else f"{v:,.2f}")  # noqa: E731
pct_rows = {"top1_id_share_files", "top10_ids_share_files", "top1_id_share_targets"}
cols = list(show.columns)
lines.append("| metric | " + " | ".join(cols) + " |")
lines.append("|---|" + "|".join("---" for _ in cols) + "|")
for k, rr in show.iterrows():
    vals = []
    for v in rr.values:
        if pd.isna(v):
            vals.append("—")
        elif k in pct_rows:
            vals.append(f"{v:.1%}")
        else:
            vals.append(f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}")
    lines.append(f"| {k} | " + " | ".join(vals) + " |")
bins = [1, 2, 4, 6, 11, 21, 51, 101, 10_000]
labels = ["1", "2–3", "4–5", "6–10", "11–20", "21–50", "51–100", ">100"]
lines.append("\n## Identities by number of files\n")
lines.append("| files per identity | " + " | ".join(groups) + " |")
lines.append("|---|" + "|".join("---" for _ in groups) + "|")
hist = {k: pd.cut(g.n, bins, right=False, labels=labels).value_counts().reindex(labels) for k, g in groups.items()}
for lab in labels:
    lines.append(f"| {lab} | " + " | ".join(str(int(hist[k][lab])) for k in groups) + " |")
(A / "identity_stats.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
c = {"train cleaned identities": "#4C78A8", "public raw folders": "#9E9E9E", "P-dev (cleaned)": "#F58518",
     "P-hold (cleaned)": "#54A24B"}
lb = np.unique(np.round(np.logspace(0, np.log10(700), 26)))
for k, col in c.items():
    ax[0, 0].hist(groups[k].n, bins=lb, histtype="step", lw=2, color=col, label=f"{k} ({len(groups[k])} ids)")
ax[0, 0].set_xscale("log"), ax[0, 0].set_xlabel("files per identity (log)"), ax[0, 0].set_ylabel("identities")
ax[0, 0].set_title("Files per identity"), ax[0, 0].legend(fontsize=8)
for k, col in c.items():
    n = np.sort(groups[k].n.values)[::-1]
    ax[0, 1].plot(np.arange(1, len(n) + 1) / len(n), np.cumsum(n) / n.sum(), color=col, lw=2, label=k)
ax[0, 1].plot([0, 1], [0, 1], ":", color="#bbb")
ax[0, 1].set_xlabel("fraction of identities (largest first)"), ax[0, 1].set_ylabel("fraction of files")
ax[0, 1].set_title("Concentration (Lorenz curve)"), ax[0, 1].legend(fontsize=8)
for k, col in c.items():
    ax[1, 0].hist(groups[k].dur / 60, bins=np.logspace(-1.5, 2, 30), histtype="step", lw=2, color=col, label=k)
ax[1, 0].set_xscale("log"), ax[1, 0].set_xlabel("minutes of audio per identity (log)"), ax[1, 0].set_ylabel("identities")
ax[1, 0].set_title("Audio per identity"), ax[1, 0].legend(fontsize=8)
for name, t in subsets.items():
    tg = t[t.y == 1].groupby("sa").size().sort_values(ascending=False)
    ax[1, 1].plot(np.arange(1, len(tg) + 1), tg.values, lw=2, label=f"{name}: top id {tg.iloc[0] / tg.sum():.0%} of targets",
                  color={"public (full)": "#9E9E9E", "P-dev (cleaned)": "#F58518", "P-hold (cleaned)": "#54A24B"}[name])
ax[1, 1].set_xscale("log"), ax[1, 1].set_yscale("log"), ax[1, 1].set_xlabel("identity rank"), ax[1, 1].set_ylabel("target trials")
ax[1, 1].set_title("Target trials per identity"), ax[1, 1].legend(fontsize=8)
for a_ in ax.flat:
    a_.grid(alpha=.25)
fig.suptitle("VLSP 2026 SRED — identity distribution (train vs public dev/hold)", fontsize=13)
fig.tight_layout()
fig.savefig(A / "identity_stats.png", dpi=110)
print("\nwrote", A / "identity_stats.md", A / "identity_stats.png")
