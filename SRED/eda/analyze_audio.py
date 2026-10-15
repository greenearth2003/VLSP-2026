"""EDA part 1: train vs public domain comparison, gender, and W01 error analysis.

Reads eda/out/audio_stats.csv (eda/audio_stats.py) and the W01 public scores.
Writes figures to docs/eda_figs/ and numbers to eda/out/audio_summary.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sred.data import read_trials  # noqa: E402
from sred.metrics import compute_eer  # noqa: E402

FIG = ROOT / "docs/eda_figs"
FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "eda/out"
SCORES = ROOT / "releases/version_141026_W01/sred_W01_public_141026.txt"
F0_SPLIT = 165.0  # Hz, speaker-median F0 threshold male/female
res = {}

df = pd.read_csv(OUT / "audio_stats.csv")
df = df[df["error"].isna()] if "error" in df else df
tr, pu = df[df.split == "train"], df[df.split == "public"]

# ---------------------------------------------------------------- A. domain shift
cols = {"dur": "Độ dài (s)", "rms_dbfs": "RMS (dBFS)", "snr_est": "SNR ước lượng (dB)",
        "active_frac": "Tỉ lệ khung có tiếng", "rolloff99": "Rolloff 99% (Hz)",
        "frac_above4k": "Năng lượng > 4 kHz", "f0_med": "F0 trung vị (Hz)", "peak": "Đỉnh |x|"}
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
res["domain"] = {}
for ax, (c, name) in zip(axs.ravel(), cols.items()):
    a, b = tr[c].dropna().values, pu[c].dropna().values
    lo, hi = np.percentile(np.r_[a, b], [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax.hist(a, bins, density=True, alpha=0.55, label="train")
    ax.hist(b, bins, density=True, alpha=0.55, label="public")
    ax.set_title(name)
    ks = ks_2samp(a, b).statistic
    res["domain"][c] = dict(train_p50=float(np.median(a)), public_p50=float(np.median(b)),
                            train_p10=float(np.percentile(a, 10)), public_p10=float(np.percentile(b, 10)),
                            train_p90=float(np.percentile(a, 90)), public_p90=float(np.percentile(b, 90)), ks=float(ks))
    ax.text(0.98, 0.95, f"KS={ks:.2f}", transform=ax.transAxes, ha="right", va="top")
axs[0, 0].legend()
fig.tight_layout()
fig.savefig(FIG / "01_domain_train_vs_public.png", dpi=110)
plt.close(fig)

res["quality_flags"] = {}
for name, d in [("train", tr), ("public", pu)]:
    res["quality_flags"][name] = dict(
        n=len(d), clipped_files=float((d.clip_frac > 1e-4).mean()), low_level=float((d.rms_dbfs < -40).mean()),
        low_snr_lt15=float((d.snr_est < 15).mean()), mostly_silence=float((d.active_frac < 0.5).mean()),
        bandlimited_4k=float((d.rolloff99 < 4000).mean()), narrowband_roll_lt3k=float((d.rolloff99 < 3000).mean()),
        lead_sil_gt1s=float((d.lead_sil > 1).mean()), sr_not_16k=float((d.sr != 16000).mean()))

# ---------------------------------------------------------------- B. gender (speaker-level F0)
spk_f0 = df.groupby(["split", "spk"]).f0_med.median().reset_index()
spk_f0["gender"] = np.where(spk_f0.f0_med < F0_SPLIT, "M", "F")
res["gender"] = {s: spk_f0[spk_f0.split == s].gender.value_counts(normalize=True).round(3).to_dict()
                 for s in ["train", "public"]}
fig, ax = plt.subplots(figsize=(7, 4))
for s in ["train", "public"]:
    ax.hist(spk_f0[spk_f0.split == s].f0_med.dropna(), np.linspace(70, 300, 70), density=True, alpha=0.55, label=s)
ax.axvline(F0_SPLIT, color="k", ls="--")
ax.set_xlabel("F0 trung vị theo speaker (Hz)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "02_speaker_f0.png", dpi=110)
plt.close(fig)
g_of = dict(zip(spk_f0[spk_f0.split == "public"].spk, spk_f0[spk_f0.split == "public"].gender))

# ---------------------------------------------------------------- C. W01 error analysis on public
pairs, lab = read_trials()
sc = np.loadtxt(SCORES, usecols=2)
P = pu.set_index("rel")
t = pd.DataFrame(pairs, columns=["a", "b"])
t["y"], t["s"] = lab, sc
for c in ["dur", "snr_est", "active_frac", "rolloff99", "rms_dbfs", "f0_med"]:
    va, vb = P.loc[t.a, c].values, P.loc[t.b, c].values
    t[c + "_min"], t[c + "_max"] = np.minimum(va, vb), np.maximum(va, vb)
t["spk_a"], t["spk_b"] = t.a.str.split("/").str[0], t.b.str.split("/").str[0]
t["g_a"], t["g_b"] = t.spk_a.map(g_of), t.spk_b.map(g_of)
t["same_gender"] = t.g_a == t.g_b
t["idx_gap"] = (t.a.str.split("/").str[1].str[:-4].astype(int) - t.b.str.split("/").str[1].str[:-4].astype(int)).abs()

eer_all = compute_eer(t.s.values, t.y.values)
# EER threshold
from sklearn.metrics import roc_curve  # noqa: E402
fpr, tpr, thr = roc_curve(t.y, t.s)
thr_eer = thr[np.nanargmin(np.abs((1 - tpr) - fpr))]
t["err"] = np.where(t.y == 1, t.s < thr_eer, t.s >= thr_eer)
res["w01"] = dict(eer=eer_all, thr=float(thr_eer), n_err=int(t.err.sum()))


def eer_by(col, bins, labels=None):
    out = []
    cut = pd.cut(t[col], bins, labels=labels)
    for k, g in t.groupby(cut, observed=True):
        if g.y.nunique() == 2 and min((g.y == 1).sum(), (g.y == 0).sum()) >= 50:
            out.append(dict(bin=str(k), n=len(g), eer=round(compute_eer(g.s.values, g.y.values), 2),
                            share_of_errors=round(g.err.sum() / t.err.sum(), 3)))
    return out


res["w01_by"] = {
    "dur_min": eer_by("dur_min", [0, 1.5, 2, 3, 4, 6, 8, 12, 30]),
    "snr_min": eer_by("snr_est_min", [0, 15, 20, 25, 30, 35, 40, 80]),
    "active_min": eer_by("active_frac_min", [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]),
    "rolloff_min": eer_by("rolloff99_min", [0, 2000, 3000, 4000, 5000, 6000, 8001]),
    "rms_min": eer_by("rms_dbfs_min", [-80, -40, -35, -30, -25, -20, 0]),
}
# gender: nontarget same vs cross gender -> FA rate at EER threshold; EER restricted to same-gender nontargets
non = t[t.y == 0]
res["w01_gender"] = dict(
    nontarget_same_gender_share=float(non.same_gender.mean()),
    fa_rate_same_gender=float(non[non.same_gender].err.mean()), fa_rate_cross_gender=float(non[~non.same_gender].err.mean()),
    eer_same_gender_nontargets=compute_eer(t[(t.y == 1) | t.same_gender].s.values, t[(t.y == 1) | t.same_gender].y.values),
    miss_rate_M=float(t[(t.y == 1) & (t.g_a == "M")].err.mean()), miss_rate_F=float(t[(t.y == 1) & (t.g_a == "F")].err.mean()))
# session proxy: target pairs by file-index gap
tg = t[t.y == 1]
res["w01_target_idx_gap"] = [dict(gap=str(k), n=len(g), miss_rate=round(g.err.mean(), 4))
                             for k, g in tg.groupby(pd.cut(tg.idx_gap, [0, 1, 3, 10, 30, 100, 3000]), observed=True)]
# speakers: error concentration
err_spk = pd.concat([t[t.err & (t.y == 1)].spk_a, t[t.err & (t.y == 0)].spk_a, t[t.err & (t.y == 0)].spk_b]).value_counts()
trials_spk = pd.concat([t.spk_a, t.spk_b[t.y == 0]]).value_counts()
res["w01_speakers"] = dict(
    n_spk_with_errors=int((err_spk > 0).sum()), n_spk=int(len(trials_spk)),
    top10_share_of_errors=float(err_spk.head(10).sum() / err_spk.sum()),
    top10=[dict(spk=s, errors=int(n), trials=int(trials_spk[s]), gender=g_of.get(s)) for s, n in err_spk.head(10).items()])
# per-speaker miss rate for speakers with >=30 target trials
ms = tg.groupby("spk_a").err.agg(["mean", "size"])
ms = ms[ms["size"] >= 30].sort_values("mean")
res["w01_spk_miss_rate"] = dict(n=len(ms), p10=float(ms["mean"].quantile(.1)), p50=float(ms["mean"].median()),
                                p90=float(ms["mean"].quantile(.9)), max=float(ms["mean"].max()))

fig, axs = plt.subplots(1, 4, figsize=(19, 4))
for ax, (k, ttl) in zip(axs, [("dur_min", "d_min (s)"), ("snr_min", "SNR_min (dB)"),
                              ("active_min", "tỉ lệ có tiếng (min)"), ("rolloff_min", "rolloff99 min (Hz)")]):
    r = res["w01_by"][k]
    ax.bar(range(len(r)), [x["eer"] for x in r], color="C3")
    ax.set_xticks(range(len(r)), [x["bin"] for x in r], rotation=45, ha="right", fontsize=8)
    ax.axhline(eer_all, color="k", ls="--", lw=1)
    ax.set_title(f"EER W01 theo {ttl}")
fig.tight_layout()
fig.savefig(FIG / "03_w01_eer_by_factor.png", dpi=110)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(t[t.y == 1].s, 100, alpha=0.6, density=True, label="target")
axs[0].hist(t[t.y == 0].s, 100, alpha=0.6, density=True, label="nontarget")
axs[0].axvline(thr_eer, color="k", ls="--")
axs[0].legend()
axs[0].set_title("Phân bố điểm W01 (QMF)")
axs[1].plot(np.arange(1, len(err_spk) + 1), np.cumsum(err_spk.values) / err_spk.sum())
axs[1].set_xscale("log")
axs[1].set_xlabel("số speaker (xếp theo số lỗi giảm dần)")
axs[1].set_ylabel("tỉ lệ lỗi tích luỹ")
axs[1].grid(alpha=.3)
axs[1].set_title("Lỗi tập trung vào ít speaker")
fig.tight_layout()
fig.savefig(FIG / "04_w01_scores_and_error_concentration.png", dpi=110)
plt.close(fig)

t.to_csv(OUT / "public_trials_annotated.csv", index=False)
(OUT / "audio_summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
