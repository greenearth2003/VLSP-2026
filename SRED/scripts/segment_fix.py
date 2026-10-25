"""Segment-gated scoring fixes: apply a scoring fix only to the trials of one segment, keep the baseline elsewhere.

    python scripts/segment_fix.py --model exp/ft/S3_E2b_X2/avg.pt --splits public_rumble_v2 public_noisy public_realnoise

Segments (gates) use only label-free per-file measurements of the test audio itself (exp/analysis/features_<split>.csv,
scripts/data_features.py): clip duration, energy-detected speech duration and share, level, noise floor.
Fixes (alternative raw scores):
  vad, full+vad     energy-trimmed embeddings (sred.evaluate --mode vad), or their average with the whole-clip one
  chunk_max/top3/mean   the longer clip (> 8 s) cut into ALL 2-s chunks (hop 1 s, scripts/chunk_embed.py), each chunk
                    compared with the other clip's whole-clip embedding: max, mean of the top 3, mean of all
Each fix and the baseline get the same calibration: a duration QMF (logistic regression over the raw score, log
durations and their products) fitted on train-cohort trials scored with that same fix (chunk fixes: on train trials
whose longer clip is > 8 s), so all outputs are log-odds on one scale and can be mixed. No test labels are fitted;
labels only measure EER. "fuse" = mean of the baseline and the fix inside the segment.
Prints per split the segment's EER (baseline -> fix) and the POOLED EER with the fix applied inside the segment only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF2  # noqa: E402
from sred.data import cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KMAX = 24
VAD = dict(mode="vad", crop_sec=3.0, hop_sec=1.5)  # the key sred.evaluate --mode vad caches under
CHUNK = "chunk2_gt8"  # scripts/chunk_embed.py --win 2 --min-dur 8
LONG = 8.0


def eer(s, y):
    o = np.argsort(s)
    yo = y[o]
    frr = np.cumsum(yo) / yo.sum()
    far = 1 - np.cumsum(1 - yo) / (1 - yo).sum()
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2


def raw_scores(full, vad, E, n, dur, ia, ib, chunk=20000):
    """Raw score of every fix for trials (ia, ib). Chunk scores fall back to the whole-clip cosine where the longer
    clip has no chunks (<= 8 s); those trials are outside the chunk gates."""
    out = {k: np.zeros(len(ia), np.float32) for k in ("full", "vad", "full+vad", "chunk_max", "chunk_top3", "chunk_mean")}
    fv = l2n(full + vad)
    for s in range(0, len(ia), chunk):
        a, b = ia[s:s + chunk], ib[s:s + chunk]
        a_long = dur[a] >= dur[b]
        L, S = np.where(a_long, a, b), np.where(a_long, b, a)
        cos = (full[a] * full[b]).sum(1)
        out["full"][s:s + chunk] = cos
        out["vad"][s:s + chunk] = (vad[a] * vad[b]).sum(1)
        out["full+vad"][s:s + chunk] = (fv[a] * fv[b]).sum(1)
        mask = np.arange(KMAX)[None] < n[L][:, None]
        c = np.where(mask, np.einsum("pkd,pd->pk", E[L], full[S]), np.nan)
        has = n[L] > 0
        top = -np.sort(-np.nan_to_num(c, nan=-2.0), axis=1)[:, :3]
        top3 = np.nanmean(np.where(top > -1.5, top, np.nan), axis=1) if has.any() else cos
        with np.errstate(all="ignore"):
            out["chunk_max"][s:s + chunk] = np.where(has, top[:, 0], cos)
            out["chunk_top3"][s:s + chunk] = np.where(has, top3, cos)
            out["chunk_mean"][s:s + chunk] = np.where(has, np.nanmean(c, axis=1), cos)
    return out


def qmf_feats(s, d1, d2):
    lo, hi = np.log(np.maximum(np.minimum(d1, d2), 0.05)), np.log(np.maximum(np.maximum(d1, d2), 0.05))
    return np.stack([s, lo, hi, s * lo, s * hi], 1)


class Calibrators:
    """One duration-QMF per fix, fitted on train-cohort trials scored with that fix."""

    def __init__(self, full, vad, E, n, dur, spk):
        I, J, y = QMF2(["dur"]).trials(spk)
        raw = raw_scores(full, vad, E, n, dur, I, J)
        long_ = np.maximum(dur[I], dur[J]) > LONG
        self.models = {}
        for k, s in raw.items():
            sel = long_ if k.startswith("chunk") else np.ones(len(I), bool)
            X = qmf_feats(s[sel], dur[I][sel], dur[J][sel])
            mu, sd = X.mean(0), X.std(0) + 1e-9
            lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((X - mu) / sd, y[sel])
            self.models[k] = (mu, sd, lr)

    def __call__(self, k, s, d1, d2):
        mu, sd, lr = self.models[k]
        return lr.decision_function((qmf_feats(s, d1, d2) - mu) / sd)


def gates(feat, ia, ib):
    d, sp, share = feat.dur.to_numpy(), feat.speech_dur.to_numpy(), feat.speech_ratio.to_numpy()
    lv, nf = feat.rms_db.to_numpy(), feat.noise_db.to_numpy()
    dmin, dmax = np.minimum(d[ia], d[ib]), np.maximum(d[ia], d[ib])
    ratio = dmax / np.maximum(dmin, 0.1)
    return {
        "all trials": np.ones(len(ia), bool),
        "shorter clip speech < 1 s": np.minimum(sp[ia], sp[ib]) < 1.0,
        "shorter clip < 2 s": dmin < 2.0,
        "longer clip > 8 s": dmax > LONG,
        "longer > 8 s and ratio >= 4x": (dmax > LONG) & (ratio >= 4),
        "longer clip > 15 s": dmax > 15,
        "length ratio >= 4x": ratio >= 4,
        "speech share < 50%": np.minimum(share[ia], share[ib]) < 0.5,
        "level gap > 10 dB": np.abs(lv[ia] - lv[ib]) > 10,
        "noise-floor gap > 10 dB": np.abs(nf[ia] - nf[ib]) > 10,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", nargs="+", default=["public_rumble_v2"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
    cdur = np.array([lengths[r] / 16000 for r in cf])
    ck = cache_key(a.model, "traincohort5", cf, mode="full")
    z = np.load(ck.with_name(f"traincohort5_{CHUNK}.npz"))
    cal = Calibrators(l2n(np.load(ck)), l2n(np.load(cache_key(a.model, "traincohort5", cf, **VAD))), z["E"], z["n"], cdur, cs)

    pairs, y = read_trials()
    meta = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")
    clean = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[p] for p, _ in pairs]); ib = np.array([pos[q] for _, q in pairs])
    rows, policy_rows = [], []
    for split in a.splits:
        feat = pd.read_csv(ROOT / f"exp/analysis/features_{split}.csv").set_index("rel").loc[files]
        dur = feat.dur.to_numpy()
        fk = cache_key(a.model, split, files, mode="full")
        cp = fk.with_name(f"{split}_{CHUNK}.npz")  # chunk fixes only where chunk embeddings were made
        w = np.load(cp) if cp.exists() else {"E": np.zeros((len(files), KMAX, 192), np.float32), "n": np.zeros(len(files), int)}
        raw = raw_scores(l2n(np.load(fk)), l2n(np.load(cache_key(a.model, split, files, **VAD))), w["E"], w["n"], dur, ia, ib)
        cal_s = {k: cal(k, s, dur[ia], dur[ib]) for k, s in raw.items()}
        for k in ("tile", "vadtile"):  # short clips re-embedded (scripts/short_tta_embed.py); whole-clip QMF on top
            tp = fk.with_name(f"{split}_short2_{k}3.npy")
            if tp.exists():
                e = l2n(np.load(tp))
                cal_s[k] = cal("full", (e[ia] * e[ib]).sum(1), dur[ia], dur[ib])
        base = cal_s["full"]
        base_eer = eer(base, y)
        for gname, g in gates(feat, ia, ib).items():
            seg_base = eer(base[g], y[g])
            fixes = ["vad", "full+vad"] + (["chunk_max", "chunk_top3", "chunk_mean"] if cp.exists() and ("8 s" in gname or "15 s" in gname) else [])
            fixes += [k for k in ("tile", "vadtile") if k in cal_s and gname.startswith("shorter clip < 2")]
            for fix in fixes:
                for name, s_fix in ((fix, cal_s[fix]), (f"fuse {fix}", 0.5 * (base + cal_s[fix]))):
                    mixed = np.where(g, s_fix, base)
                    rows.append(dict(split=split, segment=gname, share=round(100 * g.mean(), 1), fix=name,
                                     seg_base=round(seg_base, 2), seg_fix=round(eer(s_fix[g], y[g]), 2),
                                     pooled_base=round(base_eer, 3), pooled_fix=round(eer(mixed, y), 3)))
        print(f"{split}: baseline (whole-clip QMF) EER {base_eer:.3f}", flush=True)
        # combined policies: each gated fix replaces the baseline only inside its segment
        G = gates(feat, ia, ib)
        hard = G["noise-floor gap > 10 dB"] | G["level gap > 10 dB"] | G["shorter clip speech < 1 s"]
        pol = {"P0 baseline (whole clip)": base,
               "P1 full+vad everywhere": cal_s["full+vad"],
               "P2 vad in hard segments": np.where(hard, cal_s["vad"], base),
               "P4 full+vad in hard segments": np.where(hard, cal_s["full+vad"], base)}
        if "vadtile" in cal_s:
            short = G["shorter clip < 2 s"]
            pol["P5 vadtile for clips < 2 s"] = np.where(short, cal_s["vadtile"], base)
            pol["P3 P2 + vadtile for clips < 2 s"] = np.where(short, cal_s["vadtile"], pol["P2 vad in hard segments"])
        low = G["speech share < 50%"]
        pol["Q1 full+vad where speech share < 50%"] = np.where(low, cal_s["full+vad"], base)
        q2 = low | G["shorter clip speech < 1 s"] | G["level gap > 10 dB"]
        pol["Q2 full+vad where share < 50% | speech < 1 s | level gap"] = np.where(q2, cal_s["full+vad"], base)
        print(f"  {split}: Q1 gate covers {low.mean():.1%} of trials, Q2 gate {q2.mean():.1%}")
        for k, s in pol.items():
            policy_rows.append(dict(split=split, policy=k, EER=round(eer(s, y), 3), EER_cleaned=round(eer(s[clean], y[clean]), 3)))
    df = pd.DataFrame(rows)
    df["d_pooled"] = (df.pooled_fix - df.pooled_base).round(3)
    pd.set_option("display.width", 220)
    for split, g in df.groupby("split", sort=False):
        print(f"\n== {split}")
        print(g.drop(columns=["split", "pooled_base"]).to_string(index=False))
    if df.split.nunique() > 1:
        wide = df.pivot_table(index=["segment", "fix"], columns="split", values="d_pooled")
        wide["worst"] = wide.max(1)
        print("\nPooled EER change with the fix applied ONLY inside the segment (negative = better), by the worst split:")
        print(wide.sort_values("worst").head(20).to_string())
    pdf = pd.DataFrame(policy_rows).pivot_table(index="policy", columns="split", values=["EER", "EER_cleaned"], sort=False)
    print("\nCombined gated policies, pooled EER (hard segments = noise-floor gap > 10 dB or level gap > 10 dB or shorter clip"
          " speech < 1 s):")
    print(pdf.to_string())
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
