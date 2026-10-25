"""Compare systems on the silver set by failure group, and bound what better per-group calibration could gain.

    python scripts/silver_groups.py exp/scores/silver/W_E2b_sf2.txt exp/scores/silver/S3_E2b_X2.txt ...

Groups come from the original public audio (exp/analysis/features_public.csv): speech in the shorter clip, channel
gap (noise-floor difference), pitch gap, level gap, male-male pairs. "oracle calib" re-centres each speech-length
bin on its own EER threshold before pooling: it uses the labels, so it is an upper bound on what duration-aware
calibration could recover, not a usable method.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = Path("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")


def eer(s, y):
    o = np.argsort(s)
    yo = y[o]
    frr = np.cumsum(yo) / yo.sum()
    far = 1 - np.cumsum(1 - yo) / (1 - yo).sum()
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2, s[o][k]


def weighted_eer(s, y, w):
    o = np.argsort(s)
    yo, wo = y[o], w[o]
    frr = np.cumsum(wo * yo) / (wo * yo).sum()
    far = 1 - np.cumsum(wo * (1 - yo)) / (wo * (1 - yo)).sum()
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2


def private_length_weights(sp, edges=(1, 2, 4)):
    """Per-trial weights that give silver the private test's mix of shorter-clip speech lengths (no labels used)."""
    fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv").set_index("rel").speech_dur
    pv = pd.read_csv(ROOT / "submissions/final_private/trials_2col.txt", sep=r"\s+", header=None)
    sp_priv = np.minimum(fv.loc[pv[0]].to_numpy(), fv.loc[pv[1]].to_numpy())
    share_priv = np.bincount(np.digitize(sp_priv, edges), minlength=len(edges) + 1) / len(sp_priv)
    b = np.digitize(sp, edges)
    share_silver = np.bincount(b, minlength=len(edges) + 1) / len(sp)
    print("shorter-clip speech <1 / 1-2 / 2-4 / >=4 s: private " + " / ".join(f"{100 * v:.1f}%" for v in share_priv)
          + " | silver " + " / ".join(f"{100 * v:.1f}%" for v in share_silver))
    return (share_priv / share_silver)[b]


def main():
    m = pd.read_csv(META)
    f = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel")
    A, B = f.loc[m.file_a].reset_index(drop=True), f.loc[m.file_b].reset_index(drop=True)
    sp = np.minimum(A.speech_dur, B.speech_dur).to_numpy()
    w_priv = private_length_weights(sp)
    groups = {
        "all trials": np.ones(len(m), bool),
        "cleaned list": m.in_cleaned.to_numpy(),
        "shorter clip speech < 1 s": sp < 1,
        "shorter clip speech 1-2 s": (sp >= 1) & (sp < 2),
        "shorter clip speech >= 2 s": sp >= 2,
        "channel gap > 10 dB": (np.abs(A.noise_db - B.noise_db) > 10).to_numpy(),
        "pitch gap > 5 st": (np.abs(np.log2(A.f0.clip(60) / B.f0.clip(60))) * 12 > 5).to_numpy(),
        "level gap > 10 dB": (np.abs(A.rms_db - B.rms_db) > 10).to_numpy(),
        "male-male (f0 < 165 Hz)": ((A.f0 < 165) & (B.f0 < 165)).to_numpy(),
    }
    y = (m.label == "target").to_numpy(int)
    bins = np.digitize(sp, [1, 2, 4])
    rows = {}
    for path in sys.argv[1:]:
        sc = pd.read_csv(path, sep=r"\s+", header=None, names=["file_a", "file_b", "score"])
        s = m.merge(sc, on=["file_a", "file_b"], how="left").score.to_numpy(float)
        r = {g: eer(s[sel], y[sel])[0] for g, sel in groups.items()}
        _, t_all = eer(s, y)
        s2 = s.copy()
        for b in np.unique(bins):
            sel = bins == b
            s2[sel] += t_all - eer(s[sel], y[sel])[1]
        r["oracle calib by speech bin (all)"] = eer(s2, y)[0]
        r["private length mix (all)"] = weighted_eer(s, y, w_priv)
        cl = groups["cleaned list"]
        r["private length mix (cleaned)"] = weighted_eer(s[cl], y[cl], w_priv[cl])
        rows[Path(path).stem] = r
    df = pd.DataFrame(rows).round(2)
    df.insert(0, "trials", [int(sel.sum()) for sel in groups.values()] + [len(m), len(m), int(groups["cleaned list"].sum())])
    print(df.to_string())


if __name__ == "__main__":
    main()
