"""Where does a system fail on the silver set? Errors at its EER threshold, attributed to trial conditions.

    python scripts/failure_analysis_silver.py --scores <fileA fileB score, all 49,993 trials> [--cleaned]

For each condition (bins on trial attributes), prints the false-reject rate on same-speaker trials and the
false-accept rate on different-speaker trials, the bin's share of trials and of errors, and its EXCESS errors: errors
beyond what the bin would have at the overall rate. Excess errors rank the conditions by how much they cost.
Attributes: original public audio (exp/analysis/features_public.csv: speech duration, pitch, level, channel,
bandwidth), the silver layers (silver trial_metadata.csv: SNR, rumble), label quality (cleaned list), two-voice flags
(E2b window clustering on the synthetic stand-in, same speech) and the dominant speaker (id03289).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = Path("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--cleaned", action="store_true", help="analyse the cleaned list only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = pd.read_csv(META)
    sc = pd.read_csv(a.scores, sep=r"\s+", header=None, names=["file_a", "file_b", "score"])
    d = m.merge(sc, on=["file_a", "file_b"])
    f = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel")
    A, B = f.loc[d.file_a].reset_index(drop=True), f.loc[d.file_b].reset_index(drop=True)
    sp_min, sp_max = np.minimum(A.speech_dur, B.speech_dur), np.maximum(A.speech_dur, B.speech_dur)
    d["min_speech_s"] = sp_min.values
    d["speech_ratio"] = (sp_max / np.maximum(sp_min, 0.1)).values
    d["pitch_gap_st"] = (np.abs(np.log(A.f0.clip(60) / B.f0.clip(60))) * 12 / np.log(2)).values
    male = lambda v: v < 165  # noqa: E731
    gm = np.where(male(A.f0.values) & male(B.f0.values), "male-male",
                  np.where(~male(A.f0.values) & ~male(B.f0.values), "female-female", "mixed"))
    d["gender"] = gm
    d["level_gap_db"] = np.abs(A.rms_db - B.rms_db).values
    d["channel_gap_db"] = np.abs(A.noise_db - B.noise_db).values
    d["narrowband"] = (np.minimum(A.bw50, B.bw50) < 4000).values
    ident = json.loads((ROOT / "lists/clean/public_identity.json").read_text())
    d["dominant_spk"] = [(ident[x.split("/")[0]] == ident["id03289"]) or (ident[y.split("/")[0]] == ident["id03289"])
                         for x, y in zip(d.file_a, d.file_b)]
    try:
        from sred.data import read_trials, unique_files  # noqa: F401
        z = np.load(ROOT / "exp/emb/W_E2b_sf2_avg/public_noisy_speakers.npz")
        files = sorted(set(d.file_a) | set(d.file_b))
        files = unique_files(read_trials()[0])
        two = dict(zip(files, z["two"]))
        d["two_voice"] = [bool(two[x] or two[y]) for x, y in zip(d.file_a, d.file_b)]
    except Exception as e:  # noqa: BLE001
        print("two-voice flags unavailable:", e)
    if a.cleaned:
        d = d[d.in_cleaned].reset_index(drop=True)
    y = (d.label == "target").to_numpy()
    s = d.score.to_numpy()
    o = np.argsort(s)
    frr = np.cumsum(y[o]) / y.sum()
    far = 1 - np.cumsum(~y[o]) / (~y).sum()
    k = np.argmin(np.abs(frr - far))
    thr = s[o][k]
    err = (s > thr) != y
    d["error"] = err
    fr_rate, fa_rate = err[y].mean(), err[~y].mean()
    print(f"trials {len(d):,} ({'cleaned list' if a.cleaned else 'all'}); EER {100 * (frr[k] + far[k]) / 2:.2f}% "
          f"-> {err[y].sum():,} false rejects ({100 * fr_rate:.1f}% of same-speaker), {err[~y].sum():,} false accepts "
          f"({100 * fa_rate:.1f}% of different-speaker)\n")
    conds = [
        ("label quality", "in_cleaned", None, None),
        ("shorter clip's speech (s)", "min_speech_s", [0, 1, 2, 4, 99], ["<1", "1-2", "2-4", ">4"]),
        ("speech length ratio", "speech_ratio", [0, 2, 4, 1e9], ["<2x", "2-4x", ">=4x"]),
        ("noisier clip SNR (silver)", "snr_bin", None, None),
        ("rumble clips (silver)", "rumble_clips", None, None),
        ("pitch gap (semitones)", "pitch_gap_st", [-1, 2, 5, 99], ["<2", "2-5", ">5"]),
        ("gender pairing (f0 proxy)", "gender", None, None),
        ("level gap (dB)", "level_gap_db", [-1, 5, 10, 99], ["<5", "5-10", ">10"]),
        ("original channel gap (dB)", "channel_gap_db", [-1, 5, 10, 99], ["<5", "5-10", ">10"]),
        ("narrowband clip", "narrowband", None, None),
        ("two voices detected", "two_voice", None, None),
        ("dominant speaker id03289", "dominant_spk", None, None),
    ]
    rows = []
    for title, col, edges, labels in conds:
        if col not in d:
            continue
        g = pd.cut(d[col], edges, labels=labels).astype(str) if edges else d[col].astype(str)
        print(f"{title}:")
        print(f"  {'bin':14s} {'trials':>8s} {'FRR %':>7s} {'FAR %':>7s} {'share of errors':>16s} {'excess errors':>14s}")
        for v in (labels if labels else sorted(g.unique())):
            sel = (g == v).to_numpy()
            if sel.sum() < 50:
                continue
            fr = err[sel & y].mean() if (sel & y).sum() else np.nan
            fa = err[sel & ~y].mean() if (sel & ~y).sum() else np.nan
            excess = (err[sel & y].sum() - (sel & y).sum() * fr_rate) + (err[sel & ~y].sum() - (sel & ~y).sum() * fa_rate)
            share = err[sel].sum() / err.sum()
            print(f"  {str(v):14s} {sel.sum():>8,} {100 * fr:>7.1f} {100 * fa:>7.1f} {100 * share:>15.1f}% {excess:>+14.0f}")
            rows.append(dict(condition=title, bin=v, trials=int(sel.sum()), FRR=round(100 * fr, 2), FAR=round(100 * fa, 2),
                             error_share=round(100 * share, 1), excess_errors=round(float(excess))))
        print()
    r = pd.DataFrame(rows).sort_values("excess_errors", ascending=False)
    print("TOP excess-error bins (where the errors concentrate beyond chance):")
    print(r.head(12).to_string(index=False))
    if a.out:
        d.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
