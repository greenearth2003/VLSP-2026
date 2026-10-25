"""Error analysis of one private trial with our final system (S3 soup + segment-gated trimming + multi-speaker blend):
the trial's score and its parts, where it sits in the private score distribution, and clip A compared with every
W-s window of clip B (and the reverse when A is long enough).

    python scripts/pair_deep_dive.py id07250/0010.wav id08108/0006.wav --tag 1 [--win 3 --hop 0.5]

Writes exp/analysis/pairs/pair<tag>.png and pair<tag>.json; prints a report.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key, vad_trim  # noqa: E402
from sred.models import load_model  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "exp/ft/S3_E2b_X2/avg.pt"
VAD = dict(mode="vad", crop_sec=3.0, hop_sec=1.5)
PRIV = ROOT / "data/Vi-SRED26/private_test/data"
FINAL = ROOT / "submissions/final_private/results_result-private-S3-seg-ms025-t0.txt"
SILVER = ROOT / "exp/scores/silver/S3_seg_ms025.txt"
LAM = 0.25


def eer_thr(s, y):
    o = np.argsort(s)
    yo = y[o]
    frr = np.cumsum(yo) / yo.sum()
    far = 1 - np.cumsum(1 - yo) / (1 - yo).sum()
    k = np.argmin(np.abs(frr - far))
    return float(s[o][k]), 100 * (frr[k] + far[k]) / 2


def speech_share(x, a, b):
    """energy-based speech share of x[a:b] (frames > clip's 20th percentile + 6 dB), as the gate measures it"""
    fr = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    db = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
    sp = db > np.percentile(db, 20) + 6
    return float(sp[a // 160:max(a // 160 + 1, b // 160)].mean())


def raw_scores(split, files, ia, ib):
    """Our final system's score before calibration: cosine of the embedding it uses for the trial (whole clip, or
    whole + energy-trimmed inside the gate) + LAM * multi-speaker gain."""
    feat = pd.read_csv(ROOT / f"exp/analysis/features_{split}.csv").set_index("rel").loc[files]
    sh, sp, lv = feat.speech_ratio.to_numpy(), feat.speech_dur.to_numpy(), feat.rms_db.to_numpy()
    gate = (np.minimum(sh[ia], sh[ib]) < 0.5) | (np.minimum(sp[ia], sp[ib]) < 1.0) | (np.abs(lv[ia] - lv[ib]) > 10)
    fk = cache_key(MODEL, split, files, mode="full")
    full = l2n(np.load(fk))
    fv = l2n(full + l2n(np.load(cache_key(MODEL, split, files, **VAD))))
    z = np.load(fk.with_name(f"{split}_speakers.npz"))
    S = [l2n(z[k]) for k in ("full", "c0", "c1")]
    s_f = (S[0][ia] * S[0][ib]).sum(1)
    gain = np.maximum(0, np.max([(A[ia] * B[ib]).sum(1) for A in S for B in S], 0) - s_f)
    return np.where(gate, (fv[ia] * fv[ib]).sum(1), (full[ia] * full[ib]).sum(1)) + LAM * gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tag", default="x")
    ap.add_argument("--win", type=float, default=3.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--calibrated", action="store_true", help="show calibrated log-odds instead of raw similarity")
    a = ap.parse_args()
    out = ROOT / "exp/analysis/pairs"
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_segment_submission import fit_qmf

    # calibrators exactly as the final system (train-cohort duration QMF per branch)
    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
    cdur = np.array([lengths[r] / 16000 for r in cf])
    Cf = l2n(np.load(cache_key(MODEL, "traincohort5", cf, mode="full")))
    q_full = fit_qmf(Cf, cs, cdur)
    q_fv = fit_qmf(l2n(Cf + l2n(np.load(cache_key(MODEL, "traincohort5", cf, **VAD)))), cs, cdur)

    pairs, _ = read_trials(ROOT / "submissions/final_private/trials_2col.txt")
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    i, j = pos[a.a], pos[a.b]
    feat = pd.read_csv(ROOT / "exp/analysis/features_private.csv").set_index("rel")
    fa, fb = feat.loc[a.a], feat.loc[a.b]
    full = l2n(np.load(cache_key(MODEL, "private", files, mode="full")))
    vad = l2n(np.load(cache_key(MODEL, "private", files, **VAD)))
    fv = l2n(full + vad)
    z = np.load(cache_key(MODEL, "private", files, mode="full").with_name("private_speakers.npz"))
    S = [l2n(z[k]) for k in ("full", "c0", "c1")]
    cos_full, cos_vad, cos_fv = float(full[i] @ full[j]), float(vad[i] @ vad[j]), float(fv[i] @ fv[j])
    best_pair = max(float(A[i] @ B[j]) for A in S for B in S)
    gain = max(0.0, best_pair - float(S[0][i] @ S[0][j]))
    gate = bool(min(fa.speech_ratio, fb.speech_ratio) < 0.5 or min(fa.speech_dur, fb.speech_dur) < 1.0 or abs(fa.rms_db - fb.rms_db) > 10)
    d1, d2 = np.array([fa.dur]), np.array([fb.dur])
    s_base = float(q_full(np.array([cos_full + LAM * gain]), d1, d2)[0])
    s_gated = float(q_fv(np.array([cos_fv + LAM * gain]), d1, d2)[0])
    final = s_gated if gate else s_base

    raw = (cos_fv if gate else cos_full) + LAM * gain  # the same trial before calibration

    # threshold = EER point on silver; distribution over all private trials (raw similarity unless --calibrated)
    pairs_s, y = read_trials()
    if a.calibrated:
        sil = pd.DataFrame(pairs_s, columns=["a", "b"]).merge(pd.read_csv(SILVER, sep=" ", header=None, names=["a", "b", "s"]), how="left").s.to_numpy()
        s_priv = pd.read_csv(FINAL, sep=" ", header=None, names=["a", "b", "s"]).s.to_numpy()
        value = final
    else:
        fs = unique_files(pairs_s)
        ps = {f: k for k, f in enumerate(fs)}
        sil = raw_scores("public_rumble_v2", fs, np.array([ps[x] for x, _ in pairs_s]), np.array([ps[x] for _, x in pairs_s]))
        s_priv = raw_scores("private", files, np.array([pos[x] for x, _ in pairs]), np.array([pos[x] for _, x in pairs]))
        value = raw
    thr, eer = eer_thr(sil, y)
    pct = 100 * (s_priv < value).mean()
    unit = "calibrated score (log-odds)" if a.calibrated else "similarity (cosine)"

    # clip A (as the final system embeds it) vs every window of clip B, and the reverse
    model = load_model(MODEL).cuda().eval()
    W, H = int(a.win * 16000), int(a.hop * 16000)

    def windows(x):
        if len(x) <= W:
            return [0], torch.from_numpy(np.tile(x, int(np.ceil(W / max(1, len(x)))))[:W][None])
        st = list(range(0, len(x) - W + 1, H))
        return st, torch.from_numpy(np.stack([x[s:s + W] for s in st]))

    @torch.inference_mode()
    def emb(t):
        return F.normalize(model(t.cuda()), dim=-1).cpu().numpy()

    xa, _ = sf.read(str(PRIV / a.a), dtype="float32")
    xb, _ = sf.read(str(PRIV / a.b), dtype="float32")
    ea = fv[i] if gate else full[i]  # A as the final system embeds it
    eb = fv[j] if gate else full[j]
    curves = {}
    for name, x, ref, dref in (("A vs windows of B", xb, ea, fa.dur), ("B vs windows of A", xa, eb, fb.dur)):
        if len(x) < W + H:
            continue
        st, segs = windows(x)
        e = emb(segs)
        c = e @ ref
        cal = (q_fv if gate else q_full)(c, np.full(len(c), dref), np.full(len(c), a.win)) if a.calibrated else c
        curves[name] = dict(t=[s / 16000 + a.win / 2 for s in st], cos=c.tolist(), score=cal.tolist(),
                            speech=[speech_share(x, s, s + W) for s in st])
    # VAD-trimmed parts of B alone (what the energy trimming keeps)
    rep = dict(pair=[a.a, a.b], tag=a.tag, dur=[float(fa.dur), float(fb.dur)], speech_s=[float(fa.speech_dur), float(fb.speech_dur)],
               speech_share=[float(fa.speech_ratio), float(fb.speech_ratio)], level_db=[float(fa.rms_db), float(fb.rms_db)],
               f0_hz=[float(fa.f0), float(fb.f0)], snr_db=[float(fa.snr_db), float(fb.snr_db)],
               cos_full=cos_full, cos_trimmed=cos_vad, cos_full_plus_trimmed=cos_fv, multi_speaker_gain=gain,
               two_voice=[bool(z["two"][i]), bool(z["two"][j])], gated=gate, score_base=s_base, score_gated=s_gated,
               final_calibrated=final, raw_similarity=raw, shown=unit, value=value, threshold=thr, silver_eer=eer, margin=value - thr,
               private_percentile=pct, curves=curves)
    (out / f"pair{a.tag}.json").write_text(json.dumps(rep, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1 + len(curves), 1, figsize=(11, 3.6 * (1 + len(curves))), constrained_layout=True)
    ax = np.atleast_1d(ax)
    tgt, non = sil[y == 1], sil[y == 0]
    bins = np.linspace(np.percentile(s_priv, 0.05), np.percentile(s_priv, 99.95), 120)
    ax[0].hist(s_priv, bins=bins, color="#7a8699", alpha=0.85, label=f"private, all {len(s_priv):,} trials")
    h2 = ax[0].twinx()
    h2.hist(tgt, bins=bins, histtype="step", density=True, color="#1f7a45", lw=1.4, label="silver same-speaker (density)")
    h2.hist(non, bins=bins, histtype="step", density=True, color="#b3322a", lw=1.4, label="silver different-speaker (density)")
    h2.set_yticks([])
    ax[0].axvline(thr, color="#1f5fa8", lw=1.5, ls="--")
    ax[0].axvline(value, color="#e08a00", lw=2)
    ax[0].text(thr, ax[0].get_ylim()[1] * 0.92, f" threshold {thr:+.2f}", color="#1f5fa8", fontsize=9)
    ax[0].text(value, ax[0].get_ylim()[1] * 0.80, f" pair #{a.tag}: {value:+.3f}", color="#e08a00", fontsize=9)
    ax[0].set_xlabel(f"{unit}, our best system")
    ax[0].set_ylabel("private trials")
    ax[0].set_title(f"Private {unit}: range {s_priv.min():+.2f} to {s_priv.max():+.2f}, "
                    f"{100 * (s_priv > thr).mean():.1f}% above the silver EER threshold")
    l1, n1 = ax[0].get_legend_handles_labels()
    l2, n2 = h2.get_legend_handles_labels()
    ax[0].legend(l1 + l2, n1 + n2, loc="upper left", fontsize=8, frameon=False)
    for k, (name, c) in enumerate(curves.items(), 1):
        t = np.array(c["t"])
        ax[k].plot(t, c["score"], color="#1f5fa8", lw=1.8, marker="o", ms=3, label=f"{name} ({a.win:g} s windows, hop {a.hop:g} s)")
        ax[k].axhline(thr, color="#1f5fa8", lw=1, ls="--", label="decision threshold")
        ax[k].axhline(value, color="#e08a00", lw=1.5, label=f"whole-clip trial value {value:+.3f}")
        a2 = ax[k].twinx()
        a2.bar(t, c["speech"], width=a.hop * 0.9, color="#9aa5b5", alpha=0.35, label="speech share of window")
        a2.set_ylim(0, 1)
        a2.set_ylabel("speech share", color="#6c7583")
        ax[k].set_ylabel(unit)
        ax[k].set_xlabel("window centre (s)")
        ax[k].set_zorder(a2.get_zorder() + 1)
        ax[k].patch.set_visible(False)
        ax[k].legend(loc="upper right", fontsize=8, frameon=False)
        ax[k].set_title(f"{name}: best window {max(c['score']):+.2f} at {t[int(np.argmax(c['score']))]:.1f} s, worst {min(c['score']):+.2f}")
    fig.suptitle(f"Pair #{a.tag}: A {a.a} ({fa.dur:.1f} s) vs B {a.b} ({fb.dur:.1f} s)", fontsize=12)
    fig.savefig(out / f"pair{a.tag}.png", dpi=130)
    print(json.dumps({k: v for k, v in rep.items() if k != "curves"}, indent=1))
    for name, c in curves.items():
        print(f"{name}: windows {len(c['t'])}, {unit} min {min(c['score']):+.2f} / median {np.median(c['score']):+.2f} / max {max(c['score']):+.2f}; "
              f"windows above threshold {np.mean(np.array(c['score']) > thr):.0%}")
    print("figure:", out / f"pair{a.tag}.png")


if __name__ == "__main__":
    main()
