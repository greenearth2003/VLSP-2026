"""Length calibration with real short audio in the calibration set. The duration QMF is fitted on train-cohort trials,
but the cohort has few short clips, so its behaviour for 1-2 s clips is extrapolated. Here every cohort clip also
contributes N random crops of 0.8-4 s (whole-clip and energy-trimmed embeddings, as the final system uses), and the
QMFs are refitted on trials over clips + crops (a crop is never paired with its own source clip).

    python scripts/short_calib.py [--crops 2] [--lam 0.25]          # evaluate on the three stand-ins
    python scripts/short_calib.py --private --zip result-private-S3-seg-ms025-shortcal-t0.zip

Prints, per stand-in, EER all / cleaned list of the final pipeline (S3 + gated trimming [+ multi-speaker]) with the
current calibration and with the short-audio calibration, and EER on trials whose shorter clip is < 2 s.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_segment_submission import qmf_feats  # noqa: E402
from sred.data import TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key, embed_wav  # noqa: E402
from sred.evaluate import write_submission  # noqa: E402
from sred.models import load_model  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "exp/ft/S3_E2b_X2/avg.pt"
VAD = dict(mode="vad", crop_sec=3.0, hop_sec=1.5)


@torch.inference_mode()
def crop_embeddings(cf, cdur, n_crops, seed=0):
    cache = cache_key(MODEL, "traincohort5", cf, mode="full").with_name(f"traincohort5_shortcrops{n_crops}.npz")
    if cache.exists():
        z = np.load(cache)
        return z["src"], z["dur"], z["full"], z["vad"]
    model = load_model(MODEL).cuda().eval()
    rng = np.random.default_rng(seed)
    src, dur, full, vad = [], [], [], []
    for k, (rel, d) in enumerate(zip(cf, cdur)):
        x, _ = sf.read(str(TRAIN_DIR / rel), dtype="float32")
        for _ in range(n_crops):
            L = min(len(x), int(rng.uniform(0.8, 4.0) * 16000))
            s = int(rng.integers(0, len(x) - L + 1))
            w = torch.from_numpy(x[s:s + L]).cuda()
            full.append(embed_wav(model, w, mode="full").float().cpu().numpy())
            vad.append(embed_wav(model, w, mode="vad").float().cpu().numpy())
            src.append(k)
            dur.append(L / 16000)
        if k % 2000 == 0:
            print(f"  crops: {k}/{len(cf)}", flush=True)
    out = dict(src=np.array(src), dur=np.array(dur), full=l2n(np.stack(full)), vad=l2n(np.stack(vad)))
    np.savez(cache, **out)
    return out["src"], out["dur"], out["full"], out["vad"]


def trials(spk, srcid, n_non=400000, seed=0):
    """random different-speaker pairs + same-speaker pairs from different source recordings (crop-vs-own-clip excluded)"""
    rng = np.random.default_rng(seed)
    n = len(spk)
    I, J = rng.integers(0, n, n_non), rng.integers(0, n, n_non)
    by = pd.Series(np.arange(n)).groupby(spk).apply(list)
    ti, tj = [], []
    for idx in by:
        idx = np.array(idx)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if srcid[idx[a]] != srcid[idx[b]]:
                    ti.append(idx[a]); tj.append(idx[b])
    I, J = np.concatenate([I, ti]), np.concatenate([J, tj])
    keep = (I != J) & (srcid[I] != srcid[J])
    I, J = I[keep], J[keep]
    return I, J, (spk[I] == spk[J]).astype(int)


def fit(E, dur, I, J, y):
    X = qmf_feats((E[I] * E[J]).sum(1), dur[I], dur[J])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((X - mu) / sd, y)
    return lambda s, d1, d2: lr.decision_function((qmf_feats(s, d1, d2) - mu) / sd)


def eer(s, y):
    o = np.argsort(s)
    yo = y[o]
    frr = np.cumsum(yo) / yo.sum()
    far = 1 - np.cumsum(1 - yo) / (1 - yo).sum()
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--zip", default=None)
    a = ap.parse_args()
    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    cdur = np.array([json.loads((ROOT / "lists/train_lengths.json").read_text())[r] / 16000 for r in cf])
    Cf = l2n(np.load(cache_key(MODEL, "traincohort5", cf, mode="full")))
    Cv = l2n(np.load(cache_key(MODEL, "traincohort5", cf, **VAD)))
    src, kdur, Kf, Kv = crop_embeddings(cf, cdur, a.crops)
    # current calibration: clips only; new: clips + short crops
    I0, J0, y0 = trials(cs, np.arange(len(cs)))
    old = {"full": fit(Cf, cdur, I0, J0, y0), "fv": fit(l2n(Cf + Cv), cdur, I0, J0, y0)}
    spk = np.concatenate([cs, cs[src]])
    sid = np.concatenate([np.arange(len(cs)), src])
    D = np.concatenate([cdur, kdur])
    I1, J1, y1 = trials(spk, sid)
    print(f"calibration trials: clips only {len(y0):,} ({y0.sum():,} same); clips + crops {len(y1):,} ({y1.sum():,} same); "
          f"trials with a clip < 2 s: {np.mean(np.minimum(cdur[I0], cdur[J0]) < 2):.1%} -> {np.mean(np.minimum(D[I1], D[J1]) < 2):.1%}")
    new = {"full": fit(np.concatenate([Cf, Kf]), D, I1, J1, y1), "fv": fit(l2n(np.concatenate([Cf + Cv, Kf + Kv])), D, I1, J1, y1)}

    def score(split, cal, lam):
        private = split == "private"
        pairs, y = read_trials(ROOT / "submissions/final_private/trials_2col.txt") if private else read_trials()
        files = unique_files(pairs)
        pos = {f: i for i, f in enumerate(files)}
        ia = np.array([pos[p] for p, _ in pairs]); ib = np.array([pos[q] for _, q in pairs])
        feat = pd.read_csv(ROOT / f"exp/analysis/features_{split}.csv").set_index("rel").loc[files]
        dur, sp, lv, sh = (feat[c].to_numpy() for c in ("dur", "speech_dur", "rms_db", "speech_ratio"))
        fk = cache_key(MODEL, split, files, mode="full")
        full = l2n(np.load(fk))
        fv = l2n(full + l2n(np.load(cache_key(MODEL, split, files, **VAD))))
        gain = np.zeros(len(pairs))
        if lam > 0:
            z = np.load(fk.with_name(f"{split}_speakers.npz"))
            S = [l2n(z[k]) for k in ("full", "c0", "c1")]
            gain = np.maximum(0, np.max([(A[ia] * B[ib]).sum(1) for A in S for B in S], 0) - (S[0][ia] * S[0][ib]).sum(1))
        gate = (np.minimum(sh[ia], sh[ib]) < 0.5) | (np.minimum(sp[ia], sp[ib]) < 1.0) | (np.abs(lv[ia] - lv[ib]) > 10)
        s = np.where(gate, cal["fv"]((fv[ia] * fv[ib]).sum(1) + lam * gain, dur[ia], dur[ib]),
                     cal["full"]((full[ia] * full[ib]).sum(1) + lam * gain, dur[ia], dur[ib]))
        return pairs, y, s, np.minimum(dur[ia], dur[ib]) < 2

    if a.private:
        pairs, _, s, _ = score("private", new, a.lam)
        out = ROOT / "submissions/final_private"
        res = out / "results.txt"
        write_submission(res, pairs, s)
        (out / f"results_{Path(a.zip).stem}.txt").write_text(res.read_text())
        lines = res.read_text().splitlines()
        bad = sum(1 for l in lines if len(l.split()) != 3 or not np.isfinite(float(l.split()[2])))
        order = sum(1 for l, (x, y_) in zip(lines, pairs) if l.split()[:2] != [x, y_])
        print(f"lines: {len(lines)} / {len(pairs)}   mismatched pairs: {order}   bad score lines: {bad}")
        assert len(lines) == len(pairs) and order == 0 and bad == 0, "checks failed"
        (out / a.zip).unlink(missing_ok=True)
        subprocess.run(["zip", "-q", "-j", str(out / a.zip), str(res)], check=True)
        print("ready:", out / a.zip)
        return
    meta = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")
    for split in ("public_rumble_v2", "public_noisy", "public_realnoise"):
        lams = [0.0, a.lam] if split == "public_rumble_v2" else [0.0]  # speaker sets exist for silver and private only
        for lam in lams:
            for nm, cal in (("current calibration", old), ("+ short-crop calibration", new)):
                pairs, y, s, short = score(split, cal, lam)
                cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
                print(f"{split:17s} lam {lam:<4} {nm:26s} all {eer(s, y):.3f}  cleaned {eer(s[cl], y[cl]):.3f}  "
                      f"shorter clip < 2 s ({short.mean():.0%}) {eer(s[short], y[short]):.2f}", flush=True)


if __name__ == "__main__":
    main()
