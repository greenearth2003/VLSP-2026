"""Test-time views + mean score (the team's TTA proposal, 2026-10-21):
  clip < 2 s        -> repeated until it is at least 4 s long (one view)
  clip 2-4 s        -> itself (one view)
  clip > 4 s        -> itself + consecutive 4-s windows covering the whole clip (a last piece < 2 s is replaced by the
                       clip's last 4 s)
A trial scores by the MEAN cosine over all view pairs (A views x B views) = dot product of the two clips' mean view
embeddings. Reported raw and with a duration QMF fitted on train-cohort trials scored the same way.

    python scripts/tta_views.py --splits public_rumble_v2 [public_noisy public_realnoise]

Model-agnostic: `views()` and `mean_view_embedding()` only need an embed(wav) -> L2-normalised vector function, so the
same protocol runs with any model (e.g. C2: replace `embed`).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_embed import split_files  # noqa: E402
from sred.data import read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.models import load_model  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "exp/ft/S3_E2b_X2/avg.pt"
SR = 16000


def views(x, short=2.0, win=4.0):
    """the list of waveforms a clip is scored with"""
    W, L = int(win * SR), len(x)
    if L < short * SR:
        return [np.tile(x, int(np.ceil(W / max(1, L))) + (1 if L and (np.ceil(W / L) * L) <= W else 0))]  # >= 4 s
    if L <= W:
        return [x]
    out = [x]
    for s in range(0, L, W):
        seg = x[s:s + W]
        if len(seg) < short * SR:  # short tail: use the clip's last 4 s instead
            seg = x[L - W:]
        out.append(seg)
    return out


@torch.inference_mode()
def mean_view_embedding(files, root, embed, cache):
    if cache.exists():
        z = np.load(cache)
        return z["m"], z["n"]
    m = np.zeros((len(files), 192), np.float32)
    n = np.zeros(len(files), np.int32)
    for i, f in enumerate(files):
        x, _ = sf.read(str(root / f), dtype="float32")
        v = views(x)
        m[i] = np.mean([embed(w) for w in v], 0)
        n[i] = len(v)
        if i % 5000 == 0:
            print(f"  {root.name}: {i}/{len(files)}", flush=True)
    np.savez(cache, m=m, n=n)
    return m, n


def eer(s, y):
    o = np.argsort(s)
    yo = y[o]
    frr = np.cumsum(yo) / yo.sum()
    far = 1 - np.cumsum(1 - yo) / (1 - yo).sum()
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["public_rumble_v2"])
    a = ap.parse_args()
    from make_segment_submission import qmf_feats
    from sklearn.linear_model import LogisticRegression
    from sred.calibrate import QMF2
    from sred.data import cohort_subset, read_train_list
    model = load_model(MODEL).cuda().eval()
    embed = lambda w: F.normalize(model(torch.from_numpy(w).cuda()[None]), dim=-1)[0].float().cpu().numpy()  # noqa: E731

    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    cdur = np.array([json.loads((ROOT / "lists/train_lengths.json").read_text())[r] / 16000 for r in cf])
    ck = cache_key(MODEL, "traincohort5", cf, mode="full")
    Cf = l2n(np.load(ck))
    files_c, root_c, _ = split_files("traincohort5")
    Cm, _ = mean_view_embedding(files_c, root_c, embed, ck.with_name("traincohort5_tta4.npz"))
    I, J, yc = QMF2(["dur"]).trials(cs)

    def fit(s):
        X = qmf_feats(s, cdur[I], cdur[J])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((X - mu) / sd, yc)
        return lambda s2, d1, d2: lr.decision_function((qmf_feats(s2, d1, d2) - mu) / sd)
    q_whole, q_tta = fit((Cf[I] * Cf[J]).sum(1)), fit((Cm[I] * Cm[J]).sum(1))

    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
    meta = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")
    cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
    for split in a.splits:
        fk = cache_key(MODEL, split, files, mode="full")
        full = l2n(np.load(fk))
        M, nv = mean_view_embedding(files, ROOT / "data" / split, embed, fk.with_name(f"{split}_tta4.npz"))
        dur = pd.read_csv(ROOT / f"exp/analysis/features_{split}.csv").set_index("rel").loc[files].dur.to_numpy()
        short = np.minimum(dur[ia], dur[ib]) < 2
        print(f"\n{split}: views per clip median {np.median(nv):.0f}, mean {nv.mean():.1f}")
        for nm, raw, q in (("whole clip (current)", (full[ia] * full[ib]).sum(1), q_whole), ("TTA views, mean score", (M[ia] * M[ib]).sum(1), q_tta)):
            cal = q(raw, dur[ia], dur[ib])
            print(f"  {nm:24s} raw: all {eer(raw, y):.2f} cleaned {eer(raw[cl], y[cl]):.2f} | calibrated: all {eer(cal, y):.2f} "
                  f"cleaned {eer(cal[cl], y[cl]):.2f} | shorter clip < 2 s (calibrated) {eer(cal[short], y[short]):.2f}", flush=True)


if __name__ == "__main__":
    main()
