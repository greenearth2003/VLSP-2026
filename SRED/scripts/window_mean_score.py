"""Mean-of-window-pairs scoring: cut both clips into W-s windows (hop W/2; a clip of W s or less is one window, not
tiled), embed every window, and score a trial by the MEAN cosine over all window pairs (A_i, B_j). That mean equals the
dot product of the two clips' average window embeddings (not re-normalised), so a clip whose windows disagree (two
voices, noise bursts) gets a shorter average and lower scores.

    python scripts/window_mean_score.py --splits public_rumble_v2 public_noisy public_realnoise [--win 3]

Compares, raw and duration-calibrated (QMF fitted on train-cohort trials scored the same way), EER on all trials and the
cleaned list: whole clip (current) / mean of window pairs / mean of A-whole vs B-windows and B-whole vs A-windows.
Per-file mean window embeddings cached in exp/emb/<model>/<split>_wmean<W>.npz (rows in the split's file order).
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


class Clips(torch.utils.data.Dataset):
    def __init__(self, files, root, win):
        self.files, self.root, self.W = files, root, int(win * SR)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x, _ = sf.read(str(self.root / self.files[i]), dtype="float32")
        if len(x) <= self.W:
            return i, x[None]  # one window = the whole clip
        st = list(range(0, len(x) - self.W + 1, self.W // 2))
        return i, np.stack([x[s:s + self.W] for s in st])


@torch.inference_mode()
def mean_windows(files, root, win, cache):
    if cache.exists():
        z = np.load(cache)
        return z["m"], z["n"]
    model = load_model(MODEL).cuda().eval()
    m = np.zeros((len(files), 192), np.float32)
    n = np.zeros(len(files), np.int32)
    dl = torch.utils.data.DataLoader(Clips(files, root, win), batch_size=None, num_workers=8, prefetch_factor=8)
    buf, own = [], []

    def flush():
        e = F.normalize(model(torch.from_numpy(np.concatenate(buf)).cuda()), dim=-1).cpu().numpy()
        k = 0
        for i, c in own:
            m[i], n[i] = e[k:k + c].mean(0), c
            k += c
        buf.clear(), own.clear()

    for i, w in dl:
        w = w.numpy() if torch.is_tensor(w) else w
        if w.shape[1] != int(win * SR):  # short clip: embed alone (its own length)
            m[i], n[i] = F.normalize(model(torch.from_numpy(w).cuda()), dim=-1).cpu().numpy()[0], 1
            continue
        buf.append(w)
        own.append((int(i), len(w)))
        if sum(len(b) for b in buf) >= 256:
            flush()
    if buf:
        flush()
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
    ap.add_argument("--splits", nargs="+", default=["public_rumble_v2", "public_noisy", "public_realnoise"])
    ap.add_argument("--win", type=float, default=3.0)
    a = ap.parse_args()
    from make_segment_submission import qmf_feats
    from sklearn.linear_model import LogisticRegression
    from sred.calibrate import QMF2
    from sred.data import cohort_subset, read_train_list
    tag = f"wmean{a.win:g}"

    # train cohort: whole-clip and mean-window embeddings -> one duration QMF per scoring variant
    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    cdur = np.array([json.loads((ROOT / "lists/train_lengths.json").read_text())[r] / 16000 for r in cf])
    ck = cache_key(MODEL, "traincohort5", cf, mode="full")
    Cfull = l2n(np.load(ck))
    files_c, root_c, _ = split_files("traincohort5")
    Cm, _ = mean_windows(files_c, root_c, a.win, ck.with_name(f"traincohort5_{tag}.npz"))
    I, J, yc = QMF2(["dur"]).trials(cs)

    def variants(full, mw, i, j):
        return {"whole clip (current)": (full[i] * full[j]).sum(1),
                "mean of all window pairs": (mw[i] * mw[j]).sum(1),
                "mean of whole-vs-windows, both ways": 0.5 * ((full[i] * mw[j]).sum(1) + (mw[i] * full[j]).sum(1))}

    cal = {}
    for k, s in variants(Cfull, Cm, I, J).items():
        X = qmf_feats(s, cdur[I], cdur[J])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        cal[k] = (mu, sd, LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((X - mu) / sd, yc))

    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
    meta = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")
    cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
    rows = []
    for split in a.splits:
        fk = cache_key(MODEL, split, files, mode="full")
        full = l2n(np.load(fk))
        mw, nw = mean_windows(files, ROOT / "data" / split, a.win, fk.with_name(f"{split}_{tag}.npz"))
        dur = pd.read_csv(ROOT / f"exp/analysis/features_{split}.csv").set_index("rel").loc[files].dur.to_numpy()
        for k, s in variants(full, mw, ia, ib).items():
            mu, sd, lr = cal[k]
            q = lr.decision_function((qmf_feats(s, dur[ia], dur[ib]) - mu) / sd)
            rows.append(dict(split=split, scoring=k, raw_all=round(eer(s, y), 2), raw_cleaned=round(eer(s[cl], y[cl]), 2),
                             cal_all=round(eer(q, y), 2), cal_cleaned=round(eer(q[cl], y[cl]), 2)))
        print(f"{split}: windows per clip median {np.median(nw):.0f}, clips with one window {np.mean(nw == 1):.0%}", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    df.to_csv(ROOT / f"exp/analysis/window_mean_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
