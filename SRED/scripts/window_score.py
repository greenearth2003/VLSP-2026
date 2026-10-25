"""Windowed ("best-matched segment") scoring: each clip -> energy-selected speech windows of W s (hop W/2, no VAD
model); a trial scores by its best-matching window pair, combined with the whole-clip score and calibrated on TRAIN
(a logistic regression over [whole-clip cosine, max window cosine, mean of the top-3 window cosines, window counts,
durations], fitted on synthetic train-cohort trials; no test labels). Handles clips with several voices at different
times and pairs of very different lengths.

    python scripts/window_score.py --model exp/ft/S3_E2b_X2/avg.pt --win 1.0 [--splits public_noisy public_realnoise public_rumble]
    python scripts/window_score.py --model ... --win 1.0 --private --zip result-private-<name>.zip

Reports EER, accuracy at the EER point, accuracy at the threshold fixed on the synthetic stand-in, and EER on pairs
whose lengths differ >= 4x. Window embeddings cached in exp/emb/<model>/<split>_win<W>.npz.
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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.metrics import compute_eer  # noqa: E402
from sred.models import load_model  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR, KMAX = 16000, 24


def speech_windows(x, win, min_speech=0.5):
    """Start samples of W-s windows (hop W/2) whose share of speech frames (energy > clip floor + 6 dB) >= min_speech."""
    W, H = int(win * SR), int(win * SR / 2)
    if len(x) <= W:
        return [0]
    fr = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    db = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
    sp = db > np.percentile(db, 20) + 6
    starts = list(range(0, len(x) - W + 1, H))
    share = [sp[s // 160:(s + W) // 160].mean() if (s + W) // 160 > s // 160 else 0 for s in starts]
    keep = [s for s, v in zip(starts, share) if v >= min_speech]
    if not keep:
        keep = [starts[int(np.argmax(share))]]
    if len(keep) > KMAX:
        keep = [keep[i] for i in np.linspace(0, len(keep) - 1, KMAX).astype(int)]
    return keep


class _WindowSet(torch.utils.data.Dataset):
    """Loader-side work (read, tile short clips, pick speech windows) so the GPU sees large batches."""

    def __init__(self, files, root, win, min_speech=0.5):
        self.files, self.root, self.win, self.min_speech = files, root, win, min_speech

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        W = int(self.win * SR)
        x, _ = sf.read(str(self.root / self.files[i]), dtype="float32")
        if len(x) < W:
            x = np.tile(x, int(np.ceil(W / max(1, len(x)))))[:W] if len(x) else np.zeros(W, np.float32)
        return i, np.stack([x[s:s + W] for s in speech_windows(x, self.win, self.min_speech)])


@torch.inference_mode()
def embed_windows(model_path, files, root, win, cache, batch_windows=384, min_speech=0.5):
    """Normalised embeddings of each file's speech windows: E [files, KMAX, 192], n = windows per file.
    Windows of many files are batched together (all have the same length); per-window results are identical
    to embedding each file alone (eval mode, no cross-sample ops)."""
    if cache.exists():
        z = np.load(cache)
        return z["E"], z["n"]
    model = load_model(model_path).cuda().eval()
    E = np.zeros((len(files), KMAX, 192), np.float32)
    n = np.zeros(len(files), np.int32)
    dl = torch.utils.data.DataLoader(_WindowSet(files, root, win, min_speech), batch_size=None, num_workers=8, prefetch_factor=8)
    buf, owners = [], []

    def flush():
        e = F.normalize(model(torch.from_numpy(np.concatenate(buf)).cuda()), dim=-1).cpu().numpy()
        k = 0
        for i, m in owners:
            E[i, :m], n[i] = e[k:k + m], m
            k += m
        buf.clear(), owners.clear()

    for j, (i, segs) in enumerate(dl):
        buf.append(segs.numpy() if torch.is_tensor(segs) else segs)
        owners.append((int(i), len(segs)))
        if sum(len(b) for b in buf) >= batch_windows:
            flush()
        if j % 4000 == 0:
            print(f"  {root.name} windows: {j}/{len(files)}", flush=True)
    if buf:
        flush()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, E=E, n=n)
    return E, n


def pair_feats(full, E, n, dur, ia, ib, chunk=20000):
    out = np.zeros((len(ia), 7), np.float32)
    for s in range(0, len(ia), chunk):
        a, b = ia[s:s + chunk], ib[s:s + chunk]
        S = np.einsum("pkd,pld->pkl", E[a], E[b])
        mask = (np.arange(KMAX)[None, :, None] < n[a][:, None, None]) & (np.arange(KMAX)[None, None, :] < n[b][:, None, None])
        S = np.where(mask, S, -2.0).reshape(len(a), -1)
        top = -np.sort(-S, axis=1)[:, :3]
        top3 = np.where(top > -1.5, top, np.nan)
        out[s:s + chunk, 0] = (full[a] * full[b]).sum(1)
        out[s:s + chunk, 1] = top[:, 0]
        out[s:s + chunk, 2] = np.nanmean(top3, axis=1)
        out[s:s + chunk, 3] = np.log(np.minimum(n[a], n[b]))
        out[s:s + chunk, 4] = np.log(np.maximum(n[a], n[b]))
        out[s:s + chunk, 5] = np.log(np.maximum(np.minimum(dur[a], dur[b]), 0.05))
        out[s:s + chunk, 6] = np.log(np.maximum(np.maximum(dur[a], dur[b]), 0.05))
    return out


def design(X):
    """score columns, durations/counts, and score x duration interactions (as QMF)."""
    return np.column_stack([X, X[:, 0] * X[:, 5], X[:, 1] * X[:, 5], X[:, 1] * X[:, 3]])


class Calib:
    def fit(self, full, E, n, dur, spk, n_non=300000, seed=0):
        rng = np.random.default_rng(seed)
        N = len(spk)
        I, J = rng.integers(0, N, n_non), rng.integers(0, N, n_non)
        order = np.argsort(spk, kind="stable")
        ti, tj = [], []
        for k in range(N - 1):
            for m in range(k + 1, min(N, k + 10)):
                if spk[order[m]] != spk[order[k]]:
                    break
                ti.append(order[k]); tj.append(order[m])
        I, J = np.concatenate([I, ti]), np.concatenate([J, tj])
        keep = I != J
        I, J = I[keep], J[keep]
        y = (spk[I] == spk[J]).astype(int)
        X = design(pair_feats(full, E, n, dur, I, J))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit((X - self.mu) / self.sd, y)
        return self

    def score(self, X, cols=None):
        Z = (design(X) - self.mu) / self.sd
        return self.lr.decision_function(Z)


def eer_thr(s, y):
    o = np.argsort(s)
    frr = np.cumsum(y[o]) / y.sum()
    far = 1 - np.cumsum(1 - y[o]) / (1 - y).sum()
    return float(s[o][np.argmin(np.abs(frr - far))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--splits", nargs="+", default=["public_noisy", "public_realnoise", "public_rumble"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--zip", default=None)
    a = ap.parse_args()
    tag = f"win{a.win:g}"
    ci = cohort_subset(read_train_list(), per_spk=5)
    cf, cs = [r for _, r in ci], np.array([s for s, _ in ci])
    cdur = np.array([json.loads((ROOT / "lists/train_lengths.json").read_text())[r] / 16000 for r in cf])
    Cfull = l2n(np.load(cache_key(a.model, "traincohort5", cf, mode="full")))
    CE, Cn = embed_windows(a.model, cf, TRAIN_DIR, a.win, cache_key(a.model, "traincohort5", cf, mode="full").with_name(f"traincohort5_{tag}.npz"))
    cal = Calib().fit(Cfull, CE, Cn, cdur, cs)
    # whole-clip QMF only, for the comparison row (same train trials, same model)
    if a.private:
        pairs, _ = read_trials(ROOT / "submissions/final_private/trials_2col.txt")
        files = unique_files(pairs)
        pos = {f: i for i, f in enumerate(files)}
        ia = np.array([pos[p] for p, _ in pairs]); ib = np.array([pos[q] for _, q in pairs])
        dur = np.load(ROOT / "lists/private_durations.npy")
        full = l2n(np.load(cache_key(a.model, "private", files, mode="full")))
        E, n = embed_windows(a.model, files, ROOT / "data/Vi-SRED26/private_test/data", a.win,
                             cache_key(a.model, "private", files, mode="full").with_name(f"private_{tag}.npz"))
        s = cal.score(pair_feats(full, E, n, dur, ia, ib))
        out = ROOT / "submissions/final_private"
        (out / "results.txt").write_text("".join(f"{p} {q} {v:.6f}\n" for (p, q), v in zip(pairs, s)))
        lines = (out / "results.txt").read_text().splitlines()
        assert len(lines) == len(pairs) and all(np.isfinite(float(l.split()[2])) for l in lines)
        (out / f"results_{Path(a.zip).stem}.txt").write_text((out / "results.txt").read_text())
        (out / a.zip).unlink(missing_ok=True)
        subprocess.run(["zip", "-q", "-j", str(out / a.zip), str(out / "results.txt")], check=True)
        print(f"lines {len(lines)} / {len(pairs)}; ready: {out / a.zip}")
        return
    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[p] for p, _ in pairs]); ib = np.array([pos[q] for _, q in pairs])
    dur = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel").loc[files].dur.values
    ratio = np.maximum(dur[ia], dur[ib]) / np.maximum(np.minimum(dur[ia], dur[ib]), 0.05)
    ext = ratio >= 4
    rows, thr_fixed = [], {}
    for split in a.splits:
        full = l2n(np.load(cache_key(a.model, split, files, mode="full")))
        E, n = embed_windows(a.model, files, ROOT / "data" / split, a.win,
                             cache_key(a.model, split, files, mode="full").with_name(f"{split}_{tag}.npz"))
        X = pair_feats(full, E, n, dur, ia, ib)
        for name, s in (("whole clip", Calib.__new__(Calib)), ("windowed", cal)):
            if name == "whole clip":
                from sred.calibrate import QMF
                sc = QMF(use_asnorm=False).fit(Cfull, cs, cdur).score(full, dur, ia, ib)
            else:
                sc = cal.score(X)
            t = eer_thr(sc, y)
            thr_fixed.setdefault(name, t if split == "public_noisy" else None)
            tf = thr_fixed[name] if thr_fixed[name] is not None else t
            acc_eer = ((sc > t) == (y == 1)).mean()
            acc_fix = ((sc > tf) == (y == 1)).mean()
            rows.append(dict(split=split, scoring=name, win=a.win, EER=round(compute_eer(sc, y), 3), acc_at_EER=round(100 * acc_eer, 2),
                             acc_fixed_thr=round(100 * acc_fix, 2), EER_len_ratio_ge4=round(compute_eer(sc[ext], y[ext]), 3)))
            print(rows[-1], flush=True)
    print(f"(pairs with length ratio >= 4: {ext.sum()} of {len(y)}; fixed threshold = EER point on public_noisy)")
    pd.DataFrame(rows).to_csv(ROOT / f"exp/analysis/window_score_{Path(a.model).parent.name}_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
