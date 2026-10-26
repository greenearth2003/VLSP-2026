"""End-to-end inference of the S5 + Q3 system: trial list + audio folder -> score file.

  python infer.py --trials <trials.txt> --audio-dir <dir> --out <results.txt> [--device cuda] [--workers 4]

trials.txt : one trial per line, "fileA fileB" or "fileA fileB target|nontarget" (paths relative to --audio-dir).
results.txt: one line per trial, same order, "fileA fileB score" with 6 decimals (higher = more likely the same
             speaker). This is the VLSP 2026 SRED submission format.
If the trial file has labels, EER and minDCF(0.01) are printed. Embeddings, durations and SNRs are cached in
<out>.cache.npz, so a second run only re-scores.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5q3 import Q3, count_params, duration_s, embed, load_s5, load_wav, snr_est  # noqa: E402


def read_trials(path):
    pairs, labels = [], []
    for ln in Path(path).read_text().splitlines():
        p = ln.split()
        if not p:
            continue
        pairs.append((p[0], p[1]))
        if len(p) > 2:
            labels.append(1 if p[2] in ("target", "1") else 0)
    return pairs, (np.array(labels) if len(labels) == len(pairs) else None)


class _Files(torch.utils.data.Dataset):
    def __init__(self, files, root):
        self.files, self.root = files, Path(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = load_wav(self.root / self.files[i])
        return i, torch.from_numpy(x), duration_s(x), snr_est(x)


def eer(scores, labels):
    o = np.argsort(scores)
    y = labels[o]
    frr = np.r_[0, np.cumsum(y) / y.sum()]
    far = np.r_[1, 1 - np.cumsum(1 - y) / (1 - y).sum()]
    k = np.argmin(np.abs(frr - far))
    mdcf = np.min(0.01 * frr + 0.99 * far) / 0.01
    return 100 * (frr[k] + far[k]) / 2, float(mdcf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    pairs, labels = read_trials(a.trials)
    files = list(dict.fromkeys(f for p in pairs for f in p))
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs])
    ib = np.array([pos[y] for _, y in pairs])
    cache = Path(a.out + ".cache.npz")
    if cache.exists() and list(np.load(cache)["files"]) == files:
        z = np.load(cache)
        E, dur, snr = z["emb"], z["dur"], z["snr"]
        print(f"loaded cache {cache}")
    else:
        model = load_s5(a.device)
        print(f"S5 loaded: {count_params(model):,} parameters, device {a.device}; {len(pairs):,} trials, "
              f"{len(files):,} files", flush=True)
        E = np.zeros((len(files), 192), np.float32)
        dur, snr = np.zeros(len(files)), np.zeros(len(files))
        dl = torch.utils.data.DataLoader(_Files(files, a.audio_dir), batch_size=None, num_workers=a.workers,
                                         prefetch_factor=2 if a.workers else None)
        t0 = time.time()
        for k, (i, x, d, q) in enumerate(dl):
            E[i] = embed(model, x.numpy(), a.device)
            dur[i], snr[i] = d, q
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1:,}/{len(files):,} files, {(time.time() - t0) / (k + 1) * 1000:.1f} ms/file", flush=True)
        np.savez(cache, files=np.array(files), emb=E, dur=dur, snr=snr)
    E = E / np.linalg.norm(E, axis=-1, keepdims=True).clip(1e-8)
    cos = np.sum(E[ia] * E[ib], 1)
    s = Q3().score(cos, dur[ia], dur[ib], snr[ia], snr[ib])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for (x, y), v in zip(pairs, s):
            assert np.isfinite(v)
            f.write(f"{x} {y} {v:.6f}\n")
    print(f"wrote {a.out} ({len(pairs):,} lines)")
    if labels is not None:
        e, m = eer(s, labels)
        ec, _ = eer(cos, labels)
        print(f"EER {e:.3f}%  minDCF(0.01) {m:.4f}   (raw cosine EER {ec:.3f}%)")


if __name__ == "__main__":
    main()
