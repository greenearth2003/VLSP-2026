"""Teammate's C2 + Q3 with transductive kNN smoothing of the test embeddings (sred.transductive.knn_smooth: each test
embedding is pulled toward its nearest other test files above a similarity threshold, 2 passes, only files that form a
cluster of >= 3). Uses unlabelled test audio, so it is only valid if the rules allow transductive scoring.

    python scripts/c2_transductive.py --split silver                       # EER for a few thresholds
    python scripts/c2_transductive.py --split private --tau 0.6 --zip result-private-C2Q3-t1.zip

Embeddings: the whole-clip C2 embeddings from scripts/c2_tta.py's cache (exp/emb/ext_C2/<split>_tta.npz), or computed
here (whole clip only) into exp/emb/ext_C2/<split>_whole.npz.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/mnt/tsharp/thucth/tmp/claude-1004/-home-thuc-thuc/fa0eee2f-9671-4c94-8b41-6d41fb8da6d5/scratchpad")
import c2q3  # noqa: E402
from c2_tta import SPLITS, eer  # noqa: E402
from sred.data import read_trials, unique_files  # noqa: E402
from sred.transductive import knn_smooth  # noqa: E402


class Whole(torch.utils.data.Dataset):
    def __init__(self, files, root):
        self.files, self.root = files, root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = c2q3.load_wav(self.root / self.files[i])
        return i, x, c2q3.duration_s(x), c2q3.snr_est(x)


@torch.inference_mode()
def embeddings(split, files, kind="whole"):
    for name in (f"{split}_tta.npz", f"{split}_whole.npz"):
        p = ROOT / "exp/emb/ext_C2" / name
        if p.exists() and (kind == "whole" or name.endswith("_tta.npz")):
            z = np.load(p)
            assert list(z["files"]) == files
            return z[kind], z["dur"], z["snr"]
    from c2_loader import load_c2_safe
    model = load_c2_safe("cuda")
    E, dur, snr = np.zeros((len(files), 192), np.float32), np.zeros(len(files)), np.zeros(len(files))
    dl = torch.utils.data.DataLoader(Whole(files, SPLITS[split][1]), batch_size=None, num_workers=6, prefetch_factor=4,
                                     collate_fn=lambda b: b)
    for k, (i, x, d, q) in enumerate(dl):
        xw = x[: 60 * 16000] if len(x) >= 8000 else np.tile(x, int(np.ceil(8000 / max(1, len(x)))))[:8000]
        E[i] = F.normalize(model(torch.from_numpy(np.ascontiguousarray(xw)).cuda()[None]), dim=-1)[0].float().cpu().numpy()
        dur[i], snr[i] = d, q
        if k % 5000 == 0:
            print(f"  {split} whole-clip: {k}/{len(files)}", flush=True)
    p = ROOT / f"exp/emb/ext_C2/{split}_whole.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, files=np.array(files), whole=E, dur=dur, snr=snr)
    return E, dur, snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--emb", choices=["whole", "tta"], default="whole", help="embedding to smooth (tta = mean of TTA views)")
    a = ap.parse_args()
    trials, _ = SPLITS[a.split]
    pairs, y = read_trials(trials) if trials else read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
    E, dur, snr = embeddings(a.split, files, a.emb)
    q3 = c2q3.Q3()
    score = lambda X: q3.score((X[ia] * X[ib]).sum(1), dur[ia], dur[ib], snr[ia], snr[ib])  # noqa: E731
    if a.split == "silver":
        meta = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")
        cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
        s = score(E)
        print(f"C2+Q3 ({a.emb}) non-transductive: all {eer(s, y):.3f} cleaned {eer(s[cl], y[cl]):.3f}", flush=True)
        for tau in (0.5, 0.6, 0.7):
            Es, mask = knn_smooth(E, 20, 0.5, tau, 2, min_cluster=3, return_mask=True)
            s = score(Es)
            print(f"C2+Q3 ({a.emb}) transductive (k 20, w 0.5, tau {tau}): {mask.mean():.0%} of files smoothed; "
                  f"all {eer(s, y):.3f} cleaned {eer(s[cl], y[cl]):.3f}", flush=True)
        return
    Es, mask = knn_smooth(E, 20, 0.5, a.tau, 2, min_cluster=3, return_mask=True)
    s = score(Es)
    print(f"private: {mask.mean():.0%} of files smoothed (tau {a.tau})")
    out = ROOT / "submissions/final_private"
    res = out / "results.txt"
    res.write_text("".join(f"{x} {w} {v:.6f}\n" for (x, w), v in zip(pairs, s)))
    (out / f"results_{Path(a.zip).stem}.txt").write_text(res.read_text())
    lines = res.read_text().splitlines()
    bad = sum(1 for l in lines if len(l.split()) != 3 or not np.isfinite(float(l.split()[2])))
    order = sum(1 for l, (x, w) in zip(lines, pairs) if l.split()[:2] != [x, w])
    print(f"lines {len(lines)} / {len(pairs)}, mismatched {order}, bad {bad}")
    assert len(lines) == len(pairs) and order == 0 and bad == 0
    (out / a.zip).unlink(missing_ok=True)
    subprocess.run(["zip", "-q", "-j", str(out / a.zip), str(res)], check=True)
    print("ready:", out / a.zip)


if __name__ == "__main__":
    main()
