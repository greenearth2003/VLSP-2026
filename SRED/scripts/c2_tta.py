"""The team's TTA protocol on the teammate's C2 model (+ their frozen Q3 calibration):
  clip < 2 s -> repeated until >= 4 s (one view); 2-4 s -> itself; > 4 s -> itself + consecutive 4-s windows covering
  the clip (a tail < 2 s is replaced by the last 4 s). Trial score = mean cosine over all view pairs
  (= dot product of mean view embeddings), then Q3 with the clip's own duration and SNR, as their pipeline does.

    python scripts/c2_tta.py --split silver     # embeds + prints EER (whole clip vs TTA, raw and Q3)
    python scripts/c2_tta.py --split private --zip result-private-C2Q3-tta-t0.zip

Cache: exp/emb/ext_C2/<split>_tta.npz (files, whole, tta, dur, snr, nviews).
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
sys.path.insert(0, "/mnt/tsharp/thucth/tmp/claude-1004/-home-thuc-thuc/fa0eee2f-9671-4c94-8b41-6d41fb8da6d5/scratchpad")
from c2_loader import load_c2_safe  # noqa: E402  (teammate package, safe weights-only load)
import c2q3  # noqa: E402
from sred.data import read_trials, unique_files  # noqa: E402

SR = 16000
SPLITS = {"silver": (None, ROOT / "data/public_rumble_v2"),
          "private": (ROOT / "submissions/final_private/trials_2col.txt", ROOT / "data/Vi-SRED26/private_test/data")}


class Clips(torch.utils.data.Dataset):
    def __init__(self, files, root):
        self.files, self.root = files, root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = c2q3.load_wav(self.root / self.files[i])
        L, W = len(x), 4 * SR
        if L < 2 * SR:  # repeat until at least 4 s
            singles, wins = [np.tile(x, int(np.ceil(W / max(1, L))) + (1 if L and np.ceil(W / L) * L <= W else 0))], []
        elif L <= W:
            singles, wins = [x], []
        else:
            singles, wins = [x], []
            for s in range(0, L, W):
                seg = x[s:s + W]
                wins.append(seg if len(seg) >= 2 * SR else x[L - W:])
            wins = [w for w in wins if len(w) == W]
        return i, x, singles, np.stack(wins) if wins else None, c2q3.duration_s(x), c2q3.snr_est(x)


@torch.inference_mode()
def run(split, model):
    trials, root = SPLITS[split]
    pairs, y = read_trials(trials) if trials else read_trials()
    files = unique_files(pairs)
    cache = ROOT / f"exp/emb/ext_C2/{split}_tta.npz"
    if cache.exists():
        z = np.load(cache)
        return pairs, y, files, z
    cache.parent.mkdir(parents=True, exist_ok=True)
    whole, tta = np.zeros((len(files), 192), np.float32), np.zeros((len(files), 192), np.float32)
    dur, snr, nv = np.zeros(len(files)), np.zeros(len(files)), np.zeros(len(files), np.int32)
    emb = lambda t: F.normalize(model(t.cuda()), dim=-1).float().cpu().numpy()  # noqa: E731
    dl = torch.utils.data.DataLoader(Clips(files, root), batch_size=None, num_workers=6, prefetch_factor=4,
                                     collate_fn=lambda b: b)
    for k, (i, x, singles, wins, d, q) in enumerate(dl):
        xw = x[: 60 * SR] if len(x) >= SR // 2 else np.tile(x, int(np.ceil(SR / 2 / max(1, len(x)))))[: SR // 2]
        e_whole = emb(torch.from_numpy(np.ascontiguousarray(xw))[None])[0]  # identical to c2q3.embed
        views = [e_whole if (len(singles[0]) == len(x)) else emb(torch.from_numpy(np.ascontiguousarray(singles[0][: 60 * SR]))[None])[0]]
        if wins is not None:
            views += list(emb(torch.from_numpy(wins)))
        whole[i], tta[i], dur[i], snr[i], nv[i] = e_whole, np.mean(views, 0), d, q, len(views)
        if k % 4000 == 0:
            print(f"  {split}: {k}/{len(files)}", flush=True)
    np.savez(cache, files=np.array(files), whole=whole, tta=tta, dur=dur, snr=snr, nviews=nv)
    return pairs, y, files, np.load(cache)


def eer(s, y):
    o = np.argsort(s)
    yo = y[o]
    frr = np.cumsum(yo) / yo.sum()
    far = 1 - np.cumsum(1 - yo) / (1 - yo).sum()
    k = np.argmin(np.abs(frr - far))
    return 100 * (frr[k] + far[k]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--zip", default=None)
    a = ap.parse_args()
    model = load_c2_safe("cuda")
    pairs, y, files, z = run(a.split, model)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
    q3 = c2q3.Q3()
    S = {}
    for nm, E in (("whole", z["whole"]), ("tta", z["tta"])):
        cos = (E[ia] * E[ib]).sum(1)
        S[nm] = (cos, q3.score(cos, z["dur"][ia], z["dur"][ib], z["snr"][ia], z["snr"][ib]))
    print(f"{a.split}: views per clip mean {z['nviews'].mean():.2f}")
    if a.split == "silver":
        meta = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/hf_repo/standins/trial_metadata.csv")
        cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(meta, how="left").in_cleaned.fillna(False).to_numpy(bool)
        dmin = np.minimum(z["dur"][ia], z["dur"][ib])
        for nm, (cos, q) in S.items():
            print(f"  C2 {nm:5s} raw cosine: all {eer(cos, y):.3f} cleaned {eer(cos[cl], y[cl]):.3f} | Q3: all {eer(q, y):.3f} "
                  f"cleaned {eer(q[cl], y[cl]):.3f} | shorter clip < 2 s {eer(q[dmin < 2], y[dmin < 2]):.2f}, >= 4 s {eer(q[dmin >= 4], y[dmin >= 4]):.2f}")
        ref = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/incoming/c2q3/evidence/silver_C2_Q3.txt", sep=" ", header=None).iloc[:, 2].to_numpy()
        print(f"  reproduction of their silver file: max |diff| {np.abs(ref - S['whole'][1]).max():.2e}")
        return
    out = ROOT / "submissions/final_private"
    res = out / "results.txt"
    res.write_text("".join(f"{x} {w} {v:.6f}\n" for (x, w), v in zip(pairs, S["tta"][1])))
    (out / f"results_{Path(a.zip).stem}.txt").write_text(res.read_text())
    lines = res.read_text().splitlines()
    bad = sum(1 for l in lines if len(l.split()) != 3 or not np.isfinite(float(l.split()[2])))
    order = sum(1 for l, (x, w) in zip(lines, pairs) if l.split()[:2] != [x, w])
    print(f"lines {len(lines)} / {len(pairs)}, mismatched {order}, bad {bad}")
    assert len(lines) == len(pairs) and order == 0 and bad == 0
    ref = pd.read_csv("/mnt/tsharp/thucth/dataset/vlsp2026-sred/incoming/c2q3/evidence/private_C2_Q3_results.txt", sep=" ", header=None).iloc[:, 2].to_numpy()
    print(f"whole-clip reproduction of their private file: max |diff| {np.abs(ref - S['whole'][1]).max():.2e}; "
          f"TTA vs their scores correlation {np.corrcoef(ref, S['tta'][1])[0, 1]:.4f}")
    (out / a.zip).unlink(missing_ok=True)
    subprocess.run(["zip", "-q", "-j", str(out / a.zip), str(res)], check=True)
    print("ready:", out / a.zip)


if __name__ == "__main__":
    main()
