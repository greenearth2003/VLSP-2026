"""C2-family model + the teammate's frozen Q3 on raw score variants (whole / whole+trimmed / gated / tta+trimmed).
    python scripts/c2_q3_eval.py --tag W_C2X --splits silver [public_realnoise public_noisy]
EER all / cleaned list (+ private-like weighting on silver)."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), "/mnt/tsharp/thucth/dataset/vlsp2026-sred/incoming/c2pkg"]
import c2_eval as C, c2q3  # noqa: E402
from sred.data import read_trials, unique_files  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--splits", nargs="+", default=["silver"])
a = ap.parse_args()
C.E = ROOT / "exp/emb" / a.tag
pairs, y = read_trials(); files = unique_files(pairs); pos = {f: i for i, f in enumerate(files)}
ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
cl = pd.DataFrame(pairs, columns=["file_a", "file_b"]).merge(pd.read_csv(C.META), how="left").in_cleaned.fillna(False).to_numpy(bool)
w = C.private_like_weights(pairs, y); q3 = c2q3.Q3()
for split in a.splits:
    z = C.load(split); g = C.gate_of(split, files, ia, ib); raw = C.raw_scores(z, ia, ib, g)
    args = (z["dur"][ia], z["dur"][ib], z["snr"][ia], z["snr"][ib])
    for k in ("whole", "fv", "gated", "tta_fv"):
        s = q3.score(raw[k], *args)
        pl = f"  private-like {C.eer(s, y, w):.3f}" if split == "silver" else ""
        print(f"{a.tag:7s} {split:16s} {k:7s}+Q3  all {C.eer(s, y):.3f}  cleaned {C.eer(s[cl], y[cl]):.3f}{pl}", flush=True)
