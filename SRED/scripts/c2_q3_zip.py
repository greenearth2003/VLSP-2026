"""Private zip for a C2-family model with the frozen Q3: --tag <emb dir> --variant whole|fv|gated --zip NAME"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), "/mnt/tsharp/thucth/dataset/vlsp2026-sred/incoming/c2pkg"]
import c2_eval as C, c2q3  # noqa: E402
from sred.data import read_trials, unique_files  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--variant", default="fv"); ap.add_argument("--zip", required=True)
a = ap.parse_args(); C.E = ROOT / "exp/emb" / a.tag
pairs, _ = read_trials(ROOT / "submissions/final_private/trials_2col.txt"); files = unique_files(pairs); pos = {f: i for i, f in enumerate(files)}
ia = np.array([pos[x] for x, _ in pairs]); ib = np.array([pos[x] for _, x in pairs])
z = C.load("private"); assert list(z["files"]) == files
raw = C.raw_scores(z, ia, ib, C.gate_of("private", files, ia, ib))
s = c2q3.Q3().score(raw[a.variant], z["dur"][ia], z["dur"][ib], z["snr"][ia], z["snr"][ib])
off = [l.split()[:2] for l in open("/mnt/tsharp/thucth/dataset/vlsp2026-sred/Vi-SRED26/private_test/private-test.txt") if l.strip()]
assert [list(p) for p in pairs] == off and np.isfinite(s).all()
out = ROOT / "submissions/final_private"; res = out / f"results_{Path(a.zip).stem}.txt"
res.write_text("".join(f"{x} {w} {v:.6f}\n" for (x, w), v in zip(pairs, s)))
tmp = out / "results.txt"; tmp.write_text(res.read_text()); (out / a.zip).unlink(missing_ok=True)
subprocess.run(["zip", "-q", "-j", str(out / a.zip), str(tmp)], check=True)
print(f"{a.zip}: {len(pairs):,} lines in official order, all finite")
