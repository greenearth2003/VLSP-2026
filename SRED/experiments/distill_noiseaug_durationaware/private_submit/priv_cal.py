"""E8s5000_wise0.7 + qmf2_norm_speech (no AS-norm) back-end, fitted on the training-speaker cohort only
(posttrain_eval.run with norm=True, speech=True), applied to Public (check) and Private Test.

    cd SRED && .venv/bin/python priv_cal.py NSHARDS
"""
import sys
from pathlib import Path

import numpy as np

EXP = Path("experiments/distill_noiseaug_durationaware").resolve()
sys.path[:0] = [str(EXP.parents[1]), str(EXP), str(Path(__file__).parent)]
import public_fast as pf  # noqa: E402
import priv_cos as pc  # noqa: E402
from posttrain_eval import QMF2Files, build  # noqa: E402

NAME, TAG = "E8s5000_wise0.7", "E8s5000_wise0.7_A"
C = dict(np.load(pf.R / "_calib" / TAG / "posttrain.npz"))
COH = {k: v for k, v in C.items() if k.startswith("coh_") or k.startswith("q_coh")}
names = ["speech", "snr", "norm"]
qd = lambda q: {"speech": q[:, 0], "snr": q[:, 1], "norm": q[:, 2]}  # noqa: E731


def fit_score(w, sp, q, ia, ib):
    P = dict(COH, silver_whole=w, silver_speech=sp, q_silver_whole=q[:, :2], q_silver_speech=q[:, 2:4])
    ce, cs, cf, cq, se, sq = build(P, False, True)
    m = QMF2Files(names)
    m.fit_files(ce, cs, cf, qd(cq), None)
    return m.score(se, qd(sq), ia, ib, None)


# 1) check on Public Test: must equal public_fast backend "no AS-norm" (4.449)
D, tr, ia, ib = pf.load_all()
s = fit_score(D[f"w_{NAME}"], D[f"s_{NAME}"], D["q"], ia, ib)
print(f"public check: EER all {pf.ev.eer(s, tr.label.to_numpy()):.3f}", flush=True)

# 2) Private Test
n = int(sys.argv[1])
lines, fs = pc.trials()
parts = [dict(np.load(pc.OUT / "E8w" / f"shard{i}.npz")) for i in range(n)]
G = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
o = np.argsort(G["files"])
assert list(G["files"][o]) == fs
pos = {f: i for i, f in enumerate(fs)}
ja, jb = np.array([pos[a] for a, _ in lines]), np.array([pos[b] for _, b in lines])
s = fit_score(G[f"w_{NAME}"][o], G[f"s_{NAME}"][o], G["q"][o], ja, jb)
assert np.isfinite(s).all()
out = pc.OUT / f"results_cal_{NAME}.txt"
out.write_text("".join(f"{a} {b} {v:.6f}\n" for (a, b), v in zip(lines, s)))
print(f"wrote {out} ({len(lines):,} trials)")
