"""Provenance script (runs INSIDE the SRED training repo, not needed for inference): exports the two inference assets.
  1. s5q3/assets/s5_model.pt : self-contained S5 checkpoint = compressed-B6 structure (b6_cfg, low-rank ranks,
     dropped blocks) from exp/b6c/t1/init.pt + trained weights from exp/ft/S5_ens/step2500.pt.
  2. s5q3/assets/q3_qmf.json : the Q3 calibrator for S5 (QMF with SNR features, no AS-norm) fitted exactly as in
     scripts/qeval.py (Data(S5).fit("Q3")) on the TRAIN cohort only: standardisation mean/std + logistic-regression weights.
  python tools/export_assets.py --repo <SRED repo>     (uses <repo>/.venv)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
CKPT = "exp/ft/S5_ens/step2500.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/jovyan/code/VLSP-2026/SRED")
    root = Path(ap.parse_args().repo).resolve()
    sys.path[:0] = [str(root), str(root / "scripts")]
    os.chdir(root)
    init = torch.load(root / "exp/b6c/t1/init.pt", map_location="cpu", weights_only=False)
    ft = torch.load(root / CKPT, map_location="cpu", weights_only=False)
    out = {"b6_cfg": init["b6_cfg"], "ranks": init["ranks"], "drop": init["drop"], "model": ft["model"],
           "source": {"structure": "exp/b6c/t1/init.pt", "weights": CKPT, "step": ft["step"],
                      "lineage": "b6c_t1 -> C2 -> C2full -> S2 -> S5 (ensemble distillation from T1 + S2)"}}
    (PKG / "s5q3/assets").mkdir(parents=True, exist_ok=True)
    torch.save(out, PKG / "s5q3/assets/s5_model.pt")
    import qeval
    D = qeval.Data(CKPT)
    qmf, use_snr = D.fit("Q3")
    assert use_snr and not qmf.use_asnorm
    q = {"name": "Q3", "model": "S5", "features": ["s", "log_dmin", "log_dmax", "s*log_dmin", "s*log_dmax",
                                                   "snr_lo", "snr_hi", "s*snr_lo", "s*snr_hi"],
         "snr_transform": "snr_lo = clip(min(snrA, snrB), -5, 40) / 10 ; snr_hi = clip(max(snrA, snrB), -5, 40) / 10",
         "mean": qmf.mu.tolist(), "std": qmf.sd.tolist(),
         "coef": qmf.lr.coef_[0].tolist(), "intercept": float(qmf.lr.intercept_[0]),
         "score": "sum(coef * (x - mean) / std) + intercept  (logistic-regression log-odds)"}
    (PKG / "s5q3/assets/q3_qmf.json").write_text(json.dumps(q, indent=2))
    print("exported", q["coef"], q["intercept"])


if __name__ == "__main__":
    main()
