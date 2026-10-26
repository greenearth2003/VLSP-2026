"""Provenance check (runs with the SRED training repo available): this package reproduces the competition pipeline.
  python tools/verify_against_repo.py --repo <SRED repo>
Checks: (1) package network == repo network (same outputs); (2) package embeddings / duration / SNR of 300 random
private files == cached repo values; (3) package Q3 scoring on the cached private embeddings reproduces the submitted
file exp/private_cand/S5_ens_Q3/results.txt line by line; (4) writes tests/expected_private_sample.npz for the smoke test.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PKG = Path(__file__).resolve().parents[1]
CK = "exp/ft/S5_ens/step2500.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/jovyan/code/VLSP-2026/SRED")
    ap.add_argument("--private", default="/home/jovyan/data/VLSP2026/Vi-SRED26/private_test")
    a = ap.parse_args()
    root, priv = Path(a.repo).resolve(), Path(a.private)
    sys.path[:0] = [str(PKG), str(root)]
    os.chdir(root)
    import s5q3
    from sred.data import read_trials, unique_files
    from sred.embed import cache_key
    from sred.models import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = {}
    m_pkg, m_repo = s5q3.load_s5(dev), load_model(CK).to(dev).eval()
    r["params"] = s5q3.count_params(m_pkg)
    x = torch.randn(2, 48000, device=dev) * 0.1
    with torch.inference_mode():
        r["net_max_abs_diff"] = float((m_pkg(x) - m_repo(x)).abs().max())
    del m_repo
    pairs, _ = read_trials(priv / "private-test.txt")
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    Ec = np.load(cache_key(CK, "private", files, mode="full"))
    Ec = Ec / np.linalg.norm(Ec, axis=-1, keepdims=True).clip(1e-8)
    st = pd.read_csv(root / "eda/out/audio_stats_private.csv").set_index("rel").loc[files]
    pick = np.random.default_rng(0).choice(len(files), 300, replace=False)
    cos, dd, ds, Es = [], [], [], []
    for i in pick:
        w = s5q3.load_wav(priv / "data" / files[i])
        e = s5q3.embed(m_pkg, w, dev)
        Es.append(e)
        cos.append(float(e @ Ec[i]))
        dd.append(abs(s5q3.duration_s(w) - st.dur.values[i]))
        ds.append(abs(s5q3.snr_est(w) - st.snr_est.values[i]))
    r["emb_cos_vs_cache_min"] = float(np.min(cos))
    r["dur_max_abs_diff"], r["snr_max_abs_diff"] = float(np.max(dd)), float(np.max(ds))
    ia = np.array([pos[x] for x, _ in pairs])
    ib = np.array([pos[y] for _, y in pairs])
    s = s5q3.Q3().score(np.sum(Ec[ia] * Ec[ib], 1), st.dur.values[ia], st.dur.values[ib],
                        st.snr_est.values[ia], st.snr_est.values[ib])
    ref = (root / "exp/private_cand/S5_ens_Q3/results.txt").read_text().splitlines()
    new = [f"{x} {y} {v:.6f}" for (x, y), v in zip(pairs, s)]
    r["private_lines"] = len(ref)
    r["private_lines_identical"] = int(sum(p == q for p, q in zip(ref, new)))
    refv = np.array([float(x.split()[2]) for x in ref])
    r["private_score_max_abs_diff"] = float(np.max(np.abs(refv - s)))
    np.savez(PKG / "tests/expected_private_sample.npz", files=np.array([files[i] for i in pick[:40]]),
             emb=np.stack(Es[:40]), dur=st.dur.values[pick[:40]], snr=st.snr_est.values[pick[:40]])
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
