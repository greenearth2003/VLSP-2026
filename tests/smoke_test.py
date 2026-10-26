"""Standalone smoke test of this package (no training repo needed).
  python tests/smoke_test.py --audio-dir <private_test/data> [--device cuda]
Checks: asset md5s (MANIFEST.md5), parameter count (6,917,480 < 7M), and that 40 private-test files give the same
embedding (cosine >= 0.9999), duration and SNR as the competition pipeline (tests/expected_private_sample.npz)."""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import s5q3  # noqa: E402

ASSETS = ["s5q3/assets/s5_model.pt", "s5q3/assets/q3_qmf.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True, help="folder containing the private-test wav files (idXXXXX/NNNN.wav)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    man = {ln.split()[1]: ln.split()[0] for ln in (PKG / "MANIFEST.md5").read_text().splitlines() if ln.strip()}
    for f in ASSETS:
        h = hashlib.md5((PKG / f).read_bytes()).hexdigest()
        assert h == man[f], f"{f}: md5 {h} != {man[f]}"
    m = s5q3.load_s5(a.device)
    n = s5q3.count_params(m)
    assert n == 6_917_480 and n < 7_000_000, n
    z = np.load(PKG / "tests/expected_private_sample.npz")
    cos = []
    for f, e, d, q in zip(z["files"], z["emb"], z["dur"], z["snr"]):
        w = s5q3.load_wav(Path(a.audio_dir) / str(f))
        cos.append(float(s5q3.embed(m, w, a.device) @ e))
        assert s5q3.duration_s(w) == d and s5q3.snr_est(w) == q, (f, s5q3.duration_s(w), d, s5q3.snr_est(w), q)
    assert min(cos) >= 0.9999, min(cos)
    print(f"OK: params {n:,}; 40 files: min embedding cosine {min(cos):.7f}, durations and SNRs identical")


if __name__ == "__main__":
    main()
