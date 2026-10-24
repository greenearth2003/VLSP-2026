"""External augmentation banks from the BTC-allowed corpora MUSAN (OpenSLR 17) and RIRS_NOISES (OpenSLR 28), saved once:

  lists/rir_bank.npz    rir [N, 1 s] float16 (unit norm, cut 2.5 ms before the direct-path peak) + cat [N]
                        cats: sim_small / sim_medium / sim_large (simulated, 1500 each) / real (all 325 real RIRs, channel 0)
  lists/musan_bank.npz  audio (concatenated float16) + offsets + cat per segment, segments capped at 20 s
                        cats: musan_noise / musan_music / musan_speech / rirs_pointsource / rirs_isotropic (channel 0)

    python scripts/make_ext_banks.py [--root /mnt/tsharp/thucth/dataset/vlsp2026-sred/external/noise]
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SR, RIR_LEN, SEG_CAP, N_SIM = 16000, 16000, 20 * 16000, 1500


def _ch0(path, n=None):
    x, sr = sf.read(path, frames=n if n else -1, dtype="float32", always_2d=True)
    assert sr == SR, (path, sr)
    return x[:, 0]


def _rir(path):
    h = _ch0(path)
    st = max(0, int(np.argmax(np.abs(h))) - int(0.0025 * SR))
    h = h[st:st + RIR_LEN]
    h = np.pad(h, (0, RIR_LEN - len(h)))
    return h / (np.linalg.norm(h) + 1e-8)


def _seg(path):
    x = _ch0(path, SEG_CAP)
    return x if len(x) > SR // 2 and np.abs(x).max() > 1e-4 else None


def _list_paths(lst, base):
    # Kaldi lists: "... RIRS_NOISES/<sub>/<file>.wav" (last token, relative to the corpus parent)
    return [str(base / ln.split()[-1]) for ln in open(lst) if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/tsharp/thucth/dataset/vlsp2026-sred/external/noise")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    root, rng = Path(a.root), np.random.default_rng(0)
    rirs_dir = root / "RIRS_NOISES"

    rir_jobs = []
    for size in ("small", "medium", "large"):
        p = sorted((rirs_dir / f"simulated_rirs/{size}room").rglob("*.wav"))
        rir_jobs += [(str(x), f"sim_{size}") for x in rng.choice(p, N_SIM, replace=False)]
    rir_jobs += [(p, "real") for p in _list_paths(rirs_dir / "real_rirs_isotropic_noises/rir_list", root)]

    noise_jobs = [(str(p), f"musan_{c}") for c in ("noise", "music", "speech")
                  for p in sorted((root / f"musan/{c}").rglob("*.wav"))]
    noise_jobs += [(p, "rirs_pointsource") for p in _list_paths(rirs_dir / "pointsource_noises/noise_list", root)]
    noise_jobs += [(p, "rirs_isotropic") for p in _list_paths(rirs_dir / "real_rirs_isotropic_noises/noise_list", root)]

    with Pool(a.workers) as pool:
        H = pool.map(_rir, [p for p, _ in rir_jobs], chunksize=16)
        S = pool.map(_seg, [p for p, _ in noise_jobs], chunksize=4)
    np.savez(ROOT / "lists/rir_bank.npz", rir=np.stack(H).astype(np.float16), cat=np.array([c for _, c in rir_jobs]))
    keep = [(s, c) for s, (_, c) in zip(S, noise_jobs) if s is not None]
    off = np.cumsum([0] + [len(s) for s, _ in keep])
    np.savez(ROOT / "lists/musan_bank.npz", audio=np.concatenate([s for s, _ in keep]).astype(np.float16), offsets=off,
             cat=np.array([c for _, c in keep]))
    for name, cats in (("rir_bank", [c for _, c in rir_jobs]), ("musan_bank", [c for _, c in keep])):
        u, n = np.unique(cats, return_counts=True)
        print(name, dict(zip(u.tolist(), n.tolist())))
    print("musan_bank:", round(off[-1] / SR / 3600, 2), "h")


if __name__ == "__main__":
    main()
