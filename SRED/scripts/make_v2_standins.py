"""v2 stand-ins, loop-free (2026-10-20):
  public_realnoise_v2 = public + real background from train recordings (lists/realnoise_bank.npz), each clip's noise
                        built from several DIFFERENT segments with 50 ms crossfades (sred.augment.fill_from_bank);
                        SNR = min(a private SNR draw, the file's own SNR) - 3 dB, as v1
  public_rumble_v2    = public_realnoise_v2 + rumble v2 (sred.augment.rumble_v2: steady drone < ~150 Hz + harmonic hum,
                        random level drift, generated over the whole clip) on 30% of files, full-band SNR -3..8 dB
Same files and labels as public; fixed seeds.

    python scripts/make_v2_standins.py --root /mnt/tsharp/thucth/dataset/vlsp2026-sred
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.augment import fill_from_bank, rumble_v2  # noqa: E402
from sred.data import PUBLIC_DIR, read_trials, unique_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
BANK = dict(np.load(ROOT / "lists/realnoise_bank.npz"))


def mix(x, n, snr):
    fr = x[: len(x) // 400 * 400].reshape(-1, 400)
    p = np.mean(fr ** 2, 1)
    ps = np.mean(p[p >= np.quantile(p, 0.5)]) if len(p) else np.mean(x ** 2)
    return x + n * np.sqrt(ps / (10 ** (snr / 10)) / (np.mean(n ** 2) + 1e-12))


def job(args):
    i, rel, snr, out = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(60_000_000 + i)
    x, _ = sf.read(str(PUBLIC_DIR / rel), dtype="float32")
    y = mix(x, fill_from_bank(BANK["audio"], BANK["offsets"], len(x), rng), snr)
    rumble = rng.random() < 0.3
    z = y
    if rumble:
        torch.manual_seed(70_000_000 + i)
        r = rumble_v2(1, len(x), "cpu")[0].numpy()
        z = y + r * np.sqrt(np.mean(y ** 2) / (10 ** (rng.uniform(-3, 8) / 10)) / (np.mean(r ** 2) + 1e-12))
    for name, a in (("public_realnoise_v2", y), ("public_rumble_v2", z)):
        o = Path(out) / name / rel
        o.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(o), (a / max(1.0, np.abs(a).max() / 0.99)).astype(np.float32), SR, subtype="PCM_16")
    return rel, rumble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    files = unique_files(read_trials()[0])
    fp = pd.read_csv(ROOT / "exp/analysis/features_public.csv").set_index("rel").loc[files]
    fv = pd.read_csv(ROOT / "exp/analysis/features_private.csv")
    draws = np.random.default_rng(11).choice(fv.snr_db.values, len(files))
    jobs = [(i, rel, float(min(draws[i], own)) - 3.0, a.root) for i, (rel, own) in enumerate(zip(files, fp.snr_db.values))]
    with Pool(24) as pool:
        res = pool.map(job, jobs, chunksize=32)
    pd.DataFrame(res, columns=["rel", "rumble"]).to_csv(Path(a.root) / "public_rumble_v2_files.csv", index=False)
    print("wrote", len(res), "files to public_realnoise_v2 and public_rumble_v2;", sum(r for _, r in res), "with rumble")


if __name__ == "__main__":
    main()
