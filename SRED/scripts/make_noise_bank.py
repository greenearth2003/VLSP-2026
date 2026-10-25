"""Real background-noise bank for training (quiet frames of train recordings, scripts/make_realnoise_public.py's harvest),
saved once as lists/realnoise_bank.npz: concatenated float16 audio + offsets; segments capped at 10 s.

    python scripts/make_noise_bank.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_realnoise_public as mrp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ft = pd.read_csv(ROOT / "exp/analysis/features_train.csv")
cand = ft[(ft.dur > 8) & (ft.speech_ratio < 0.8)].sample(6000, random_state=2).rel.tolist()
with Pool(32) as pool:
    bank = [b[: 10 * 16000] for b in pool.map(mrp.harvest, cand, chunksize=32) if b is not None and len(b) > 16000]
off = np.cumsum([0] + [len(b) for b in bank])
np.savez(ROOT / "lists/realnoise_bank.npz", audio=np.concatenate(bank).astype(np.float16), offsets=off)
print("bank:", len(bank), "segments,", round(off[-1] / 16000 / 60, 1), "min")
