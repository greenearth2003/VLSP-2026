"""Listen to what a run's training crops looked like: clean crop -> the run's exact augmentation (sred.augment),
at the start and at the end of a matched-noise curriculum. Records which augmentations hit each crop.

    python scripts/sample_train_aug.py --run M4_curr --n 6 --out exp/analysis/train_samples_M4
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.augment import Augmenter  # noqa: E402
from sred.data import TRAIN_DIR, read_train_list  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
# order in which Augmenter.__call__ draws its per-op masks (one _rand call each)
OPS = ["prosody", "matched", "overlap", "reverb", "noise", "band", "eq", "mulaw", "freqmask", "clip", "timemask", "gain"]


def build(a, p_matched):
    rng = lambda v: tuple(float(x) for x in v.split(","))  # noqa: E731
    return Augmenter(p_noise=a["aug_noise_p"] * a["aug"], snr_noise=rng(a["aug_noise_snr"]), snr_babble=rng(a["aug_babble_snr"]),
                     noise_beta=rng(a["aug_noise_beta"]), p_reverb=0.3 * a["aug"], p_band=0.2 * a["aug"], p_eq=0.3 * a["aug"],
                     p_mulaw=0.1 * a["aug"], p_overlap=a["aug_overlap"], p_matched=p_matched,
                     matched=dict(np.load(ROOT / "lists/matched_noise.npz")) if p_matched > 0 else None,
                     matched_snr_shift=a["aug_matched_shift"], matched_hiss_frac=a["aug_matched_hiss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="M4_curr")
    ap.add_argument("--n", type=int, default=6, help="crops shown per curriculum point")
    ap.add_argument("--out", default="exp/analysis/train_samples_M4")
    a = ap.parse_args()
    args = json.loads((ROOT / f"exp/ft/{a.run}/args.json").read_text())
    items = read_train_list(Path(args["train_list"]))
    random.seed(3), torch.manual_seed(3), np.random.seed(3)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    end = args["aug_matched_end"] if args["aug_matched_end"] is not None else args["aug_matched"]
    stages = [("start", args["aug_matched"]), ("end", end)]
    t = np.arange(int(0.12 * SR)) / SR
    beep = lambda f: (0.1 * np.sin(2 * np.pi * f * t) * np.hanning(len(t))).astype(np.float32)  # noqa: E731
    gap = np.zeros(int(0.3 * SR), np.float32)
    rows, parts = [], []
    for stage, p in stages:
        aug = build(args, p)
        draws = []
        orig = aug._rand
        aug._rand = lambda b, d, _o=orig: draws.append(_o(b, d)) or draws[-1]
        B = 32  # training batch: babble / overlap draw from the other crops of the batch
        batch = []
        for spk, rel in random.sample(items, B):
            x, _ = sf.read(str(TRAIN_DIR / rel), dtype="float32")
            st = random.randint(0, max(0, len(x) - 3 * SR))
            x = x[st:st + 3 * SR]
            batch.append(np.pad(x, (0, 3 * SR - len(x))))
        clean = torch.from_numpy(np.stack(batch))
        with torch.no_grad():
            y = aug(clean).numpy()
        probs = dict(prosody=0, matched=aug.p_matched, overlap=aug.p_overlap, reverb=aug.p_reverb, noise=aug.p_noise,
                     band=aug.p_band, eq=aug.p_eq, mulaw=aug.p_mulaw, freqmask=aug.p_freqmask, clip=aug.p_clip,
                     timemask=aug.p_timemask, gain=aug.p_gain)
        hit = {op: (draws[k] < probs[op]).numpy() for k, op in enumerate(OPS) if k < len(draws)}
        for i in range(a.n):
            ops = [op for op in OPS if op in hit and hit[op][i]] or ["none (clean)"]
            name = f"{stage}_{i + 1}"
            sf.write(out / f"{name}_clean.wav", clean[i].numpy(), SR, subtype="PCM_16")
            sf.write(out / f"{name}_aug.wav", y[i], SR, subtype="PCM_16")
            parts += [beep(660), gap, clean[i].numpy(), gap, beep(990), gap, y[i].astype(np.float32), gap, gap]
            rows.append(f"{name:8s} matched-noise prob {p:.1f} | applied: {', '.join(ops)}")
    sf.write(out / "listen_clean_then_aug.wav", np.concatenate(parts), SR, subtype="PCM_16")
    (out / "index.txt").write_text("low beep = clean crop, high beep = the same crop as the model saw it\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
