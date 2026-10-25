"""All C2 embedding variants for one split, in one pass (teammate's C2, loaded weights-only):
  whole       whole-clip embedding (first 60 s; clips < 0.5 s repeated), L2-normalised  (= their infer.py)
  whole_norm  its norm before normalisation (a quality measure)
  tta         mean of normalised view embeddings: < 2 s repeated to >= 4 s; 2-4 s itself; > 4 s itself + 4-s windows
  vad         energy-trimmed clip (sred.embed.vad_trim, no VAD model), L2-normalised
  dur, snr    their c2q3 duration and SNR estimate (the Q3 inputs)

    python scripts/c2_embed_all.py --split traincohort5 [traincohort_noisy traincohort_realnoise silver public_realnoise public_noisy private]
        [--model c2 | exp/ft/<run>/avg.pt] [--tag ext_C2] [--no-tta]

Cache: exp/emb/<tag>/<split>_all.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/mnt/tsharp/thucth/tmp/claude-1004/-home-thuc-thuc/fa0eee2f-9671-4c94-8b41-6d41fb8da6d5/scratchpad")
from c2_loader import load_c2_safe  # noqa: E402
import c2q3  # noqa: E402
from sred.data import TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import vad_trim  # noqa: E402

SR = 16000


def split_files(split):
    if split.startswith("traincohort"):
        files = [r for _, r in cohort_subset(read_train_list(), per_spk=5)]
        return files, (TRAIN_DIR if split == "traincohort5" else ROOT / "data" / split)
    if split == "private":
        return unique_files(read_trials(ROOT / "submissions/final_private/trials_2col.txt")[0]), ROOT / "data/Vi-SRED26/private_test/data"
    files = unique_files(read_trials()[0])
    return files, ROOT / "data" / ("public_rumble_v2" if split == "silver" else split)


def views(x):
    L, W = len(x), 4 * SR
    if L < 2 * SR:
        return [np.tile(x, int(np.ceil(W / max(1, L))) + (1 if L and np.ceil(W / L) * L <= W else 0))], None
    if L <= W:
        return [], None  # the whole clip is the only view
    wins = []
    for s in range(0, L, W):
        seg = x[s:s + W]
        wins.append(seg if len(seg) >= 2 * SR else x[L - W:])
    return [], np.stack([w for w in wins if len(w) == W])


class Clips(torch.utils.data.Dataset):
    def __init__(self, files, root):
        self.files, self.root = files, root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = c2q3.load_wav(self.root / self.files[i])
        whole = x[: 60 * SR] if len(x) >= SR // 2 else np.tile(x, int(np.ceil(SR / 2 / max(1, len(x)))))[: SR // 2]
        v = vad_trim(torch.from_numpy(x)).numpy()
        v = v[: 60 * SR] if len(v) >= SR // 2 else np.tile(v, int(np.ceil(SR / 2 / max(1, len(v)))))[: SR // 2]
        singles, wins = views(x)
        return i, np.ascontiguousarray(whole), np.ascontiguousarray(v), singles, wins, c2q3.duration_s(x), c2q3.snr_est(x)


@torch.inference_mode()
def run(split, model, tag="ext_C2", do_tta=True):
    out = ROOT / f"exp/emb/{tag}/{split}_all.npz"
    if out.exists():
        print("cached", out)
        return
    files, root = split_files(split)
    n = len(files)
    whole, tta, vad = (np.zeros((n, 192), np.float32) for _ in range(3))
    norm, dur, snr = np.zeros(n, np.float32), np.zeros(n), np.zeros(n)
    fwd = lambda a: model(torch.from_numpy(a).cuda()).float()  # noqa: E731
    dl = torch.utils.data.DataLoader(Clips(files, root), batch_size=None, num_workers=8, prefetch_factor=4, collate_fn=lambda b: b)
    for k, (i, xw, xv, singles, wins, d, q) in enumerate(dl):
        raw = fwd(xw[None])[0]
        e = F.normalize(raw, dim=-1).cpu().numpy()
        whole[i], norm[i] = e, float(raw.norm())
        vad[i] = F.normalize(fwd(xv[None]), dim=-1)[0].cpu().numpy()
        vs = [e] if (not singles or not do_tta) else [F.normalize(fwd(singles[0][: 60 * SR][None]), dim=-1)[0].cpu().numpy()]
        if wins is not None and do_tta:
            vs += list(F.normalize(fwd(wins), dim=-1).cpu().numpy())
        tta[i], dur[i], snr[i] = np.mean(vs, 0), d, q
        if k % 5000 == 0:
            print(f"  {split}: {k}/{n}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, files=np.array(files), whole=whole, whole_norm=norm, tta=tta, vad=vad, dur=dur, snr=snr)
    print(f"{split}: {n} files -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", required=True)
    ap.add_argument("--model", default="c2", help="'c2' or a fine-tuned checkpoint whose base_model is c2")
    ap.add_argument("--tag", default="ext_C2")
    ap.add_argument("--no-tta", action="store_true", help="skip TTA views (tta = whole)")
    a = ap.parse_args()
    from sred.models import load_model
    model = load_model(a.model).cuda().eval()
    for s in a.split:
        run(s, model, a.tag, not a.no_tta)


if __name__ == "__main__":
    main()
