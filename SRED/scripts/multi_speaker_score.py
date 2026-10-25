"""Multi-speaker scoring test: each clip -> its speakers (whole-clip embedding, plus 2 cluster centroids when the
clip is confidently two-voice); a trial scores by the best speaker pair ("matched if at least one pair matches").

    python scripts/multi_speaker_score.py --model exp/ft/W_E2b_sf2/avg.pt --splits public_noisy public_noisy_multi

Per split, cosine EER / minDCF on the public trial list for:
  full      whole-clip embedding (baseline)
  maxpair   max cosine over all speaker pairs (the proposal)
  blend     0.5 * full + 0.5 * maxpair
  gated     maxpair only when at least one clip is two-voice, else full
Per-file speaker sets are cached in exp/emb/<model>/<split>_speakers.npz.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import read_trials, unique_files  # noqa: E402
from sred.embed import cache_key, vad_trim  # noqa: E402
from sred.metrics import compute_eer, compute_min_dcf  # noqa: E402
from sred.models import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


@torch.inference_mode()
def speakers(model, wav, win_sec=1.0, hop_sec=0.5, split_cos=0.45, min_minor=0.2):
    """Returns (full_embedding, [centroid0, centroid1] or [])."""
    full = F.normalize(model(wav[None]), dim=-1)[0]
    w = vad_trim(wav)
    win, hop = int(win_sec * SR), int(hop_sec * SR)
    if w.shape[-1] < 3 * SR:
        return full, []
    segs = w.unfold(0, win, hop)
    n = segs.shape[0]
    if n < 4:
        return full, []
    e = F.normalize(model(segs), dim=-1)
    i, j = divmod(int(torch.argmin(e @ e.T)), n)
    cent = torch.stack([e[i], e[j]])
    for _ in range(10):
        a = (e @ cent.T).argmax(1)
        if a.min() == a.max():
            return full, []
        cent = F.normalize(torch.stack([e[a == k].mean(0) for k in (0, 1)]), dim=-1)
    a = (e @ cent.T).argmax(1)
    n0 = int((a == 0).sum())
    if min(n0, n - n0) < max(2, int(round(min_minor * n))) or float(cent[0] @ cent[1]) >= split_cos:
        return full, []
    return full, [cent[0], cent[1]]


def extract(model_name, split, files):
    path = cache_key(model_name, split, files, mode="full").with_name(f"{split}_speakers.npz")
    if path.exists():
        z = np.load(path)
        return z["full"], z["c0"], z["c1"], z["two"]
    model = load_model(model_name).cuda().eval()
    root = ROOT / "data" / split
    full, c0, c1, two = [], [], [], []
    for k, f in enumerate(files):
        x, _ = sf.read(str(root / f), dtype="float32")
        if len(x) < SR // 2:
            x = np.tile(x, int(np.ceil(SR / 2 / max(1, len(x)))))[: SR // 2]
        e, cs = speakers(model, torch.from_numpy(x).cuda())
        full.append(e.cpu().numpy())
        two.append(bool(cs))
        c0.append((cs[0] if cs else e).cpu().numpy())
        c1.append((cs[1] if cs else e).cpu().numpy())
        if k % 2000 == 0:
            print(f"  {split}: {k}/{len(files)}", flush=True)
    out = dict(full=np.stack(full), c0=np.stack(c0), c1=np.stack(c1), two=np.array(two))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out["full"], out["c0"], out["c1"], out["two"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="exp/ft/W_E2b_sf2/avg.pt")
    ap.add_argument("--splits", nargs="+", default=["public_noisy", "public_noisy_multi"])
    a = ap.parse_args()
    pairs, y = read_trials()
    files = unique_files(pairs)
    pos = {f: i for i, f in enumerate(files)}
    ia = np.array([pos[p] for p, _ in pairs])
    ib = np.array([pos[q] for _, q in pairs])
    for split in a.splits:
        full, c0, c1, two = extract(a.model, split, files)
        print(f"\n{split}: two-voice clips {two.mean():.1%}; trials touching one {np.mean(two[ia] | two[ib]):.1%}")
        s_full = (full[ia] * full[ib]).sum(1)
        cands = []
        for A in (full, c0, c1):
            for B in (full, c0, c1):
                cands.append((A[ia] * B[ib]).sum(1))
        s_max = np.max(cands, 0)
        s_blend = 0.5 * s_full + 0.5 * s_max
        s_gated = np.where(two[ia] | two[ib], s_max, s_full)
        print(f"  {'backend':8s} {'EER':>6s} {'minDCF':>7s}")
        for name, s in (("full", s_full), ("maxpair", s_max), ("blend", s_blend), ("gated", s_gated)):
            print(f"  {name:8s} {compute_eer(s, y):6.3f} {compute_min_dcf(s, y):7.3f}")
        sub = two[ia] | two[ib]
        if sub.sum() > 500:
            print(f"  on the {sub.sum()} trials touching a two-voice clip: full {compute_eer(s_full[sub], y[sub]):.3f} "
                  f"maxpair {compute_eer(s_max[sub], y[sub]):.3f} blend {compute_eer(s_blend[sub], y[sub]):.3f}")


if __name__ == "__main__":
    main()
