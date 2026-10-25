"""Test-time variants for SHORT clips only (duration < --max-dur s), for segment-gated scoring (scripts/segment_fix.py):
  tile      the clip repeated to --tile-sec seconds before embedding
  vadtile   energy-trimmed (sred.embed.vad_trim), then repeated to --tile-sec seconds
Longer clips keep their whole-clip embedding, so a fix built on these only changes trials with a short clip.

    python scripts/short_tta_embed.py --model exp/ft/S3_E2b_X2/avg.pt --splits public_rumble_v2 public_noisy public_realnoise private

Cache: exp/emb/<model>/<split>_short<max-dur>_{tile,vadtile}<tile-sec>.npy, rows in the split's file order, whole-clip
embedding for clips >= max-dur.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_embed import split_files  # noqa: E402
from sred.embed import cache_key, vad_trim  # noqa: E402
from sred.models import load_model  # noqa: E402
from sred.scoring import l2n  # noqa: E402

SR = 16000


def tile(x, n):
    return np.tile(x, int(np.ceil(n / max(1, len(x)))))[:n] if len(x) else np.zeros(n, np.float32)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--max-dur", type=float, default=2.0)
    ap.add_argument("--tile-sec", type=float, default=3.0)
    a = ap.parse_args()
    model = load_model(a.model).cuda().eval()
    N = int(a.tile_sec * SR)
    for split in a.splits:
        files, root, dur = split_files(split)
        fk = cache_key(a.model, split, files, mode="full")
        outs = {k: fk.with_name(f"{split}_short{a.max_dur:g}_{k}{a.tile_sec:g}.npy") for k in ("tile", "vadtile")}
        if all(p.exists() for p in outs.values()):
            print("cached", split)
            continue
        full = l2n(np.load(fk))
        res = {k: full.copy() for k in outs}
        sel = np.where(dur < a.max_dur)[0]
        for i in sel:
            x, _ = sf.read(str(root / files[i]), dtype="float32")
            v = vad_trim(torch.from_numpy(x)).numpy()
            segs = torch.from_numpy(np.stack([tile(x, N), tile(v, N)])).cuda()
            e = F.normalize(model(segs), dim=-1).cpu().numpy()
            res["tile"][i], res["vadtile"][i] = e[0], e[1]
        for k, p in outs.items():
            np.save(p, res[k])
        print(f"{split}: {len(sel)} of {len(files)} clips < {a.max_dur:g} s re-embedded (tile / vad+tile to {a.tile_sec:g} s)", flush=True)


if __name__ == "__main__":
    main()
