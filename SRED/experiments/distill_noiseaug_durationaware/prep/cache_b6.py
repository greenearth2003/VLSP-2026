"""B6 teacher cache (NEW_PLAN §11.1; spec: HANDOFF.md "Teacher caches"): the raw approved ReDimNet2-B6 embedding of
every CLEAN crop of the crop bank, row i = crop i.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/cache_b6.py               # all 859,684 crops
    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/cache_b6.py --limit 1000  # test: first 1,000 crops

Writes artifacts/teacher_cache/b6.npy (float16 [859684, 192], rows L2-normalised in float32) + b6.json. Crops of one bin
share a length, so batches hold one bin: no padding, no truncation. Checks before writing the .json: no all-zero row,
row norms within 1 +- 0.01, and cos(cache, fresh strict-fp32 embedding) > 0.999 on the first 1,000 crops.
Precision: fp32 with TF32 matmul/conv by default. bf16 autocast (the spec's first choice) fails the 0.999 check on
0.18 % of crops, all in the 0.5-1 s bins (min cos 0.9966 on 6,000 random crops); TF32 gives min 0.9997, ~40 % slower.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

from dnd.data import load_manifest  # noqa: E402
from prep.verify_artifacts import digest  # noqa: E402
from sred.data import TRAIN_DIR  # noqa: E402
from sred.models import _load_redimnet2_pt  # noqa: E402

MODEL = "pretrained/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt"
BANK = EXP / "artifacts/crop_bank.npz"
SR = 16000


class Crops(torch.utils.data.Dataset):
    def __init__(self, bank, manifest):
        self.row, self.start, self.length = bank["row"], bank["start"], bank["length"]
        self.path = [str(TRAIN_DIR / m["audio_path"]) for m in manifest]

    def __len__(self):
        return len(self.row)

    def __getitem__(self, i):
        wav, _ = sf.read(self.path[self.row[i]], start=int(self.start[i]), frames=int(self.length[i]),
                         dtype="float32", always_2d=False)
        assert len(wav) == self.length[i], (i, len(wav))
        return i, wav


def collate(items):
    return torch.tensor([i for i, _ in items]), torch.from_numpy(np.stack([w for _, w in items]))


def batches(length, ids, max_audio_s):
    """Lists of crop ids, one bin (= one length) per batch, about max_audio_s seconds of audio each."""
    out = []
    for L in np.unique(length[ids]):
        sel = ids[length[ids] == L]
        bs = max(1, int(max_audio_s * SR // L))
        out += [sel[k:k + bs].tolist() for k in range(0, len(sel), bs)]
    return out


@torch.inference_mode()
def run(model, ds, ids, max_audio_s, workers, precision, log_every=200):
    bf16 = precision == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = precision != "fp32"
    bs = batches(ds.length, ids, max_audio_s)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=bs, num_workers=workers, collate_fn=collate,
                                     pin_memory=True, prefetch_factor=4 if workers else None)
    out = np.zeros((len(ds), 192), np.float16)
    t0, done = time.time(), 0
    for k, (idx, x) in enumerate(dl):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            e = model(x.cuda(non_blocking=True))
        out[idx.numpy()] = F.normalize(e.float(), dim=1).cpu().numpy().astype(np.float16)
        done += len(idx)
        if (k + 1) % log_every == 0 or k + 1 == len(bs):
            el = time.time() - t0
            print(f"{done}/{len(ids)} crops, {el / 60:.1f} min, ETA {(len(ids) - done) * el / done / 60:.1f} min",
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N crops (test); writes b6_testN.npy")
    ap.add_argument("--max-audio-s", type=float, default=512, help="audio seconds per batch")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--precision", choices=["tf32", "bf16"], default="tf32")
    args = ap.parse_args()

    bank = dict(np.load(BANK))
    manifest = load_manifest(EXP / "artifacts/manifest.jsonl")
    ds = Crops(bank, manifest)
    n = len(ds)
    ids = np.arange(args.limit or n)
    model = _load_redimnet2_pt(EXP / MODEL).cuda().eval()

    t0 = time.time()
    T = run(model, ds, ids, args.max_audio_s, args.workers, args.precision)
    seconds = round(time.time() - t0)

    # checks (HANDOFF.md "Teacher caches")
    Tc = T[ids].astype(np.float32)
    norm = np.linalg.norm(Tc, axis=1)
    assert (np.abs(Tc).sum(1) > 0).all(), "all-zero rows"
    assert np.abs(norm - 1).max() <= 0.01, f"row norm off by {np.abs(norm - 1).max():.4f}"
    ref_ids = ids[:1000]
    ref = run(model, ds, ref_ids, args.max_audio_s, args.workers, "fp32")[ref_ids].astype(np.float32)
    cos = (Tc[:len(ref_ids)] * ref).sum(1) / np.linalg.norm(ref, axis=1)
    print(f"check ({args.precision}): {len(ids)} rows, norm in [{norm.min():.4f}, {norm.max():.4f}], "
          f"cos vs fp32 on first {len(ref_ids)}: min {cos.min():.5f} mean {cos.mean():.5f}")
    assert cos.min() > 0.999, f"cos vs fp32 {cos.min():.5f} <= 0.999"

    out_dir = EXP / "artifacts/teacher_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"b6_test{args.limit}" if args.limit else "b6"
    npy = out_dir / f"{name}.npy"
    np.save(npy, np.ascontiguousarray(T[ids] if args.limit else T))
    sha = hashlib.sha256(npy.read_bytes()).hexdigest()
    meta = {"teacher": "b6", "model_file": MODEL, "model_hash16": digest(EXP / MODEL),
            "crop_bank_hash16": digest(BANK), "rows": len(ids), "dim": 192, "dtype": "float16",
            "l2_normalised": True,
            "autocast": "bf16" if args.precision == "bf16" else "none (fp32, TF32 matmul/conv)", "host": socket.gethostname(), "seconds": seconds,
            "sha256": sha}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
