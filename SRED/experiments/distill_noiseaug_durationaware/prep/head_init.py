"""Speaker centroids of the pretrained B4 on clean train audio, used to initialise every new SF2 head (E0-E3), so the
backbone does not receive random-head gradients at the start. Cached in artifacts/b4_centroids.npy (keyed to the
manifest hash); ~10 min once.

Per utterance: the most speech-dense window of min(utterance, 6 s), on a 0.5 s grid so batches share a length;
embeddings L2-normalised, averaged per speaker, re-normalised.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/head_init.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
sys.path.insert(0, str(SRED))
sys.path.insert(0, str(EXP))

from dnd.vad import SR, dense_windows  # noqa: E402

OUT = EXP / "artifacts/b4_centroids.npy"
CAP, GRID, BS = 6.0, 0.5, 64


def centroids(model, manifest, spk2id, train_dir, device="cuda") -> np.ndarray:
    jobs = defaultdict(list)  # crop length (samples) -> [(row, start)]
    for m in manifest:
        sec = min(CAP, int(m["num_samples"] / SR / GRID) * GRID)
        if sec < GRID:
            sec = m["num_samples"] / SR
        L = int(sec * SR)
        jobs[L].append((m["row"], dense_windows(m["active_intervals"], m["num_samples"], L, 1)[0]))
    emb = np.zeros((len(manifest), model(torch.zeros(1, 32000, device=device)).shape[-1]), np.float32)
    done = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for L, items in sorted(jobs.items()):
            for i in range(0, len(items), BS):
                chunk = items[i:i + BS]
                wav = np.stack([sf.read(str(train_dir / manifest[r]["audio_path"]), start=s, frames=L,
                                        dtype="float32")[0] for r, s in chunk])
                e = F.normalize(model(torch.from_numpy(wav).to(device)).float(), dim=1)
                emb[[r for r, _ in chunk]] = e.cpu().numpy()
                done += len(chunk)
            print(f"  {done}/{len(manifest)}", end="\r", flush=True)
    cent = np.zeros((len(spk2id), emb.shape[1]), np.float32)
    for m in manifest:
        cent[spk2id[m["speaker_id"]]] += emb[m["row"]]
    return cent / np.linalg.norm(cent, axis=1, keepdims=True)


def load_or_build(model, manifest, spk2id, train_dir, man_hash: str, out: Path = OUT) -> np.ndarray:
    """Centroids of `model` cached at `out` (one file per pretrained model, e.g. b4_ / b6_centroids.npy)."""
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists() and json.loads(meta.read_text()).get("manifest_sha256") == man_hash:
        return np.load(out)
    was_training = model.training
    model.eval()
    c = centroids(model, manifest, spk2id, train_dir)
    model.train(was_training)
    np.save(out, c)
    meta.write_text(json.dumps({"manifest_sha256": man_hash, "speakers": len(spk2id), "cap_sec": CAP}))
    return c


if __name__ == "__main__":
    from dnd.data import load_manifest
    from sred.data import TRAIN_DIR
    from sred.models import _load_redimnet2_pt

    man = load_manifest(EXP / "artifacts/manifest.jsonl")
    spk2id = {s: i for i, s in enumerate(sorted({m["speaker_id"] for m in man}))}
    h = (EXP / "artifacts/manifest.sha256").read_text().split()[0]
    model = _load_redimnet2_pt(EXP / "pretrained/redimnet2/b4-vox2-lm.pt").cuda()
    c = load_or_build(model, man, spk2id, TRAIN_DIR, h)
    print("\ncentroids", c.shape, "->", OUT)
