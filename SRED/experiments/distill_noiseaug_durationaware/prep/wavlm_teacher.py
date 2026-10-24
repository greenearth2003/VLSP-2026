"""WavLM-Large speaker teacher (NEW_PLAN §10.1): prepare the speaker head on frozen WavLM, then cache its embeddings
of every clean crop of the crop bank (HANDOFF.md "Teacher caches").

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/wavlm_teacher.py prep  [--steps 4000]
    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/wavlm_teacher.py cache
    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/wavlm_teacher.py eval   # silver EER of the teacher

Teacher = WavLM-Large (frozen, torchaudio state dict) -> its 24 transformer layer outputs -> softmax-weighted sum
(learned) -> attentive statistics pooling -> linear 192 (the teacher embedding) -> SphereFace2 head over the train
speakers (training only). `prep` trains the layer weights, pooling, projection and head on clean bank crops
(32 speakers x 2 crops per step, both views, the student's bins); WavLM itself never changes.
Outputs: artifacts/teacher_cache/wavlm_head.pt (head weights), wavlm.npy + wavlm.json (cache).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

from dnd.data import build_loader, load_config, load_manifest  # noqa: E402
from dnd.heads import SphereFace2C  # noqa: E402
from sred.data import TRAIN_DIR  # noqa: E402

OUT = EXP / "artifacts/teacher_cache"
WAVLM = EXP / "pretrained/wavlm/wavlm_large.pth"


class ASTP(nn.Module):
    """Attentive statistics pooling (ECAPA style, global context)."""

    def __init__(self, dim, bottleneck=128):
        super().__init__()
        self.att = nn.Sequential(nn.Conv1d(dim * 3, bottleneck, 1), nn.Tanh(), nn.Conv1d(bottleneck, dim, 1))

    def forward(self, x):  # x [B, D, T]
        mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True).clamp_min(1e-4)
        a = torch.softmax(self.att(torch.cat([x, mu.expand_as(x), sd.expand_as(x)], 1)), dim=-1)
        m = (a * x).sum(-1)
        s = ((a * x * x).sum(-1) - m * m).clamp_min(1e-6).sqrt()
        return torch.cat([m, s], 1)


class SpeakerHead(nn.Module):
    def __init__(self, n_layers=24, dim=1024, emb=192):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.pool = ASTP(dim)
        self.proj = nn.Sequential(nn.BatchNorm1d(2 * dim), nn.Linear(2 * dim, emb))

    def forward(self, layers):  # list of [B, T, D]
        h = sum(w * x for w, x in zip(torch.softmax(self.layer_w, 0), layers))
        return self.proj(self.pool(self.norm(h).transpose(1, 2)))


def load_wavlm(dev="cuda"):
    m = torchaudio.models.wavlm_large()
    m.load_state_dict(torch.load(WAVLM, map_location="cpu", weights_only=True))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(dev).eval()


@torch.no_grad()  # not inference_mode: the head trains on these outputs in `prep`
def wavlm_layers(wavlm, wav):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats, _ = wavlm.extract_features(wav)
    return [f.float() for f in feats]


def digest16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def prep(steps: int, lr: float = 1e-3):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(EXP / "configs/base.json", ["aug=null"])  # clean crops, same two-view sampler and bins
    loader, info = build_loader(cfg, seed=0, workers=8)
    wavlm = load_wavlm()
    head = SpeakerHead().cuda()
    cls = SphereFace2C(192, len(info["speakers"])).cuda()
    opt = torch.optim.AdamW([{"params": head.parameters(), "weight_decay": 1e-4},
                             {"params": cls.parameters(), "weight_decay": 0.0}], lr=lr)
    t0, run, it = time.time(), [0.0, 0.0, 0], iter(loader)
    for step in range(1, steps + 1):
        b = next(it)
        f = step / steps
        for g in opt.param_groups:  # 200-step warm-up, cosine to 5 %
            g["lr"] = lr * min(1.0, step / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * f)))
        opt.zero_grad(set_to_none=True)
        for view in ("short", "long"):
            v = b[view]
            z = head(wavlm_layers(wavlm, v["wav"].cuda(non_blocking=True)))
            loss, hit = cls(z, v["label"].cuda(), margin=0.2 * min(1.0, f / 0.2))
            (0.5 * loss.mean()).backward()
            run[0] += float(loss.mean()) * 0.5
            run[1] += float(hit.mean()) * 0.5
        opt.step()
        run[2] += 1
        if step % 100 == 0 or step == steps:
            w = torch.softmax(head.layer_w, 0).detach().cpu().numpy()
            print(f"step {step} loss={run[0] / run[2]:.3f} acc={run[1] / run[2]:.3f} top layers "
                  f"{np.argsort(-w)[:4].tolist()} ({(time.time() - t0) / step:.2f}s/step)", flush=True)
            run = [0.0, 0.0, 0]
    torch.save({"head": head.state_dict(), "steps": steps, "speakers": len(info["speakers"])}, OUT / "wavlm_head.pt")
    print("->", OUT / "wavlm_head.pt")


def load_teacher():
    wavlm = load_wavlm()
    head = SpeakerHead().cuda().eval()
    head.load_state_dict(torch.load(OUT / "wavlm_head.pt", map_location="cpu", weights_only=True)["head"])
    return wavlm, head


@torch.inference_mode()
def embed(wavlm, head, wav):
    return F.normalize(head(wavlm_layers(wavlm, wav)).float(), dim=1)


def cache(bs_audio_s: float = 256.0, limit: int = 0):
    wavlm, head = load_teacher()
    bank = dict(np.load(EXP / "artifacts/crop_bank.npz"))
    man = load_manifest(EXP / "artifacts/manifest.jsonl")
    n = len(bank["row"])
    out = np.zeros((n, 192), np.float16)
    t0, done = time.time(), 0
    for L in sorted(set(bank["length"].tolist())):
        idx = np.flatnonzero(bank["length"] == L)
        idx = idx[:limit] if limit else idx  # smoke test: first `limit` crops of each bin
        bs = max(8, int(bs_audio_s * 16000 / L))
        for i in range(0, len(idx), bs):
            ch = idx[i:i + bs]
            wav = np.stack([sf.read(str(TRAIN_DIR / man[bank["row"][c]]["audio_path"]), start=int(bank["start"][c]),
                                    frames=int(L), dtype="float32", always_2d=False)[0] for c in ch])
            out[ch] = embed(wavlm, head, torch.from_numpy(wav).cuda()).cpu().numpy().astype(np.float16)
            done += len(ch)
            if (i // bs) % 50 == 0:
                print(f"  {done}/{n} crops, {L / 16000:.2f}s bin, {(time.time() - t0) / 60:.1f} min", flush=True)
    norms = np.linalg.norm(out.astype(np.float32), axis=1)
    if limit:
        print(f"smoke test: {done} crops, norms {norms[norms > 0].min():.4f}..{norms.max():.4f}; nothing saved")
        return
    assert (norms > 0.99).all() and (norms < 1.01).all(), (norms.min(), norms.max())
    np.save(OUT / "wavlm.npy", out)
    meta = {"teacher": "wavlm", "model_file": "pretrained/wavlm/wavlm_large.pth", "model_hash16": digest16(WAVLM),
            "head_file": "artifacts/teacher_cache/wavlm_head.pt", "head_hash16": digest16(OUT / "wavlm_head.pt"),
            "crop_bank_hash16": "3a3499fbd42c2a78", "rows": n, "dim": 192, "dtype": "float16", "l2_normalised": True,
            "autocast": "bf16", "host": socket.gethostname(), "seconds": round(time.time() - t0),
            "sha256": hashlib.sha256((OUT / "wavlm.npy").read_bytes()).hexdigest()}
    (OUT / "wavlm.json").write_text(json.dumps(meta, indent=2))
    print("->", OUT / "wavlm.npy", meta["seconds"], "s")


def eval_silver():
    """Plain cosine EER of the teacher on silver full files (sanity check of the prepared head)."""
    import pandas as pd
    from evaluate import SILVER, eer
    wavlm, head = load_teacher()
    tr = pd.read_csv(EXP / "artifacts/duration_eval/trials.csv")
    files = sorted(set(tr.a) | set(tr.b))
    E = {}
    for f in files:
        x, _ = sf.read(str(SILVER / f), dtype="float32")
        E[f] = embed(wavlm, head, torch.from_numpy(x)[None].cuda())[0].cpu().numpy()
    s = np.array([E[a] @ E[b] for a, b in zip(tr.a, tr.b)])
    print(f"WavLM teacher silver EER all {eer(s, tr.label.to_numpy()):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "cache", "eval"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0, help="cache: smoke test on the first N crops of each bin")
    a = ap.parse_args()
    {"prep": lambda: prep(a.steps), "cache": lambda: cache(limit=a.limit), "eval": eval_silver}[a.stage]()
