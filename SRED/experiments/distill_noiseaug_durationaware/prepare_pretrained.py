"""Collect and check the approved pretrained weights for E0-E3 (NEW_PLAN.md) in this experiment's own folder.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prepare_pretrained.py [--force]

pretrained/
  redimnet2/b4-vox2-lm.pt               student init (E0-E3), copied from SRED/pretrained/redimnet2/
  redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt   B6 teacher (E3), same source
  wavlm/wavlm_large.pth                 WavLM-Large teacher (E2, E3), torchaudio state dict for
                                        torchaudio.models.wavlm_large() (no transformers needed)
  MANIFEST.json                         sha256, size, source, param counts, embedding dims

Each file is loaded and run on 2 s of noise so a broken download fails here, not inside training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path.insert(0, str(SRED))

from sred.models import _load_redimnet2_pt  # noqa: E402

OUT = EXP / "pretrained"
WAVLM_URL = "https://download.pytorch.org/torchaudio/models/wavlm_large.pth"
FILES = {
    "redimnet2/b4-vox2-lm.pt": ("copy", SRED / "pretrained/redimnet2/b4-vox2-lm.pt"),
    "redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt": ("copy", SRED / "pretrained/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt"),
    "wavlm/wavlm_large.pth": ("url", WAVLM_URL),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dst: Path, kind: str, src, force: bool):
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if kind == "copy":
        shutil.copy2(src, tmp)
    else:
        urllib.request.urlretrieve(src, tmp)
    tmp.rename(dst)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def check_redimnet2(path: Path) -> dict:
    model = _load_redimnet2_pt(path).eval()
    emb = model(torch.randn(1, 32000))
    sd_keys = {k.split(".")[0] for k in model.state_dict()}
    return {
        "params": n_params(model),
        "embed_dim": int(emb.shape[-1]),
        "top_level_modules": sorted(sd_keys),
        "has_classification_head": False,  # checkpoint holds spec/backbone/pool/bn/linear only
        "model_config": model.model_config,
    }


@torch.no_grad()
def check_wavlm(path: Path) -> dict:
    import torchaudio

    model = torchaudio.models.wavlm_large()
    res = model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    assert not res.missing_keys and not res.unexpected_keys, res
    feats, _ = model.eval().extract_features(torch.randn(1, 32000))
    return {
        "params": n_params(model),
        "num_layers": len(feats),
        "hidden_dim": int(feats[-1].shape[-1]),
        "frames_per_2s": int(feats[-1].shape[1]),
        "loader": "torchaudio.models.wavlm_large(); load_state_dict(torch.load(path))",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-copy / re-download even if present")
    a = ap.parse_args()

    manifest = {}
    for rel, (kind, src) in FILES.items():
        dst = OUT / rel
        fetch(dst, kind, src, a.force)
        info = check_wavlm(dst) if rel.startswith("wavlm/") else check_redimnet2(dst)
        manifest[rel] = {"source": str(src), "bytes": dst.stat().st_size, "sha256": sha256(dst), **info}
        print(f"{rel:40s} {info['params'] / 1e6:7.2f}M params  ok")

    assert manifest["redimnet2/b4-vox2-lm.pt"]["params"] < 7_000_000, "student must stay below 7M"
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {OUT / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
