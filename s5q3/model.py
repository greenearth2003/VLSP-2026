"""S5 = ReDimNet2-B6 (PalabraAI) compressed below 7M parameters by activation-aware low-rank factorisation of 14
1x1 projections, then trained with KD and ensemble distillation (see README.md). This module rebuilds the exact network and loads the
trained weights from assets/s5_model.pt. Input: float waveform batch [B, T], 16 kHz, values in [-1, 1].
Output: 192-d embedding (NOT normalised; `embed` L2-normalises it)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG.parent / "third_party"))  # PalabraAI redimnet2 source (MIT), unmodified copy

ASSET = PKG / "assets/s5_model.pt"


class LowRank(nn.Module):
    """y = B(A x) replacing a 1x1 Conv1d or a Linear layer (rank r)."""

    def __init__(self, kind: str, n_in: int, n_out: int, rank: int):
        super().__init__()
        if kind == "conv1d":
            self.a = nn.Conv1d(n_in, rank, 1, bias=False)
            self.b = nn.Conv1d(rank, n_out, 1)
        else:
            self.a = nn.Linear(n_in, rank, bias=False)
            self.b = nn.Linear(rank, n_out)

    def forward(self, x):
        return self.b(self.a(x))


def _set_module(model: nn.Module, name: str, new: nn.Module):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, new)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_s5(device: str = "cpu", asset: Path = ASSET) -> nn.Module:
    from redimnet2.redimnet2 import ReDimNet2Wrap
    st = torch.load(asset, map_location="cpu", weights_only=False)
    model = ReDimNet2Wrap(**dict(st["b6_cfg"]))
    for name, r in st["ranks"].items():
        m = model.get_submodule(name)
        if isinstance(m, nn.Conv1d):
            _set_module(model, name, LowRank("conv1d", m.in_channels, m.out_channels, r))
        else:
            _set_module(model, name, LowRank("linear", m.in_features, m.out_features, r))
    for name in st["drop"]:
        _set_module(model, name, nn.Identity())
    model.load_state_dict(st["model"])  # strict: every tensor must match
    return model.to(device).eval()


@torch.inference_mode()
def embed(model: nn.Module, wav: np.ndarray, device: str = "cpu", max_sec: float = 60.0,
          min_sec: float = 0.5) -> np.ndarray:
    """Full-utterance embedding, identical to the competition pipeline (sred.embed.embed_wav, mode='full'):
    keep the first 60 s; if shorter than 0.5 s repeat the signal up to 0.5 s; ONE forward pass; L2-normalise."""
    sr = 16000
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[: int(max_sec * sr)].to(device)
    if x.shape[-1] < int(min_sec * sr):
        reps = int(np.ceil(min_sec * sr / max(1, x.shape[-1])))
        x = x.repeat(reps)[: int(min_sec * sr)]
    e = F.normalize(model(x[None]), dim=-1)[0]
    return e.float().cpu().numpy()
