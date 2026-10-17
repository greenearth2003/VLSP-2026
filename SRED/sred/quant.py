"""Simulated low-precision (fake-quant) helpers, see scripts/quant_eval.py."""
from __future__ import annotations

import torch
import torch.nn as nn

QMODS = (nn.Conv1d, nn.Conv2d, nn.Linear)


def quant_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(1).clamp_min(1e-12) / qmax
    q = (flat / scale[:, None]).round().clamp(-qmax, qmax) * scale[:, None]
    return q.reshape(w.shape)


class ActQuant(nn.Module):
    """Wraps a module; fake-quantizes its input with a fixed per-tensor scale (static INT8)."""

    def __init__(self, mod: nn.Module, bits: int = 8):
        super().__init__()
        self.mod, self.qmax = mod, 2 ** (bits - 1) - 1
        self.register_buffer("amax", torch.zeros(()))
        self.calib = True

    def forward(self, x):
        if self.calib:
            self.amax = torch.maximum(self.amax, x.detach().abs().amax())
            return self.mod(x)
        s = self.amax.clamp_min(1e-12) / self.qmax
        return self.mod((x / s).round().clamp(-self.qmax, self.qmax) * s)


def wrap_acts(model: nn.Module):
    for name, child in list(model.named_children()):
        if isinstance(child, QMODS):
            setattr(model, name, ActQuant(child))
        else:
            wrap_acts(child)
