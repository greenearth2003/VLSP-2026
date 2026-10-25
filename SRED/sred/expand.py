"""Function-preserving depth expansion of a ReDimNet student (LLaMA-Pro-style block expansion).

Inserts copies of pretrained blocks right after the originals, with the residual branch zero-gated, so the
expanded model computes exactly the same function at init and the new blocks start from pretrained weights:

  - 2D: copy of the last ConvBlock2d (ResBasicBlock) of a stage; bn2 gamma = beta = 0 -> relu(0 + x) = x
        (x is the previous block's ReLU output, so >= 0).
  - 1D: copy of the stage's TimeContextBlock1d; exp_dim_conv weight = bias = 0 -> skip + 0 = x.

cfg = {"blocks2d": [stage indices], "tcb": [stage indices]}; stored in checkpoints as "expand" so
sred.models.load_finetuned can rebuild the architecture before loading the weights.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _is(mod, name):
    return type(mod).__name__ == name


def base_state(base_name: str, cfg: dict | None) -> dict:
    """Pretrained weights in the layout of an expanded checkpoint: the inserted blocks zero-gated, so WiSE
    interpolation pulls them toward identity (the pretrained function), never toward a missing key."""
    from .models import load_model
    m = load_model(base_name)
    if cfg:
        expand_model(m, cfg, zero_init=True)
    return m.state_dict()


def expand_model(model: nn.Module, cfg: dict, zero_init: bool = True) -> list[nn.Module]:
    """In place. Returns the inserted modules (for a separate optimizer group)."""
    new = []
    for kind, test in (("blocks2d", "ConvBlock2d"), ("tcb", "TimeContextBlock1d")):
        for s in cfg.get(kind, []):
            stage = getattr(model.backbone, f"stage{s}")
            mods = list(stage)
            idx = max(i for i, m in enumerate(mods) if _is(m, test))
            blk = copy.deepcopy(mods[idx])
            if zero_init:
                with torch.no_grad():
                    if kind == "blocks2d":
                        assert _is(blk.conv_block, "ResBasicBlock"), type(blk.conv_block)
                        blk.conv_block.bn2.weight.zero_()
                        blk.conv_block.bn2.bias.zero_()
                    else:
                        blk.exp_dim_conv.weight.zero_()
                        blk.exp_dim_conv.bias.zero_()
            mods.insert(idx + 1, blk)
            setattr(model.backbone, f"stage{s}", nn.Sequential(*mods))
            new.append(blk)
    return new
