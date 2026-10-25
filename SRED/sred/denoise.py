"""Learned denoising front-end with auxiliary losses (D-series).

Sits between the log-mel extractor and the backbone of a ReDimNet student and is part of the
submitted model (~0.15M params). It works on log-mel power *before* the per-utterance mean norm:

    X       = log mel(noisy)
    N_hat   = noise head(X)                       -> aux loss vs log mel(noisy - clean)  ("learn the noise first")
    g       = mask head(features, N_hat, time-pooled noise profile)
    S_hat   = X - softplus(g)                     (= power * mask, mask in (0, 1]: it can only remove energy)
    backbone(mean_norm(S_hat))                    -> aux loss mean_norm(S_hat) vs mean_norm(log mel(clean))

The mask bias starts at -8 (softplus ~ 3e-4), so at init the model is exactly the pretrained student.
Training pairs come from sred.augment: clean crop -> matched private-like noise, quieter overlapping
speaker, generic noise / reverb; the clean crop is the target.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _Block(nn.Module):
    def __init__(self, c: int, dil: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=(1, dil), dilation=(1, dil))  # dilated along time
        self.norm = nn.GroupNorm(1, c)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))


class Denoiser(nn.Module):
    def __init__(self, channels: int = 48, dilations=(1, 2, 4, 8, 1, 2)):
        super().__init__()
        c = channels
        self.inp = nn.Conv2d(1, c, 3, padding=1)
        self.blocks = nn.Sequential(*[_Block(c, d) for d in dilations])
        self.noise_head = nn.Conv2d(c, 1, 1)
        # mask sees local features, the noise estimate and the utterance-level (time-pooled) noise profile
        self.mask_head = nn.Sequential(nn.Conv2d(2 * c + 1, c, 1), nn.GELU(), nn.Conv2d(c, 1, 1))
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.constant_(self.mask_head[-1].bias, -8.0)

    def forward(self, logmel: torch.Tensor):
        """logmel: [B, F, T] (log power, not mean-normalised). Returns (enhanced, noise_estimate)."""
        x = logmel.unsqueeze(1)
        mu = x.mean(-1, keepdim=True)
        h = F.gelu(self.inp(x - mu))  # level-independent input
        for blk in self.blocks:  # full-resolution activations are large: recompute them in backward
            h = checkpoint(blk, h, use_reentrant=False) if self.training and torch.is_grad_enabled() else blk(h)
        n_hat = self.noise_head(h) + mu  # back to absolute log power
        prof = h.mean(-1, keepdim=True).expand_as(h)
        g = self.mask_head(torch.cat([h, prof, n_hat - mu], 1))
        return (x - F.softplus(g)).squeeze(1), n_hat.squeeze(1)


class DenoiseFrontEnd(nn.Module):
    """Replaces ReDimNetWrap.spec: mel extractor (no params) + Denoiser + the original mean norm."""

    def __init__(self, spec: nn.Module, denoiser: Denoiser):
        super().__init__()
        self.spec = spec
        self.dn = denoiser
        self.last = None  # (enhanced, noise_estimate) of the last forward, for the aux losses

    def logmel(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            return (self.spec.torchfbank(wav.float()) + 1e-6).log()

    @staticmethod
    def mn(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(-1, keepdim=True)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        s, n_hat = self.dn(self.logmel(wav))
        self.last = (s, n_hat)
        return self.mn(s).to(wav.dtype)


def attach_denoiser(model: nn.Module, cfg: dict) -> nn.Module:
    """In place: model.spec -> DenoiseFrontEnd(model.spec, Denoiser(**cfg))."""
    assert not isinstance(model.spec, DenoiseFrontEnd)
    model.spec = DenoiseFrontEnd(model.spec, Denoiser(**cfg))
    return model


def aux_losses(front: DenoiseFrontEnd, clean: torch.Tensor, noisy: torch.Tensor):
    """L1 losses after the last forward on `noisy`: enhanced vs clean (mean-normalised log mel) and
    noise estimate vs log mel of the residual (noisy - clean), on frames/bands where it is audible."""
    s, n_hat = front.last
    with torch.no_grad():
        c = front.logmel(clean)
        r = front.logmel(noisy - clean)
    l_clean = (front.mn(s.float()) - front.mn(c)).abs().mean()
    floor = c.amax((1, 2), keepdim=True) - 12.0  # ignore residual far below the speech (numerical floor)
    l_noise = ((n_hat.float() - r).abs() * (r > floor)).sum() / (r > floor).sum().clamp_min(1)
    return l_clean, l_noise
