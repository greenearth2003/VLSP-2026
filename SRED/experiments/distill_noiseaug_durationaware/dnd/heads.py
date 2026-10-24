"""Training-only classification heads and the E0 / DAME objectives (NEW_PLAN §0).

SphereFace2-C (WeSpeaker / ReDimNet recipe): per-class one-vs-all logistic loss on g(cos) = 2((cos+1)/2)^t - 1,
CosFace-type margin, positive weight lambda. Returns PER-SAMPLE losses so a view can be processed in chunks with
exact gradient accumulation.

  single  (E0): one 192-D head; L = 0.5 * L_short + 0.5 * L_long
  dame    (E1-E3): one head per prefix d_k in {24, 48, 96, 192};
          L_view = sum_k C[view][k] * SF2_k(z[:, :d_k]);  L = (1 - alpha(p)) * L_short + alpha(p) * L_long,
          alpha(p) = max(end, start - (start - end) * p / until)   (= max(0.5, 1 - p) for 1.0 / 0.5 / 0.5)
Margins ramp linearly 0 -> target over the first `margin_warmup` fraction of training (every head).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SphereFace2C(nn.Module):
    def __init__(self, dim, n_cls, scale=32.0, lanbuda=0.7, t=3, init: torch.Tensor | None = None):
        super().__init__()
        self.scale, self.lanbuda, self.t = scale, lanbuda, t
        self.weight = nn.Parameter(init.clone() if init is not None else torch.randn(n_cls, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, emb, label, margin: float):
        """Per-sample loss [B] and top-1 hits [B]."""
        cos = F.linear(F.normalize(emb), F.normalize(self.weight))
        g = 2 * ((cos + 1) / 2).pow(self.t) - 1
        pos = self.scale * (g - margin) + self.bias
        neg = self.scale * (g + margin) + self.bias
        onehot = F.one_hot(label, cos.shape[1]).float()
        loss = (onehot * self.lanbuda * F.softplus(-pos) + (1 - onehot) * (1 - self.lanbuda) * F.softplus(neg)).sum(1)
        return loss, (cos.argmax(-1) == label).float()


class Objective(nn.Module):
    def __init__(self, cfg: dict, n_cls: int, emb_dim: int, centroids: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        self.kind = cfg["type"]
        self.dims = cfg["prefix_dims"] if self.kind == "dame" else [emb_dim]
        self.margins = cfg["margins"] if self.kind == "dame" else [cfg["margin"]]
        assert self.dims[-1] == emb_dim and len(self.dims) == len(self.margins)

        def init(d):  # speaker centroids of the pretrained model, restricted to the prefix and re-normalized
            return None if centroids is None else F.normalize(centroids[:, :d], dim=1)

        self.heads = nn.ModuleList(SphereFace2C(d, n_cls, cfg["scale"], cfg["lanbuda"], cfg["t"], init(d))
                                   for d in self.dims)

    def margin_scale(self, p: float) -> float:
        w = self.cfg["margin_warmup"]
        return 1.0 if w <= 0 else min(1.0, p / w)

    def alpha(self, p: float) -> float:
        if self.kind != "dame":
            return 0.5
        a = self.cfg["alpha"]
        return max(a["end"], a["start"] - (a["start"] - a["end"]) * p / a["until"])

    def view_weight(self, view: str, p: float) -> float:
        a = self.alpha(p)
        return a if view == "long" else 1 - a

    def view_loss(self, z, label, view: str, p: float):
        """Per-sample loss [B] of one view (before the alpha weight), and per-head stats."""
        ms = self.margin_scale(p)
        coef = self.cfg["C"][view] if self.kind == "dame" else [1.0]
        total, stats = 0.0, {}
        for head, d, m, c in zip(self.heads, self.dims, self.margins, coef):
            l, hit = head(z[:, :d], label, m * ms)
            total = total + c * l
            stats[d] = (l.detach().sum().item(), hit.sum().item())
        return total, stats
