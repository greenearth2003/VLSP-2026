"""Low-rank compression of ReDimNet2-B6 to the 7M budget (E5-b6-lowrank-kd), our own implementation.

Every nn.Linear and 1x1 nn.Conv1d (groups=1) is a candidate: W [out, in] ≈ U_r S_r V_r^T, replaced by two layers
in -> r (no bias) -> out (keeps the bias), initialised from the truncated SVD (sqrt(S) split between them).
Ranks: every layer keeps the same share of energy (largest share that fits the budget); a layer stays dense when its
factorised form would not be smaller. Default method "whiten" (activation-aware SVD): truncation minimises the error of
each layer's OUTPUT on 256 real train crops instead of the weight error.

    .venv/bin/python experiments/distill_noiseaug_durationaware/dnd/lowrank.py SRC.pt OUT.pt [--budget 6950000]
        SRC: a B6 checkpoint (pretrained format or a train.py step<k>.pt); OUT: checkpoint with a "lowrank" key that
        load_lowrank() (and evaluate.py) rebuilds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
B6_PRETRAINED = EXP / "pretrained/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt"


class LowRank(nn.Module):
    """in -> r -> out, for a Linear or a 1x1 Conv1d (same call signature as the layer it replaces)."""

    def __init__(self, kind: str, n_in: int, n_out: int, r: int, bias: bool):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.a, self.b = nn.Linear(n_in, r, bias=False), nn.Linear(r, n_out, bias=bias)
        else:
            self.a, self.b = nn.Conv1d(n_in, r, 1, bias=False), nn.Conv1d(r, n_out, 1, bias=bias)

    def forward(self, x):
        return self.b(self.a(x))


def candidates(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv1d) and m.kernel_size == (1,) and m.groups == 1)]


def _set(model, name, new):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, new)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def spectra(model, cov=None):
    """Per candidate layer: cumulative normalised energy of W's spectrum (weight SVD) or, with input covariances
    `cov` measured on real audio, of W C^1/2's spectrum (= energy of the layer's OUTPUT on that audio)."""
    out = {}
    for n, m in candidates(model):
        W = m.weight.detach().float().reshape(m.weight.shape[0], -1)
        M = W @ cov[n][0] if cov is not None else W
        s2 = torch.linalg.svdvals(M) ** 2
        out[n] = (torch.cumsum(s2, 0) / s2.sum(), W.shape[1], W.shape[0])
    return out


def choose_ranks(model, budget: int, cov=None):
    """Keep the same share tau of (output) energy in every candidate layer; binary-search the largest tau that fits
    the budget. A greedy per-parameter allocation was tried first and collapsed the thin 4608 <-> C stage-interface
    layers to rank 8 (cos 0.31 to the original), so every layer gets the same energy share instead."""
    info = spectra(model, cov)
    dense = n_params(model) - sum(i * o for _, i, o in info.values())

    def ranks_for(tau):
        rk, tot = {}, dense
        for n, (ce, i, o) in info.items():
            r = int((ce < tau).sum()) + 1
            if r * (i + o) < i * o:
                rk[n] = r
                tot += r * (i + o)
            else:
                tot += i * o
        return rk, tot

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ranks_for(mid)[1] <= budget else (lo, mid)
    rk, tot = ranks_for(lo)
    return rk, tot, lo


@torch.no_grad()
def input_covariances(model, wavs, eps=1e-5):
    """Input covariance C = X X^T / N of every candidate layer on real audio, and its symmetric square root and
    inverse square root (for whitened SVD)."""
    acc, hooks = {}, []
    for n, m in candidates(model):
        def hook(mod, inp, out, n=n):
            x = inp[0].double()  # float64: the fp32 eigh of 4608x4608 / 6048x6048 covariances fails in LAPACK
            x = x.transpose(1, 2).reshape(-1, x.shape[1]) if x.dim() == 3 and isinstance(mod, nn.Conv1d) \
                else x.reshape(-1, x.shape[-1])
            c, k = acc.get(n, (0.0, 0))
            acc[n] = (c + x.T @ x, k + x.shape[0])
        hooks.append(m.register_forward_hook(hook))
    for w in wavs:
        model(w)
    for h in hooks:
        h.remove()
    cov = {}
    for n, (c, k) in acc.items():
        C = c / k
        assert torch.isfinite(C).all(), n
        C = C + eps * C.diagonal().mean() * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
        ev, Q = torch.linalg.eigh(C)
        ev = ev.clamp_min(1e-12)
        cov[n] = ((Q @ torch.diag(ev.sqrt()) @ Q.T).float(), (Q @ torch.diag(ev.rsqrt()) @ Q.T).float())
    return cov


def apply_ranks(model, ranks: dict, init_from_svd: bool = True, cov=None):
    """Replace the chosen layers by LowRank modules, initialised from the truncated SVD of W (or of W C^1/2 then
    mapped back with C^-1/2 when input covariances are given: best rank-r fit of the layer's outputs on that audio).
    init_from_svd=False only builds the structure (to load a saved state)."""
    mods = dict(model.named_modules())
    for n, r in ranks.items():
        m = mods[n]
        kind = "linear" if isinstance(m, nn.Linear) else "conv1d"
        n_out, n_in = m.weight.shape[0], m.weight.shape[1]
        lr = LowRank(kind, n_in, n_out, r, m.bias is not None).to(m.weight.device, m.weight.dtype)
        if init_from_svd:
            W = m.weight.detach().float().reshape(n_out, n_in)
            if cov is None:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                sq = S[:r].sqrt()
                A, B = (Vh[:r] * sq[:, None]), (U[:, :r] * sq[None])  # W ≈ B @ A
            else:
                half, inv_half = cov[n]
                U, S, Vh = torch.linalg.svd(W @ half, full_matrices=False)
                sq = S[:r].sqrt()
                A, B = (Vh[:r] * sq[:, None]) @ inv_half, (U[:, :r] * sq[None])  # W C^1/2 ≈ B A'  ->  W ≈ B A' C^-1/2
            with torch.no_grad():
                lr.a.weight.copy_(A.reshape(lr.a.weight.shape))
                lr.b.weight.copy_(B.reshape(lr.b.weight.shape))
                if m.bias is not None:
                    lr.b.bias.copy_(m.bias)
        _set(model, n, lr)
    return model


def load_source(src: Path):
    """B6 from the pretrained file or from a train.py checkpoint of a B6 run (E4)."""
    sys.path.insert(0, str(SRED))
    from sred.models import _load_redimnet2_pt
    model = _load_redimnet2_pt(B6_PRETRAINED)
    st = torch.load(src, map_location="cpu", weights_only=False)
    if "model" in st and "state_dict" not in st:  # train.py checkpoint
        model.load_state_dict(st["model"])
    return model


def calibration_audio(n=256, sec=2.0, seed=0):
    """n clean bank crops of `sec` seconds (train data), batches of 32, for the input covariances."""
    import json
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, str(SRED))
    from sred.data import TRAIN_DIR
    bank = np.load(EXP / "artifacts/crop_bank.npz")
    man = [json.loads(ln) for ln in open(EXP / "artifacts/manifest.jsonl")]
    L = int(sec * 16000)
    idx = np.random.default_rng(seed).choice(np.flatnonzero(bank["length"] == L), n, replace=False)
    x = np.stack([sf.read(str(TRAIN_DIR / man[bank["row"][c]]["audio_path"]), start=int(bank["start"][c]), frames=L,
                          dtype="float32")[0] for c in idx])
    return [torch.from_numpy(x[i:i + 32]) for i in range(0, n, 32)]


def compress(src: Path, out: Path, budget: int, method: str = "whiten"):
    model = load_source(src).eval()
    cov = input_covariances(model, calibration_audio()) if method == "whiten" else None
    ranks, est, tau = choose_ranks(model, budget, cov)
    apply_ranks(model, ranks, cov=cov)
    n = n_params(model)
    assert n < 7_000_000, n
    torch.save({"base_model": "redimnet2_b6_mix_teacher_pt", "lowrank": ranks, "model": model.state_dict(),
                "source": str(src), "params": n, "budget": budget, "method": method, "energy_kept": tau}, out)
    return model, ranks, n


def load_lowrank(path: Path):
    """Rebuild a compressed (and possibly fine-tuned) model saved with a "lowrank" key."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    sys.path.insert(0, str(SRED))
    from sred.models import _load_redimnet2_pt
    model = apply_ranks(_load_redimnet2_pt(B6_PRETRAINED), st["lowrank"], init_from_svd=False)
    model.load_state_dict(st["model"])
    return model.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--budget", type=int, default=6_950_000)
    ap.add_argument("--method", choices=["whiten", "svd"], default="whiten",
                    help="whiten: activation-aware SVD on real train crops (default); svd: plain weight SVD")
    a = ap.parse_args()
    m, ranks, n = compress(Path(a.src), Path(a.out), a.budget, a.method)
    print(f"{len(ranks)} layers factorised, {n:,} params (budget {a.budget:,}) -> {a.out}")
    for k, r in sorted(ranks.items(), key=lambda x: x[1])[:6]:
        print(f"  lowest ranks: {k} r={r}")
