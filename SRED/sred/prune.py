"""Structured (channel) pruning of ReDimNet2-B6 to fit the 7M-parameter budget.

Three groups of channels are internal to a block and can be removed without touching the
C*F = 4608 interface that ties the stages together:

  1d    TimeContextBlock1d bottleneck hC (red_dim_conv -> tcm -> exp_dim_conv): 58 % of B6's params.
        Attention / FFN *inner* dims are kept, only the residual-stream width is cut.
  2d    ResBasicBlock mid channels (conv1pw -> bn1 -> conv2(dw) -> conv2pw).
  head  head Conv2d out channels (224), which also slices the ASTP pooling, the final BN and
        the embedding linear (feature index = c * F_out + f).

Importance = weight magnitude (product of the in- and out-side norms of each channel). The pruned
model is saved as a checkpoint carrying the kept indices, so `load_model(path)` rebuilds it.

    python -m sred.prune --keep-1d 0.4 --keep-2d 0.5 --keep-head 0.8 --out exp/ft/P_B6/init.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .models import count_params, load_model


def _topk(score: torch.Tensor, keep: float, mult: int = 8, min_keep: int = 16) -> torch.Tensor:
    n = len(score)
    k = max(min_keep, int(round(n * keep / mult)) * mult)
    k = min(n, k)
    return torch.sort(torch.topk(score, k).indices).values


def _conv_out(conv, idx):
    conv.weight = nn.Parameter(conv.weight.data[idx].clone())
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[idx].clone())
    conv.out_channels = len(idx)


def _conv_in(conv, idx):
    conv.weight = nn.Parameter(conv.weight.data[:, idx].clone())
    conv.in_channels = len(idx)


def _conv_dw(conv, idx):  # depthwise: groups == channels
    conv.weight = nn.Parameter(conv.weight.data[idx].clone())
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[idx].clone())
    conv.in_channels = conv.out_channels = conv.groups = len(idx)


def _lin_out(lin, idx):
    lin.weight = nn.Parameter(lin.weight.data[idx].clone())
    if lin.bias is not None:
        lin.bias = nn.Parameter(lin.bias.data[idx].clone())
    lin.out_features = len(idx)


def _lin_in(lin, idx):
    lin.weight = nn.Parameter(lin.weight.data[:, idx].clone())
    lin.in_features = len(idx)


def _norm(norm, idx):
    """BatchNorm*, nn.LayerNorm or the redimnet2 channels-first LayerNorm."""
    for name in ("weight", "bias"):
        p = getattr(norm, name, None)
        if p is not None:
            setattr(norm, name, nn.Parameter(p.data[idx].clone()))
    for name in ("running_mean", "running_var"):
        b = getattr(norm, name, None)
        if b is not None:
            setattr(norm, name, b[idx].clone())
    if hasattr(norm, "num_features"):
        norm.num_features = len(idx)
    if hasattr(norm, "normalized_shape"):
        norm.normalized_shape = (len(idx),)
    if hasattr(norm, "C"):  # redimnet2 LayerNorm
        norm.C = (len(idx),)


# ----------------------------------------------------------------------------- 1d blocks
def prune_tcb(block, idx):
    _conv_out(block.red_dim_conv[0], idx)
    _norm(block.red_dim_conv[1], idx)
    for sub in block.tcm:
        t = type(sub).__name__
        if t == "ConvNeXtLikeBlock":
            for dw in sub.dwconvs:
                _conv_dw(dw, idx)
            _norm(sub.norm, idx)
            _conv_out(sub.pwconv1, idx)
            _conv_in(sub.pwconv1, idx)
        elif t.startswith("TransformerEncoderLayer"):
            a = sub.attention
            for p in (a.q_proj, a.k_proj, a.v_proj):
                _lin_in(p, idx)
            _lin_out(a.out_proj, idx)
            _norm(sub.layer_norm, idx)
            _norm(sub.final_layer_norm, idx)
            _lin_in(sub.feed_forward.intermediate_dense, idx)
            _lin_out(sub.feed_forward.output_dense, idx)
        else:
            raise NotImplementedError(t)
    _conv_in(block.exp_dim_conv, idx)


def score_tcb(block):
    w_in = block.red_dim_conv[0].weight.data.flatten(1).norm(dim=1)
    w_out = block.exp_dim_conv.weight.data.transpose(0, 1).flatten(1).norm(dim=1)
    return w_in * w_out * block.red_dim_conv[1].weight.data.abs()


# ----------------------------------------------------------------------------- 2d blocks
def prune_res(block, idx):
    _conv_out(block.conv1pw, idx)
    _norm(block.bn1, idx)
    _conv_dw(block.conv2, idx)
    _conv_in(block.conv2pw, idx)


def score_res(block):
    w_in = block.conv1pw.weight.data.flatten(1).norm(dim=1)
    w_out = block.conv2pw.weight.data.transpose(0, 1).flatten(1).norm(dim=1)
    return w_in * w_out * block.bn1.weight.data.abs()


# ----------------------------------------------------------------------------- head group
def head_feature_idx(model, c_idx):
    C = model.backbone.head.out_channels
    n_feat = model.linear.in_features // 2
    F_out = n_feat // C
    return torch.cat([c * F_out + torch.arange(F_out) for c in c_idx.tolist()]), n_feat


def prune_head(model, c_idx):
    fidx, n_feat = head_feature_idx(model, c_idx)
    _conv_out(model.backbone.head, c_idx)
    pool = model.pool
    _conv_in(pool.linear1, torch.cat([fidx, n_feat + fidx, 2 * n_feat + fidx]) if pool.global_context_att else fidx)
    _conv_out(pool.linear2, fidx)
    pool.in_dim = len(fidx)
    both = torch.cat([fidx, n_feat + fidx])
    _norm(model.bn, both)
    _lin_in(model.linear, both)


def score_head(model):
    C = model.backbone.head.out_channels
    n_feat = model.linear.in_features // 2
    F_out = n_feat // C
    w = model.linear.weight.data
    per_feat = w[:, :n_feat].norm(dim=0) + w[:, n_feat:].norm(dim=0)
    return per_feat.view(C, F_out).sum(1) * model.backbone.head.weight.data.flatten(1).norm(dim=1)


# ----------------------------------------------------------------------------- data-driven pruning
def _ridge(H, Y, lam=1e-3):
    """Least squares Y ~ W H + b. H: [k, N], Y: [o, N] -> W [o, k], b [o]."""
    H = torch.cat([H, torch.ones(1, H.shape[1], device=H.device)], 0).double()
    Y = Y.double()
    G = H @ H.T
    G += lam * G.diagonal().mean() * torch.eye(len(G), device=H.device, dtype=G.dtype)
    Wb = torch.linalg.solve(G, H @ Y.T).T  # [o, k+1]
    return Wb[:, :-1].float(), Wb[:, -1].float()


class _Tap:
    """Forward hook keeping a seeded random subsample of positions of a module's output (channels x N).
    Two taps with the same seed (and the same batches) sample identical positions."""

    def __init__(self, mod, per_item: int, seed: int = 0):
        self.buf, self.per_item, self.seed = [], per_item, seed
        self.gen = None
        self.h = mod.register_forward_hook(self)

    def __call__(self, mod, inp, out):
        x = out.detach()
        if x.ndim == 2:  # [B, C]
            self.buf.append(x.T)
            return
        x = x.flatten(2)  # [B, C, P]
        B, C, Pn = x.shape
        n = min(self.per_item, Pn)
        if self.gen is None:
            self.gen = torch.Generator(device=x.device).manual_seed(self.seed)
        sel = torch.randint(0, Pn, (B, n), device=x.device, generator=self.gen)
        self.buf.append(torch.gather(x, 2, sel[:, None, :].expand(B, C, n)).permute(1, 0, 2).reshape(C, -1))

    def cat(self):
        return torch.cat(self.buf, 1)

    def close(self):
        self.h.remove()


@torch.no_grad()
def _run(model, wavs, bs=32):
    for i in range(0, len(wavs), bs):
        model(wavs[i:i + bs])


def _tap_pair(model, wavs, m_reg, m_tgt, per_item, seed):
    tr, tt = _Tap(m_reg, per_item, seed), _Tap(m_tgt, per_item, seed)
    _run(model, wavs)
    H, Y = tr.cat(), tt.cat()
    tr.close(), tt.close()
    return H, Y


def _tap_one(model, wavs, m_reg, per_item, seed):
    tr = _Tap(m_reg, per_item, seed)
    _run(model, wavs)
    H = tr.cat()
    tr.close()
    return H


def _relerr(Y, W, H, b):
    return float((Y - (W @ H + b[:, None])).pow(2).mean() / Y.pow(2).mean())


def _adaptive_k(score, H, Y, keep_min, err_target, mult=8, lam=1e-3):
    """Smallest keep ratio >= keep_min whose ridge refit reconstructs Y within err_target."""
    n = len(score)
    for keep in [keep_min + 0.1 * i for i in range(int((1 - keep_min) / 0.1) + 2)]:
        keep = min(1.0, keep)
        idx = _topk(score, keep, mult)
        W, b = _ridge(H[idx], Y, lam)
        if _relerr(Y, W, H[idx], b) <= err_target or keep >= 1.0:
            return idx, W, b
    return idx, W, b


@torch.no_grad()
def prune_data_driven(model, wavs, keep_1d=0.3, keep_2d=0.5, keep_head=0.8, err_1d=0.05, err_2d=0.15,
                      per_item=64, head_wavs=None, log=print):
    """Sequential (block by block) pruning. Importance = output-weight norm x activation RMS.
    1d/head: slice, re-run to get the ACTUAL pruned activations (internal channel mixing), refit the
    output layer to the original block output. 2d: mid channels are independent, refit directly.
    Keep ratio per block grows from the minimum until the refit error target is met."""
    model.eval()
    indices = {"1d": {}, "2d": {}, "head": None}
    seed = 0
    for kind, name, mod in list(iter_blocks(model)):
        seed += 1
        if kind == "1d" and keep_1d < 1:
            H, Y = _tap_pair(model, wavs, mod.tcm, mod.exp_dim_conv, per_item, seed)
            W = mod.exp_dim_conv.weight.data.flatten(1)
            score = W.norm(dim=0) * H.pow(2).mean(1).sqrt()
            idx, _, _ = _adaptive_k(score, H, Y, keep_1d, err_1d)
            prune_tcb(mod, idx)
            Hp = _tap_one(model, wavs, mod.tcm, per_item, seed)  # actual activations of the pruned block
            e0 = _relerr(Y, mod.exp_dim_conv.weight.data.flatten(1), Hp, mod.exp_dim_conv.bias.data)
            Wn, bn_ = _ridge(Hp, Y)
            mod.exp_dim_conv.weight.data.copy_(Wn[:, :, None])
            mod.exp_dim_conv.bias.data.copy_(bn_)
            indices["1d"][name] = idx.tolist()
            log(f"  1d {name}: {len(score)}->{len(idx)}  rel.err sliced {e0:.3f} -> refit {_relerr(Y, Wn, Hp, bn_):.3f}")
        elif kind == "2d" and keep_2d < 1:
            G, Y = _tap_pair(model, wavs, mod.conv2, mod.conv2pw, per_item, seed)
            W = mod.conv2pw.weight.data.flatten(1)
            score = W.norm(dim=0) * G.pow(2).mean(1).sqrt()
            idx, Wn, bn_ = _adaptive_k(score, G, Y, keep_2d, err_2d)
            prune_res(mod, idx)
            mod.conv2pw.weight.data.copy_(Wn[:, :, None, None])
            mod.conv2pw.bias.data.copy_(bn_)
            indices["2d"][name] = idx.tolist()
            log(f"  2d {name}: {len(score)}->{len(idx)}  rel.err refit {_relerr(Y, Wn, G[idx], bn_):.3f}")
    if keep_head < 1:
        hw = wavs if head_wavs is None else head_wavs
        Hf, Y = _tap_pair(model, hw, model.bn, model.linear, per_item, 0)
        C = model.backbone.head.out_channels
        n_feat = model.linear.in_features // 2
        F_out = n_feat // C
        per_feat = model.linear.weight.data.norm(dim=0) * Hf.pow(2).mean(1).sqrt()
        score = (per_feat[:n_feat] + per_feat[n_feat:]).view(C, F_out).sum(1)
        c_idx = _topk(score, keep_head, mult=4)
        prune_head(model, c_idx)
        Hp = _tap_one(model, hw, model.bn, per_item, 0)
        b0 = model.linear.bias.data if model.linear.bias is not None else torch.zeros(Y.shape[0], device=Y.device)
        e0 = _relerr(Y, model.linear.weight.data, Hp, b0)
        Wn, bn_ = _ridge(Hp, Y, lam=1e-2)
        model.linear.weight.data.copy_(Wn)
        if model.linear.bias is not None:
            model.linear.bias.data.copy_(bn_)
        indices["head"] = c_idx.tolist()
        log(f"  head: {C}->{len(c_idx)} channels  rel.err sliced {e0:.3f} -> refit {_relerr(Y, Wn, Hp, bn_):.3f} (n={Hp.shape[1]})")
    return model, indices


def load_calib_wavs(n: int, sec: float = 3.0, seed: int = 0, list_path=None):
    """Random crops from (cleaned) train utterances for calibration."""
    import random

    import numpy as np
    import soundfile as sf

    from .data import TRAIN_DIR, read_train_list

    items = read_train_list(Path(list_path)) if list_path else read_train_list()
    rng = random.Random(seed)
    rels = [r for _, r in rng.sample(items, n)]
    L = int(sec * 16000)
    out = []
    for r in rels:
        info = sf.info(str(TRAIN_DIR / r))
        st = rng.randint(0, max(0, info.frames - L))
        w, _ = sf.read(str(TRAIN_DIR / r), start=st, frames=L, dtype="float32")
        if len(w) < L:
            w = np.tile(w, int(np.ceil(L / max(1, len(w)))))[:L]
        out.append(torch.from_numpy(w))
    return torch.stack(out)


# ----------------------------------------------------------------------------- driver
def iter_blocks(model):
    for name, mod in model.backbone.named_modules():
        t = type(mod).__name__
        if t == "TimeContextBlock1d":
            yield "1d", name, mod
        elif t == "ResBasicBlock":
            yield "2d", name, mod


def apply_indices(model, indices: dict):
    """Deterministically rebuild a pruned architecture from stored indices."""
    for kind, name, mod in iter_blocks(model):
        if name in indices["1d"]:
            prune_tcb(mod, torch.as_tensor(indices["1d"][name]))
        elif name in indices["2d"]:
            prune_res(mod, torch.as_tensor(indices["2d"][name]))
    if indices.get("head") is not None:
        prune_head(model, torch.as_tensor(indices["head"]))
    return model


def prune(model, keep_1d=0.4, keep_2d=0.5, keep_head=0.8):
    indices = {"1d": {}, "2d": {}, "head": None}
    for kind, name, mod in iter_blocks(model):
        if kind == "1d" and keep_1d < 1:
            indices["1d"][name] = _topk(score_tcb(mod), keep_1d).tolist()
        elif kind == "2d" and keep_2d < 1:
            indices["2d"][name] = _topk(score_res(mod), keep_2d).tolist()
    if keep_head < 1:
        indices["head"] = _topk(score_head(model), keep_head, mult=4).tolist()
    return apply_indices(model, indices), indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="redimnet2_b6_mix_teacher_pt")
    ap.add_argument("--keep-1d", type=float, default=0.4)
    ap.add_argument("--keep-2d", type=float, default=0.5)
    ap.add_argument("--keep-head", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", type=int, default=0, help="calibration utterances for data-driven pruning (0 = magnitude)")
    ap.add_argument("--head-data", type=int, default=6144, help="calibration utterances for the head refit")
    ap.add_argument("--err-1d", type=float, default=0.05)
    ap.add_argument("--err-2d", type=float, default=0.15)
    ap.add_argument("--train-list", default="lists/clean/train_clean.txt")
    a = ap.parse_args()
    model = load_model(a.base)
    n0 = count_params(model)
    if a.data:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        allw = load_calib_wavs(max(a.data, a.head_data), list_path=a.train_list).to(dev)
        model, idx = prune_data_driven(model.to(dev), allw[:a.data], a.keep_1d, a.keep_2d, a.keep_head,
                                       a.err_1d, a.err_2d, head_wavs=allw[:a.head_data])
        model = model.cpu()
    else:
        model, idx = prune(model, a.keep_1d, a.keep_2d, a.keep_head)
    n1 = count_params(model)
    with torch.no_grad():
        e = model.eval()(torch.randn(2, 32000))
    print(f"{a.base}: {n0 / 1e6:.3f}M -> {n1 / 1e6:.3f}M params ({n1 / n0:.1%}); emb {tuple(e.shape)}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"base_model": a.base, "prune": idx, "keep": dict(k1d=a.keep_1d, k2d=a.keep_2d, khead=a.keep_head, data=a.data),
                "model": model.state_dict(), "step": 0}, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
