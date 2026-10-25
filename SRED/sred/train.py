"""Supervised fine-tuning of a pretrained speaker embedding model on VLSP train data.

Loss = SubCenter-AAM-softmax (+ optional clean->noisy consistency distillation
from the frozen pretrained model).

Example:
  python -m sred.train --base redimnet_M_mix --name ft_M_aam --steps 6000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import Augmenter
from .denoise import aux_losses, attach_denoiser
from .expand import expand_model
from .data import PUBLIC_DIR, TRAIN_DIR, cohort_subset, read_train_list, read_trials, unique_files
from .embed import cache_key, embed_wav
from .metrics import compute_eer
from .models import count_params, load_model
from .scoring import cosine_scores, speaker_means

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
class SubCenterAAM(nn.Module):
    def __init__(self, emb_dim, n_cls, k=3, margin=0.2, scale=30.0, init: torch.Tensor | None = None):
        super().__init__()
        self.k, self.n_cls, self.margin, self.scale = k, n_cls, margin, scale
        w = torch.randn(n_cls * k, emb_dim) * 0.01
        if init is not None:  # start sub-centres at speaker centroids (+ jitter)
            w = init.repeat_interleave(k, 0) + 0.01 * torch.randn(n_cls * k, emb_dim)
        self.weight = nn.Parameter(w)

    def class_cos(self, emb):
        return F.linear(F.normalize(emb), F.normalize(self.weight)).view(-1, self.n_cls, self.k).amax(-1)

    def forward(self, emb, label):
        cos = self.class_cos(emb)
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.margin)
        # standard "easy margin off" fallback for theta + m > pi
        target = torch.where(theta + self.margin < math.pi, target, cos - self.margin * math.sin(self.margin))
        onehot = F.one_hot(label, self.n_cls).bool()
        logits = torch.where(onehot, target, cos) * self.scale
        loss = F.cross_entropy(logits, label)
        acc = (cos.argmax(-1) == label).float().mean()
        return loss, acc


# ----------------------------------------------------------------------------
class SphereFace2C(nn.Module):
    """SphereFace2, CosFace-type margin (C), as in WeSpeaker / the ReDimNet recipe: per-class binary
    (one-vs-all) logistic loss on g(cos) with g(z) = 2((z+1)/2)^t - 1, positive weight lambda."""

    def __init__(self, emb_dim, n_cls, margin=0.2, scale=32.0, lanbuda=0.7, t=3, init: torch.Tensor | None = None,
                 ada_h: float = 0.0):
        super().__init__()
        self.n_cls, self.margin, self.scale, self.lanbuda, self.t = n_cls, margin, scale, lanbuda, t
        # AdaFace-style quality-adaptive margin: the embedding norm is the quality proxy; low-norm (noisy, short)
        # crops get a smaller margin, high-norm ones a larger one: m_i = m * (1 + ada_h * zhat_i), zhat in [-1, 1]
        # from batch-EMA norm statistics. ada_h = 0: plain SphereFace2.
        self.ada_h = ada_h
        self.register_buffer("norm_mu", torch.tensor(20.0))
        self.register_buffer("norm_sd", torch.tensor(5.0))
        w = init.clone() if init is not None else torch.randn(n_cls, emb_dim) * 0.01
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(1))

    def class_cos(self, emb):
        return F.linear(F.normalize(emb), F.normalize(self.weight))

    def forward(self, emb, label):
        cos = self.class_cos(emb)
        g = 2 * ((cos + 1) / 2).pow(self.t) - 1
        m = self.margin
        if self.ada_h > 0:
            with torch.no_grad():
                nrm = emb.norm(dim=-1)
                if self.training:
                    self.norm_mu.mul_(0.99).add_(0.01 * nrm.mean())
                    self.norm_sd.mul_(0.99).add_(0.01 * nrm.std())
                zhat = ((nrm - self.norm_mu) / (self.norm_sd + 1e-3)).clamp(-1, 1)
            m = (self.margin * (1 + self.ada_h * zhat)).clamp_min(0)[:, None]
        pos = self.scale * (g - m) + self.bias
        neg = self.scale * (g + m) + self.bias
        onehot = F.one_hot(label, self.n_cls).float()
        loss = (onehot * self.lanbuda * F.softplus(-pos) + (1 - onehot) * (1 - self.lanbuda) * F.softplus(neg)).sum(1).mean()
        acc = (cos.argmax(-1) == label).float().mean()
        return loss, acc


def dkd_loss(s_logit, t_logit, label, alpha=1.0, gamma=2.0, T=4.0, conf_k=0):
    """Decoupled KD (target-class KD + gamma * non-target-class KD). conf_k > 0 restricts the non-target
    distribution to the teacher's top-conf_k non-target classes (confusion set, cf. TRKD)."""
    n = s_logit.shape[1]
    onehot = F.one_hot(label, n).bool()
    ps, pt = F.softmax(s_logit / T, 1), F.softmax(t_logit / T, 1)
    bs = torch.stack([(ps * onehot).sum(1), (ps * ~onehot).sum(1)], 1).clamp_min(1e-8)
    bt = torch.stack([(pt * onehot).sum(1), (pt * ~onehot).sum(1)], 1).clamp_min(1e-8)
    tckd = (bt * (bt.log() - bs.log())).sum(1).mean()
    mask = ~onehot
    if conf_k and conf_k < n - 1:
        tl = t_logit.masked_fill(onehot, -1e9)
        idx = tl.topk(conf_k, 1).indices
        mask = torch.zeros_like(onehot).scatter_(1, idx, True)
    big = -1e9
    ls = F.log_softmax((s_logit / T).masked_fill(~mask, big), 1)
    lt = F.log_softmax((t_logit / T).masked_fill(~mask, big), 1)
    nckd = (lt.exp() * (lt - ls)).masked_fill(~mask, 0).sum(1).mean()
    return (alpha * tckd + gamma * nckd) * T * T


def train_lengths(items, cache=ROOT / "lists/train_lengths.json"):
    if cache.exists():
        return json.loads(cache.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    lens = {rel: sf.info(str(TRAIN_DIR / rel)).frames for _, rel in items}
    cache.write_text(json.dumps(lens))
    return lens


class CropDataset(torch.utils.data.Dataset):
    """Each __getitem__ returns a batch (list of indices from the sampler) of random crops."""

    def __init__(self, items, spk2id, lengths, crop_sec):
        self.items, self.spk2id, self.lengths = items, spk2id, lengths
        self.crop = int(crop_sec * 16000)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        spk, rel = self.items[i]
        n = self.lengths[rel]
        if n > self.crop:
            st = random.randint(0, n - self.crop)
            wav, _ = sf.read(str(TRAIN_DIR / rel), start=st, frames=self.crop, dtype="float32")
        else:
            wav, _ = sf.read(str(TRAIN_DIR / rel), dtype="float32")
            wav = np.tile(wav, int(np.ceil(self.crop / max(1, len(wav)))))[: self.crop]
        return torch.from_numpy(wav), self.spk2id[spk], i


class EndlessSampler(torch.utils.data.Sampler):
    def __init__(self, items, spk_balanced: bool, seed=0):
        self.items, self.spk_balanced, self.seed = items, spk_balanced, seed
        by = {}
        for i, (s, _) in enumerate(items):
            by.setdefault(s, []).append(i)
        self.by = list(by.values())

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            if self.spk_balanced:
                for _ in range(len(self.items)):
                    yield rng.choice(rng.choice(self.by))
            else:
                idx = list(range(len(self.items)))
                rng.shuffle(idx)
                yield from idx


class HardPairSampler(EndlessSampler):
    """Yields crops in pairs (0,1), (2,3), ... (the pairs the augmenter's shared background uses). With prob p the
    second crop comes from one of the k speakers whose teacher centroid is closest to the first one's: similar-voice
    impostors, which the batch-similarity KD then has to keep apart (private's impostors are voice- and channel-matched)."""

    def __init__(self, items, neighbours, p, seed=0):
        super().__init__(items, False, seed)
        self.neighbours, self.p = neighbours, p
        self.spk_of = [0] * len(items)
        for k, idx in enumerate(self.by):
            for i in idx:
                self.spk_of[i] = k

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            idx = list(range(len(self.items)))
            rng.shuffle(idx)
            for i, j in zip(idx[0::2], idx[1::2]):
                yield i
                if rng.random() < self.p:
                    j = rng.choice(self.by[rng.choice(self.neighbours[self.spk_of[i]])])
                yield j


def speaker_neighbours(by, emb, k):
    """k nearest other speakers per speaker (cosine of mean teacher embeddings); `by` = list of item-index lists."""
    cent = torch.stack([emb[torch.as_tensor(ix, device=emb.device)].mean(0) for ix in by])
    cent = F.normalize(cent, dim=-1)
    sim = cent @ cent.T
    sim.fill_diagonal_(-2)
    return sim.topk(k, dim=1).indices.cpu().tolist()


# ----------------------------------------------------------------------------
@torch.inference_mode()
def quick_eval(model, dev, device):
    pairs, labels, files = dev
    pos = {f: i for i, f in enumerate(files)}
    model.eval()
    torch.backends.cudnn.benchmark = False  # variable lengths: autotuning per shape is very slow
    embs = []
    for f in files:
        wav, _ = sf.read(str(PUBLIC_DIR / f), dtype="float32")
        embs.append(embed_wav(model, torch.from_numpy(wav).to(device)).float().cpu().numpy())
    model.train()
    torch.backends.cudnn.benchmark = True
    emb = np.stack(embs)
    ia = np.array([pos[a] for a, _ in pairs])
    ib = np.array([pos[b] for _, b in pairs])
    return compute_eer(cosine_scores(emb, ia, ib), labels)


def make_dev(n_trials, seed=0):
    """Monitoring subset drawn from the P-dev half only (P-hold is never touched in training)."""
    from .protocol import _half, _spk
    pairs, labels = read_trials()
    pool = [i for i, (a, b) in enumerate(pairs) if _half(_spk(a)) == 0 and _half(_spk(b)) == 0]
    rng = np.random.default_rng(seed)
    sel = rng.choice(pool, n_trials, replace=False)
    pairs = [pairs[i] for i in sel]
    return pairs, labels[sel], unique_files(pairs)


def freeze_before(model, name):
    """Freeze spec front-end and backbone children that come before `name`."""
    if not name:
        return []
    frozen = []
    if hasattr(model, "spec"):  # WeSpeaker wraps a parameter-free fbank instead
        for p in model.spec.parameters():
            p.requires_grad_(False)
    for n, mod in model.backbone.named_children():
        if n == name:
            break
        for p in mod.parameters():
            p.requires_grad_(False)
        frozen.append(n)
    return frozen


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="redimnet_M_mix")
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--crop-sec", type=float, default=3.0, help="crop loaded from disk (= max length)")
    ap.add_argument("--min-crop-sec", type=float, default=None,
                    help="if set, each batch is cut to a random length in [min-crop-sec, crop-sec]")
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone lr")
    ap.add_argument("--head-lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--head-only-steps", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--subcenters", type=int, default=3)
    ap.add_argument("--loss", default="aam", choices=["aam", "sf2"], help="aam = sub-center AAM; sf2 = SphereFace2-C")
    ap.add_argument("--ada-h", type=float, default=0.0, help="sf2 only: AdaFace-style norm-adaptive margin strength (0 = off)")
    ap.add_argument("--tlogits", default=None, help="cached teacher class cosines (.npy [N_train_txt, n_cls], float16)")
    ap.add_argument("--dkd", type=float, default=0.0, help="weight of decoupled logit KD vs --tlogits")
    ap.add_argument("--dkd-gamma", type=float, default=2.0)
    ap.add_argument("--dkd-T", type=float, default=4.0)
    ap.add_argument("--dkd-scale", type=float, default=30.0, help="logit = scale * cosine (student and teacher)")
    ap.add_argument("--dkd-conf-k", type=int, default=0,
                    help="confusion-set size at the END of training (0 = all classes); curriculum from all -> k")
    ap.add_argument("--margin-min-frac", type=float, default=1.0,
                    help="duration-adaptive margin: margin * lerp(min_frac..1) as the crop goes min-crop-sec..crop-sec "
                         "(short/low-quality crops get a smaller margin, cf. AdaFace / Adaptive Margin Circle Loss)")
    ap.add_argument("--aug", type=float, default=1.0, help="scale all augmentation probs; 0 disables")
    for k, h in (("speed", "speed perturbation 0.9-1.1"), ("pitch", "pitch shift +-2 semitones"),
                 ("tempo", "tempo 0.85-1.15, pitch kept"), ("gain", "level -20..+10 dB"),
                 ("timemask", "zero 1-3 chunks"), ("freqmask", "1-2 band-stops"), ("clip", "hard clipping")):
        ap.add_argument(f"--aug-{k}", type=float, default=0.0, help=f"prob of {h} (default off)")
    ap.add_argument("--aug-noise-p", type=float, default=0.6, help="prob of additive noise (babble or coloured)")
    ap.add_argument("--aug-noise-snr", default="5,20", help="coloured-noise SNR range in dB, e.g. 0,15")
    ap.add_argument("--aug-babble-snr", default="13,20", help="babble SNR range in dB")
    ap.add_argument("--aug-noise-beta", default="0,2", help="1/f^beta exponent range; negative = hiss-like")
    ap.add_argument("--aug-matched", type=float, default=0.0, help="prob of matched noise (lists/matched_noise.npz)")
    ap.add_argument("--aug-matched-end", type=float, default=None, help="curriculum: linearly ramp the prob to this value")
    ap.add_argument("--aug-matched-shift", type=float, default=0.0, help="dB added to the sampled SNR (negative = harder)")
    ap.add_argument("--aug-matched-hiss", type=float, default=0.2, help="share of matched noise that is hiss")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model forward (speed only)")
    ap.add_argument("--aug-rumble", type=float, default=0.0, help="prob of low-frequency rumble (private's 'wave' backgrounds)")
    ap.add_argument("--rumble-version", type=int, default=1, help="1 = swelling low-passed noise; 2 = steady drone + harmonic hum")
    ap.add_argument("--speech-crop", type=float, default=0.0,
                    help="prob that a step uses short crops placed on the most energetic (speech-dense) stretch of each clip")
    ap.add_argument("--speech-crop-sec", default="0.75,1.5", help="length range of those crops, seconds")
    ap.add_argument("--aug-shared", type=float, default=0.0,
                    help="prob that a pair of crops (different speakers) gets the identical real background (lists/realnoise_bank.npz)")
    ap.add_argument("--hard-pairs", type=float, default=0.0,
                    help="prob that the 2nd crop of each pair is a similar-voice speaker (teacher-centroid neighbours)")
    ap.add_argument("--hard-pairs-k", type=int, default=10, help="neighbour pool size for --hard-pairs")
    ap.add_argument("--shared-eq", type=float, default=0.0,
                    help="prob that a shared-background pair also gets the identical mic colouration (EQ)")
    ap.add_argument("--aug-overlap", type=float, default=0.0,
                    help="prob of mixing a quieter second speaker (5-15 dB below) over 30-100%% of the crop")
    ap.add_argument("--rir-bank", default=None,
                    help="real/simulated RIRS_NOISES impulse responses (lists/rir_bank.npz, scripts/make_ext_banks.py)")
    ap.add_argument("--aug-rir-bank", type=float, default=1.0, help="share of reverb draws that use --rir-bank")
    ap.add_argument("--musan-bank", default=None,
                    help="MUSAN + RIRS_NOISES noise bank (lists/musan_bank.npz, scripts/make_ext_banks.py)")
    ap.add_argument("--aug-musan", type=float, default=0.25,
                    help="share of additive-noise draws that use --musan-bank instead of babble/coloured noise")
    ap.add_argument("--musan-cats", default="musan_noise,musan_music,rirs_pointsource,rirs_isotropic",
                    help="categories drawn uniformly (also: musan_speech)")
    ap.add_argument("--kd", type=float, default=0.0, help="weight of clean->aug consistency KD to frozen base")
    ap.add_argument("--teacher", default=None, help="registry name of an external frozen teacher (e.g. B6)")
    ap.add_argument("--rkd", type=float, default=0.0, help="weight of relational KD (batch similarity matrix)")
    ap.add_argument("--teacher-clean", type=int, default=1, help="teacher sees clean audio (1) or augmented (0)")
    ap.add_argument("--teacher-emb", default=None,
                    help="cached FULL-utterance teacher embeddings (.npy, rows = original train.txt order): "
                         "long->short KD, the student sees crops, the target is the whole clean utterance")
    ap.add_argument("--temb-kd", type=float, default=0.0, help="weight of 1-cos(proj(student), teacher_full)")
    ap.add_argument("--temb-rkd", type=float, default=0.0, help="weight of relational KD vs cached teacher")
    ap.add_argument("--proj-init", default=None,
                    help=".npy [teacher_dim, emb_dim] start for the KD projection (e.g. least squares from the base's to the teacher's train embeddings)")
    ap.add_argument("--head-init-emb", default=None,
                    help="init the AAM head from speaker centroids of these cached embeddings (.npy, train.txt order); "
                         "use the teacher's when the student shares its embedding space (e.g. pruned B6)")
    ap.add_argument("--spk-balanced", action="store_true")
    ap.add_argument("--freeze-bn", action="store_true")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--dev-trials", type=int, default=3000, help="0 = no monitoring eval")
    ap.add_argument("--save-every", type=int, default=0, help="keep step checkpoints every N steps")
    ap.add_argument("--avg-last", type=int, default=0,
                    help="write avg.pt = weight average of the last K step checkpoints (no selection on test)")
    ap.add_argument("--init-ckpt", default=None, help="continue from a fine-tuned checkpoint (LMFT)")
    ap.add_argument("--init-head", default=None, help="take the classification head from this checkpoint (e.g. last.pt)")
    ap.add_argument("--dn-channels", type=int, default=0, help="learned denoising front-end (sred/denoise.py); 0 = off")
    ap.add_argument("--dn-clean", type=float, default=1.0, help="weight of the enhanced-vs-clean log-mel L1")
    ap.add_argument("--dn-noise", type=float, default=0.5, help="weight of the noise-estimate L1")
    ap.add_argument("--dn-pre-steps", type=int, default=0,
                    help="first N steps train only the denoiser on the aux losses (backbone not run)")
    ap.add_argument("--dn-lr", type=float, default=1e-3)
    ap.add_argument("--expand-tcb", default="", help="stages whose TimeContextBlock1d gets a zero-gated pretrained copy, e.g. 1,2,3,4,5")
    ap.add_argument("--expand-2d", default="", help="stages whose last 2D block gets a zero-gated pretrained copy, e.g. 0,1,2")
    ap.add_argument("--expand-lr", type=float, default=1e-4, help="lr of the inserted blocks")
    ap.add_argument("--train-from", default=None,
                    help="freeze backbone modules before this one, e.g. stage3 (cheaper backward)")
    ap.add_argument("--train-list", default=None, help="alternative train list, e.g. lists/clean/train_clean.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/resume.pt if present")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    out = ROOT / "exp/ft" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    log = open(out / "log.txt", "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    items = read_train_list(Path(args.train_list)) if args.train_list else read_train_list()
    spks = sorted({s for s, _ in items})
    spk2id = {s: i for i, s in enumerate(spks)}
    lengths = train_lengths(items)

    model = load_model(args.base)
    init_state = None
    prune_idx = None
    if Path(args.base).exists():
        _b = torch.load(args.base, map_location="cpu", weights_only=False)
        prune_idx = _b.get("prune")
        del _b
    if args.init_ckpt:
        init_state = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(init_state["model"])
    model = model.to(device).train()
    P(f"base={args.base} params={count_params(model) / 1e6:.3f}M")

    # head init from pretrained speaker centroids (cohort embeddings cached by evaluate.py)
    coh_items = cohort_subset(items, per_spk=5)
    init = None
    ck = cache_key(args.base, "traincohort5", [r for _, r in coh_items], mode="full")
    if init_state is not None and "head" in init_state:
        init = None
    elif args.head_init_emb:
        all_files = [r for _, r in read_train_list()]
        pos = {r: i for i, r in enumerate(all_files)}
        emb = np.load(args.head_init_emb)[[pos[r] for _, r in items]]
        init = torch.from_numpy(speaker_means(emb, [s for s, _ in items]))
        P("head init from centroids of", args.head_init_emb)
    elif args.train_list:
        # cleaned list (merged IDs / dropped files): init from full-train embeddings of the base model
        all_files = [r for _, r in read_train_list()]
        ck_full = cache_key(args.base, "trainfull", all_files, mode="full")
        if ck_full.exists():
            pos = {r: i for i, r in enumerate(all_files)}
            emb = np.load(ck_full)[[pos[r] for _, r in items]]
            init = torch.from_numpy(speaker_means(emb, [s for s, _ in items]))
            P("head init from full-train centroids (cleaned labels)", ck_full)
    elif ck.exists():
        cents = speaker_means(np.load(ck), [s for s, _ in coh_items])
        init = torch.from_numpy(cents)
        P("head init from centroids", ck)
    model.eval()
    with torch.no_grad():
        emb_dim = model(torch.zeros(1, 32000, device=device)).shape[-1]
    model.train()
    if args.loss == "sf2":
        head = SphereFace2C(emb_dim, len(spks), margin=args.margin, scale=args.scale, init=init, ada_h=args.ada_h).to(device)
    else:
        head = SubCenterAAM(emb_dim, len(spks), k=args.subcenters, margin=args.margin, scale=args.scale,
                            init=init).to(device)
    if init_state is not None and "head" in init_state and init_state["head"]["weight"].shape == head.weight.shape:
        head.load_state_dict(init_state["head"])
        P("head loaded from init ckpt")
    if args.init_head:
        head.load_state_dict(torch.load(args.init_head, map_location="cpu", weights_only=False)["head"])
        P("head loaded from", args.init_head)

    ex_cfg = None
    ex_params = []
    if args.expand_tcb or args.expand_2d:
        ex_cfg = {"tcb": [int(x) for x in args.expand_tcb.split(",") if x],
                  "blocks2d": [int(x) for x in args.expand_2d.split(",") if x]}
        n0 = count_params(model)
        ex_params = [p for m in expand_model(model, ex_cfg) for p in m.parameters()]
        model.to(device)
        P(f"expanded {ex_cfg}: {n0 / 1e6:.3f}M -> {count_params(model) / 1e6:.3f}M")
    frozen = freeze_before(model, args.train_from)
    dn_cfg = None
    if args.dn_channels > 0:
        dn_cfg = {"channels": args.dn_channels}
        attach_denoiser(model, dn_cfg)
        model.spec.to(device).train()
        P(f"denoiser front-end: {sum(p.numel() for p in model.spec.parameters()) / 1e6:.3f}M params, "
          f"total {count_params(model) / 1e6:.3f}M")
    dn_params = list(model.spec.dn.parameters()) if dn_cfg else []
    dn_ids = {id(p) for p in dn_params} | {id(p) for p in ex_params}
    if frozen:
        P("frozen:", frozen, f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

    teacher = None
    if args.kd > 0 or args.rkd > 0:
        teacher = (load_model(args.teacher) if args.teacher else copy.deepcopy(model)).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        P("teacher:", args.teacher or "frozen copy of base")

    temb = proj = None
    if args.teacher_emb:
        all_rels = [r for _, r in read_train_list()]
        row = {r: i for i, r in enumerate(all_rels)}
        T = np.load(args.teacher_emb)
        assert len(T) == len(all_rels), (T.shape, len(all_rels))
        T = T / np.linalg.norm(T, axis=1, keepdims=True)
        temb = torch.from_numpy(T[[row[r] for _, r in items]]).float().to(device)  # aligned with `items`
        proj = nn.Linear(emb_dim, temb.shape[1], bias=False).to(device)
        if args.proj_init:  # a different embedding space: start from a fitted linear map
            with torch.no_grad():
                proj.weight.copy_(torch.from_numpy(np.load(args.proj_init)).float())
            P("KD projection initialised from", args.proj_init)
        elif emb_dim == temb.shape[1]:  # same space (pruned teacher): start from identity, not a random map
            with torch.no_grad():
                proj.weight.copy_(torch.eye(emb_dim))
        P(f"long->short KD from cached teacher {args.teacher_emb} dim={temb.shape[1]} "
          f"kd={args.temb_kd} rkd={args.temb_rkd}")

    tlog = None
    if args.tlogits and args.dkd > 0:
        all_rels = [r for _, r in read_train_list()]
        row = {r: i for i, r in enumerate(all_rels)}
        TL = np.load(args.tlogits, mmap_mode="r")
        assert TL.shape[0] == len(all_rels) and TL.shape[1] == len(spks), (TL.shape, len(all_rels), len(spks))
        tlog = torch.from_numpy(np.ascontiguousarray(TL[[row[r] for _, r in items]])).to(device)  # float16
        P(f"decoupled logit KD from {args.tlogits} w={args.dkd} gamma={args.dkd_gamma} T={args.dkd_T} conf_k={args.dkd_conf_k}")

    _rng = lambda v: tuple(float(x) for x in v.split(","))  # noqa: E731
    aug = Augmenter(p_noise=args.aug_noise_p * args.aug, snr_noise=_rng(args.aug_noise_snr),
                    snr_babble=_rng(args.aug_babble_snr), noise_beta=_rng(args.aug_noise_beta), p_reverb=0.3 * args.aug, p_band=0.2 * args.aug,
                    p_eq=0.3 * args.aug, p_mulaw=0.1 * args.aug,
                    p_speed=args.aug_speed, p_pitch=args.aug_pitch, p_tempo=args.aug_tempo, p_gain=args.aug_gain,
                    p_timemask=args.aug_timemask, p_freqmask=args.aug_freqmask, p_clip=args.aug_clip,
                    p_overlap=args.aug_overlap, p_matched=args.aug_matched,
                    matched=dict(np.load(ROOT / "lists/matched_noise.npz")) if args.aug_matched > 0 else None,
                    matched_snr_shift=args.aug_matched_shift, matched_hiss_frac=args.aug_matched_hiss,
                    p_shared=args.aug_shared, p_shared_eq=args.shared_eq, p_rumble=args.aug_rumble, rumble_version=args.rumble_version,
                    shared_bank=dict(np.load(ROOT / "lists/realnoise_bank.npz")) if args.aug_shared > 0 else None,
                    rir_bank=dict(np.load(args.rir_bank)) if args.rir_bank else None, p_rir_bank=args.aug_rir_bank,
                    musan_bank=dict(np.load(args.musan_bank)) if args.musan_bank else None, p_musan=args.aug_musan,
                    musan_cats=args.musan_cats.split(",")) \
        if args.aug > 0 else None

    ds = CropDataset(items, spk2id, lengths, args.crop_sec)
    sampler = EndlessSampler(items, args.spk_balanced, args.seed)
    if args.hard_pairs > 0:
        assert temb is not None and not args.spk_balanced and args.bs % 2 == 0, "--hard-pairs needs --teacher-emb"
        sampler = HardPairSampler(items, speaker_neighbours(sampler.by, temb, args.hard_pairs_k), args.hard_pairs, args.seed)
        P(f"hard pairs: p={args.hard_pairs}, {args.hard_pairs_k} teacher-centroid neighbours per speaker")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, sampler=sampler,
                                     num_workers=args.workers, drop_last=True, prefetch_factor=4, persistent_workers=True)
    fwd = torch.compile(model) if args.compile else model  # same weights; only the forward is compiled
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in dn_ids], "lr": args.lr},
        {"params": list(head.parameters()) + (list(proj.parameters()) if proj is not None else []),
         "lr": args.head_lr, "weight_decay": 0.0},
    ] + ([{"params": dn_params, "lr": args.dn_lr}] if dn_params else [])
      + ([{"params": ex_params, "lr": args.expand_lr}] if ex_params else []), weight_decay=args.wd)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_mult(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p))) * 0.95 + 0.05

    dev = make_dev(args.dev_trials) if args.dev_trials > 0 else None
    best = quick_eval(model, dev, device) if dev else float("inf")
    P(f"step 0 dev_eer={best:.3f}")
    saved = []
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler(enabled=args.amp == "fp16")

    start = 1
    if args.resume and (out / "resume.pt").exists():
        r = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"])
        head.load_state_dict(r["head"])
        if proj is not None and "proj" in r:
            proj.load_state_dict(r["proj"])
        elif proj is not None:
            P("WARNING: resume.pt has no KD projection; it restarts from random init")
        opt.load_state_dict(r["opt"])
        saved = [Path(x) for x in r["saved"]]
        start = r["step"] + 1
        torch.manual_seed(args.seed + start)
        random.seed(args.seed + start)
        dl.sampler.seed = args.seed + start
        P(f"resumed from step {r['step']}")
    t0 = time.time()
    it = iter(dl)
    run = {"loss": 0.0, "acc": 0.0, "kd": 0.0, "n": 0}
    for step in range(start, args.steps + 1):
        wav, lab, iidx = next(it)
        wav, lab = wav.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        if args.speech_crop > 0 and random.random() < args.speech_crop:
            # energy-based short crops (no VAD model): per clip, the window of the chosen length with the most
            # frame energy, jittered among the top windows, so short training crops hold speech, not pauses
            lo, hi = (float(v) for v in args.speech_crop_sec.split(","))
            n = int(round(random.uniform(lo, hi) * 4)) * 4000
            fe = wav.unfold(1, 400, 160).pow(2).mean(-1)  # [B, frames] 25 ms / 10 ms
            k = n // 160
            win = F.avg_pool1d(fe[:, None], k, stride=4)[:, 0]  # energy of each candidate window
            top = win.topk(min(4, win.shape[1]), dim=1).indices
            pick = top.gather(1, torch.randint(0, top.shape[1], (wav.shape[0], 1), device=wav.device))[:, 0] * 4 * 160
            pick = pick.clamp(max=wav.shape[1] - n)
            wav = wav.gather(1, pick[:, None] + torch.arange(n, device=wav.device)[None])  # one gather, no per-clip loop
        elif args.min_crop_sec:
            n = int(round(random.uniform(args.min_crop_sec, args.crop_sec) * 4)) * 4000  # 0.25 s grid (few cudnn shapes)
            off = random.randint(0, wav.shape[1] - n)
            wav = wav[:, off:off + n]
        if args.margin_min_frac < 1.0 and args.min_crop_sec:
            frac = (wav.shape[1] / 16000 - args.min_crop_sec) / max(1e-6, args.crop_sec - args.min_crop_sec)
            head.margin = args.margin * (args.margin_min_frac + (1 - args.margin_min_frac) * min(1.0, max(0.0, frac)))
        if aug is not None and args.aug_matched_end is not None:
            aug.p_matched = args.aug_matched + (args.aug_matched_end - args.aug_matched) * min(1.0, step / args.steps)
        clean = wav
        if aug is not None:
            wav = aug(wav)
        m = lr_mult(step)
        head_only = step <= args.head_only_steps
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * m
        opt.param_groups[0]["lr"] = 0.0 if head_only else opt.param_groups[0]["lr"]
        dn_pre = step <= args.dn_pre_steps
        if ex_params and (head_only or dn_pre):  # inserted blocks follow the backbone schedule
            opt.param_groups[-1]["lr"] = 0.0
        if dn_pre:  # stage 1: learn the noise / clean mapping only; backbone and head untouched
            opt.param_groups[0]["lr"] = opt.param_groups[1]["lr"] = 0.0
        if args.freeze_bn:
            model.apply(set_bn_eval)
        for name in frozen:
            getattr(model.backbone, name).eval()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            if dn_pre:
                model.spec(wav)
                emb = None
                loss, acc = torch.zeros((), device=device), torch.zeros(())
            else:
                emb = fwd(wav).float()
                loss, acc = head(emb, lab)
            kd = torch.zeros((), device=device)
            if dn_cfg:
                l_c, l_n = aux_losses(model.spec, clean, wav)
                loss = loss + args.dn_clean * l_c + args.dn_noise * l_n
                run["dn_c"] = run.get("dn_c", 0.0) + float(l_c)
                run["dn_n"] = run.get("dn_n", 0.0) + float(l_n)
            if teacher is not None and emb is not None:
                with torch.no_grad():
                    t_emb = teacher(clean if args.teacher_clean else wav).float()
                if args.kd > 0:  # direct embedding consistency (same embedding space only)
                    kd = (1 - F.cosine_similarity(emb, t_emb, dim=-1)).mean()
                    loss = loss + args.kd * kd
                if args.rkd > 0:  # relational KD: match batch cosine-similarity matrices
                    s_s = F.normalize(emb) @ F.normalize(emb).T
                    s_t = F.normalize(t_emb) @ F.normalize(t_emb).T
                    r = (s_s - s_t).pow(2).mean()
                    kd = kd + r
                    loss = loss + args.rkd * r
            if tlog is not None and emb is not None:
                prog = step / args.steps
                k_now = 0 if args.dkd_conf_k <= 0 else int(round(len(spks) - (len(spks) - args.dkd_conf_k) * min(1.0, prog / 0.7)))
                s_logit = args.dkd_scale * head.class_cos(emb)
                t_logit = args.dkd_scale * tlog[iidx.to(device)].float()
                k2 = dkd_loss(s_logit, t_logit, lab, gamma=args.dkd_gamma, T=args.dkd_T, conf_k=k_now)
                kd = kd + k2
                loss = loss + args.dkd * k2
            if temb is not None and emb is not None:
                t_full = temb[iidx.to(device)]
                if args.temb_kd > 0:
                    k1 = (1 - F.cosine_similarity(proj(emb), t_full, dim=-1)).mean()
                    kd = kd + k1
                    loss = loss + args.temb_kd * k1
                if args.temb_rkd > 0:
                    e_n = F.normalize(emb)
                    r = (e_n @ e_n.T - t_full @ t_full.T).pow(2).mean()
                    kd = kd + r
                    loss = loss + args.temb_rkd * r
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"] + list(head.parameters()) + dn_params + ex_params, 5.0)
        scaler.step(opt)
        scaler.update()
        run["loss"] += float(loss)
        run["acc"] += float(acc)
        run["kd"] += float(kd)
        run["n"] += 1
        if step % 100 == 0:
            n = run["n"]
            dn = f"dn_clean={run.get('dn_c', 0) / n:.4f} dn_noise={run.get('dn_n', 0) / n:.4f} " if dn_cfg else ""
            P(f"step {step} loss={run['loss'] / n:.3f} acc={run['acc'] / n:.3f} kd={run['kd'] / n:.4f} {dn}"
              f"lr={opt.param_groups[0]['lr']:.2e} {(time.time() - t0) / (step - start + 1):.2f}s/it")
            run = {"loss": 0.0, "acc": 0.0, "kd": 0.0, "n": 0}
        if args.save_every and step % args.save_every == 0:
            torch.save({"base_model": args.base, "model": model.state_dict(), "step": step, "denoiser": dn_cfg, "expand": ex_cfg},
                       out / f"step{step}.pt")
            saved.append(out / f"step{step}.pt")
            torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                        "proj": proj.state_dict() if proj is not None else None,
                        "step": step, "saved": [str(x) for x in saved]}, out / "resume.tmp")
            (out / "resume.tmp").replace(out / "resume.pt")
        if step % args.eval_every == 0 or step == args.steps:
            eer = quick_eval(model, dev, device) if dev else float("nan")
            state = {"base_model": args.base, "model": model.state_dict(), "head": head.state_dict(),
                     "step": step, "dev_eer": eer, "args": vars(args), "denoiser": dn_cfg, "expand": ex_cfg}
            torch.save(state, out / "last.pt")
            tag = ""
            if eer < best:
                best = eer
                torch.save(state, out / "best.pt")
                tag = " *best*"
            P(f"step {step} dev_eer={eer:.3f}{tag}")
    if not (out / "best.pt").exists():
        torch.save(state, out / "best.pt")
    if args.avg_last and saved:
        avg_checkpoints(saved[-args.avg_last:], out / "avg.pt", args.base, dn_cfg, ex_cfg)
        P("wrote avg.pt from", [p.name for p in saved[-args.avg_last:]])
    P(f"done best_dev_eer={best:.3f}")


def avg_checkpoints(paths, dst, base, dn_cfg=None, ex_cfg=None):
    sd = None
    for p in paths:
        m = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if sd is None:
            sd = {k: v.clone().double() if v.is_floating_point() else v.clone() for k, v in m.items()}
        else:
            for k, v in m.items():
                if v.is_floating_point():
                    sd[k] += v.double()
    ref = torch.load(paths[-1], map_location="cpu", weights_only=False)["model"]
    sd = {k: (v / len(paths)).to(ref[k].dtype) if v.is_floating_point() else v for k, v in sd.items()}
    torch.save({"base_model": base, "model": sd, "step": "avg" + "+".join(p.stem for p in paths), "denoiser": dn_cfg, "expand": ex_cfg}, dst)


if __name__ == "__main__":
    main()
