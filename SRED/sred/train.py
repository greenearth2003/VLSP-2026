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

    def forward(self, emb, label):
        cos = F.linear(F.normalize(emb), F.normalize(self.weight)).view(-1, self.n_cls, self.k).amax(-1)
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


def sample_balanced_len() -> float:
    lo, hi = random.choices([(0.8, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 6.0)], weights=[30, 25, 25, 20])[0]
    return random.uniform(lo, hi)


def supcon_loss(emb, lab, temp=0.1):
    """Supervised contrastive loss (Khosla et al. 2020) over the batch."""
    z = F.normalize(emb, dim=-1)
    sim = z @ z.T / temp
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    pos = (lab[:, None] == lab[None, :]) & ~eye
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return torch.zeros((), device=z.device)
    return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


class PKSampler(torch.utils.data.Sampler):
    """Endless stream of P x K blocks: K utterances (with replacement if needed) of a random speaker."""

    def __init__(self, items, k, seed=0):
        self.k, self.seed = k, seed
        by = {}
        for i, (s, _) in enumerate(items):
            by.setdefault(s, []).append(i)
        self.by = [v for v in by.values() if len(v) >= 2]

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            v = rng.choice(self.by)
            yield from (rng.sample(v, self.k) if len(v) >= self.k else rng.choices(v, k=self.k))


# ----------------------------------------------------------------------------
def train_lengths(items, cache=ROOT / "lists/train_lengths.json"):
    """Frame counts per file; entries missing from the cache (e.g. external absolute paths) are added."""
    lens = json.loads(cache.read_text()) if cache.exists() else {}
    missing = [rel for _, rel in items if rel not in lens]
    if missing:
        for rel in missing:
            lens[rel] = sf.info(str(TRAIN_DIR / rel)).frames  # TRAIN_DIR / absolute path = absolute path
        cache.parent.mkdir(parents=True, exist_ok=True)
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
        return torch.from_numpy(wav), self.spk2id[spk]


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
    ap.add_argument("--aug", type=float, default=1.0, help="scale all augmentation probs; 0 disables")
    ap.add_argument("--kd", type=float, default=0.0, help="weight of clean->aug consistency KD to frozen base")
    ap.add_argument("--teacher", default=None, help="registry name of an external frozen teacher (e.g. B6)")
    ap.add_argument("--rkd", type=float, default=0.0, help="weight of relational KD (batch similarity matrix)")
    ap.add_argument("--teacher-clean", type=int, default=1, help="teacher sees clean audio (1) or augmented (0)")
    ap.add_argument("--spk-balanced", action="store_true")
    ap.add_argument("--freeze-bn", action="store_true")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--dev-trials", type=int, default=3000, help="0 = no monitoring eval")
    ap.add_argument("--save-every", type=int, default=0, help="keep step checkpoints every N steps")
    ap.add_argument("--avg-last", type=int, default=0,
                    help="write avg.pt = weight average of the last K step checkpoints (no selection on test)")
    ap.add_argument("--init-ckpt", default=None, help="continue from a fine-tuned checkpoint (LMFT)")
    ap.add_argument("--train-from", default=None,
                    help="freeze backbone modules before this one, e.g. stage3 (cheaper backward)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-list", default=None, help="alternative 'spk rel' list (e.g. cleaned)")
    ap.add_argument("--aug-profile", default="default", choices=["default", "matched"],
                    help="matched: no band-limit/mu-law, milder reverb, + pause insertion (EDA-driven)")
    ap.add_argument("--exclude-val", action="store_true", help="drop the v4 validation speakers (sred.valset)")
    ap.add_argument("--crop-dist", default="uniform", choices=["uniform", "balanced"],
                    help="balanced: per-batch length 0.8-1.5s 30%%, 1.5-2 25%%, 2-3 25%%, 3-6 20%% (needs --crop-sec 6)")
    ap.add_argument("--consistency", type=float, default=0.0,
                    help="lambda of short->long self-consistency 1-cos(e_short, sg(e_long)) (needs --crop-sec 6)")
    ap.add_argument("--cons-mu", type=float, default=0.5, help="weight of the AAM loss on the long view")
    ap.add_argument("--pk", type=int, default=0, help="P x K sampler: K utterances per speaker (0 = off)")
    ap.add_argument("--supcon", type=float, default=0.0, help="weight of supervised contrastive loss")
    ap.add_argument("--supcon-temp", type=float, default=0.1)
    ap.add_argument("--proto", type=float, default=0.0,
                    help="weight of prototypical loss: query (last of each P x K block, cropped 1-2 s) vs prototypes "
                         "of the other K-1 utterances of each speaker in the batch (needs --pk)")
    ap.add_argument("--dame", action="store_true",
                    help="Duration-Aware Matryoshka Embedding: per-duration views trained on nested embedding prefixes")
    ap.add_argument("--dame-durs", default="1,2,6", help="view durations (s); needs --crop-sec >= max")
    ap.add_argument("--dame-dims", default="48,96,192", help="embedding prefix used for each duration")
    ap.add_argument("--dame-margins", default="0.0,0.2,0.4", help="AAM margin for each duration")
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
    if args.exclude_val:
        from .valset import train_items_minus_val
        n0 = len(items)
        items = train_items_minus_val(items)
        P(f"exclude-val: {n0} -> {len(items)} utterances")
    spks = sorted({s for s, _ in items})
    spk2id = {s: i for i, s in enumerate(spks)}
    lengths = train_lengths(items)

    model = load_model(args.base)
    init_state = None
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
    else:
        if not ck.exists():  # e.g. cleaned list: slice the cached full-train embeddings of the base model
            full = read_train_list()
            ck_all = cache_key(args.base, "trainall", [r for _, r in full], mode="full")
            pos = {r: i for i, (_, r) in enumerate(full)}
            if ck_all.exists() and all(r in pos for _, r in coh_items):
                ck.parent.mkdir(parents=True, exist_ok=True)
                np.save(ck, np.load(ck_all)[[pos[r] for _, r in coh_items]])
            else:  # e.g. list with external data: extract base-model cohort embeddings once (cached)
                from .embed import extract_cached
                torch.backends.cudnn.benchmark = False  # variable lengths: autotuning per shape is very slow
                extract_cached(args.base, lambda: load_model(args.base), "traincohort5",
                               [r for _, r in coh_items], TRAIN_DIR, mode="full")
                torch.backends.cudnn.benchmark = True
    if init_state is None and ck.exists():
        cents = speaker_means(np.load(ck), [s for s, _ in coh_items])
        init = torch.from_numpy(cents)
        P("head init from centroids", ck)
    model.eval()
    with torch.no_grad():
        emb_dim = model(torch.zeros(1, 32000, device=device)).shape[-1]
    model.train()
    head = SubCenterAAM(emb_dim, len(spks), k=args.subcenters, margin=args.margin, scale=args.scale,
                        init=init).to(device)
    if init_state is not None and "head" in init_state and init_state["head"]["weight"].shape == head.weight.shape:
        head.load_state_dict(init_state["head"])
        P("head loaded from init ckpt")

    dame = None
    if args.dame:  # one sub-center AAM head per (duration, prefix-dim); training-only, not part of inference
        durs = [float(x) for x in args.dame_durs.split(",")]
        dims = [int(x) for x in args.dame_dims.split(",")]
        mars = [float(x) for x in args.dame_margins.split(",")]
        heads = nn.ModuleList([SubCenterAAM(d, len(spks), k=args.subcenters, margin=m, scale=args.scale,
                                            init=None if init is None else F.normalize(init[:, :d], dim=-1))
                               for d, m in zip(dims, mars)]).to(device)
        dame = list(zip(durs, dims, heads))
        head = heads[-1]  # full-dim head is the one saved / used for accuracy
        P("DAME views:", [(d, k, m) for d, k, m in zip(durs, dims, mars)])

    frozen = freeze_before(model, args.train_from)
    if frozen:
        P("frozen:", frozen, f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

    teacher = None
    if args.kd > 0 or args.rkd > 0:
        teacher = (load_model(args.teacher) if args.teacher else copy.deepcopy(model)).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        P("teacher:", args.teacher or "frozen copy of base")

    if args.aug_profile == "matched":
        aug = Augmenter(p_noise=0.6 * args.aug, p_reverb=0.15 * args.aug, p_band=0.0, p_eq=0.3 * args.aug,
                        p_mulaw=0.0, rt60_range=(0.2, 0.5), p_silence=0.3 * args.aug) if args.aug > 0 else None
    else:
        aug = Augmenter(p_noise=0.6 * args.aug, p_reverb=0.3 * args.aug, p_band=0.2 * args.aug,
                        p_eq=0.3 * args.aug, p_mulaw=0.1 * args.aug) if args.aug > 0 else None

    ds = CropDataset(items, spk2id, lengths, args.crop_sec)
    sampler = PKSampler(items, args.pk, args.seed) if args.pk else EndlessSampler(items, args.spk_balanced, args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, sampler=sampler,
                                     num_workers=args.workers, drop_last=True, prefetch_factor=2, persistent_workers=True)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": (list(nn.ModuleList([h for _, _, h in dame]).parameters()) if dame else list(head.parameters())),
         "lr": args.head_lr, "weight_decay": 0.0},
    ], weight_decay=args.wd)
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
        if dame and r.get("dame_heads"):
            for (_, _, h), sd in zip(dame, r["dame_heads"]):
                h.load_state_dict(sd)
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
        wav, lab = next(it)
        wav, lab = wav.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        wav_long = None
        if args.consistency > 0:  # long view 4-6 s + short view 0.8-2.0 s from the same recording window
            nL = min(int(round(random.uniform(4.0, 6.0) * 4)) * 4000, wav.shape[1])
            oL = random.randint(0, wav.shape[1] - nL)
            wav_long = wav[:, oL:oL + nL]
            nS = int(random.choice([0.8, 1.2, 1.6, 2.0]) * 16000)
            oS = random.randint(0, wav.shape[1] - nS)
            wav = wav[:, oS:oS + nS]
        elif args.crop_dist == "balanced":
            n = int(round(sample_balanced_len() * 4)) * 4000
            n = min(max(n, 12000), wav.shape[1])
            off = random.randint(0, wav.shape[1] - n)
            wav = wav[:, off:off + n]
        elif args.min_crop_sec:
            n = int(round(random.uniform(args.min_crop_sec, args.crop_sec) * 4)) * 4000  # 0.25 s grid (few cudnn shapes)
            off = random.randint(0, wav.shape[1] - n)
            wav = wav[:, off:off + n]
        clean = wav
        if aug is not None and not dame:
            wav = aug(wav)
            if wav_long is not None:
                wav_long = aug(wav_long)
        m = lr_mult(step)
        head_only = step <= args.head_only_steps
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * m
        opt.param_groups[0]["lr"] = 0.0 if head_only else opt.param_groups[0]["lr"]
        if args.freeze_bn:
            model.apply(set_bn_eval)
        for name in frozen:
            getattr(model.backbone, name).eval()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            kd = torch.zeros((), device=device)
            if args.proto > 0:  # supports: the first K-1 of each block; query: last one, short crop
                K = args.pk
                qi = torch.arange(K - 1, wav.shape[0], K, device=device)
                si = torch.tensor([i for i in range(wav.shape[0]) if i % K != K - 1], device=device)
                nq = min(int(random.choice([1.0, 1.25, 1.5, 1.75, 2.0]) * 16000), clean.shape[1])
                oq = random.randint(0, clean.shape[1] - nq)
                q = clean[qi, oq:oq + nq]
                q = aug(q) if aug is not None else q
                e_s, e_q = model(wav[si]).float(), model(q).float()
                l_s, acc = head(e_s, lab[si])
                l_q, _ = head(e_q, lab[qi])
                loss = l_s * (K - 1) / K + l_q / K
                protos = F.normalize(F.normalize(e_s, dim=-1).view(len(qi), K - 1, -1).mean(1), dim=-1)
                pl = F.cross_entropy(10.0 * F.normalize(e_q, dim=-1) @ protos.T, torch.arange(len(qi), device=device))
                loss = loss + args.proto * pl
                kd = kd + pl
                emb = None
            elif dame:  # independent random crop + augmentation per duration view; loss on nested prefixes
                loss = torch.zeros((), device=device)
                for dur, dim, h in dame:
                    n = min(int(dur * 16000), clean.shape[1])
                    o = random.randint(0, clean.shape[1] - n)
                    v = clean[:, o:o + n]
                    v = aug(v) if aug is not None else v
                    l_d, acc = h(model(v).float()[:, :dim], lab)
                    loss = loss + l_d / len(dame)
                emb = None
            else:
                emb = model(wav).float()
                loss, acc = head(emb, lab)
            if wav_long is not None:
                emb_l = model(wav_long).float()
                loss_l, _ = head(emb_l, lab)
                cons = (1 - F.cosine_similarity(emb, emb_l.detach(), dim=-1)).mean()
                loss = loss + args.cons_mu * loss_l + args.consistency * cons
                kd = kd + cons
            if args.supcon > 0:
                sc = supcon_loss(emb, lab, args.supcon_temp)
                loss = loss + args.supcon * sc
                kd = kd + sc
            if teacher is not None:
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
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"] + opt.param_groups[1]["params"], 5.0)
        scaler.step(opt)
        scaler.update()
        run["loss"] += float(loss)
        run["acc"] += float(acc)
        run["kd"] += float(kd)
        run["n"] += 1
        if step % 100 == 0:
            n = run["n"]
            P(f"step {step} loss={run['loss'] / n:.3f} acc={run['acc'] / n:.3f} kd={run['kd'] / n:.4f} "
              f"lr={opt.param_groups[0]['lr']:.2e} {(time.time() - t0) / (step - start + 1):.2f}s/it")
            run = {"loss": 0.0, "acc": 0.0, "kd": 0.0, "n": 0}
        if args.save_every and step % args.save_every == 0:
            torch.save({"base_model": args.base, "model": model.state_dict(), "step": step}, out / f"step{step}.pt")
            saved.append(out / f"step{step}.pt")
            torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                        "dame_heads": [h.state_dict() for _, _, h in dame] if dame else None,
                        "step": step, "saved": [str(x) for x in saved]}, out / "resume.tmp")
            (out / "resume.tmp").replace(out / "resume.pt")
        if step % args.eval_every == 0 or step == args.steps:
            eer = quick_eval(model, dev, device) if dev else float("nan")
            state = {"base_model": args.base, "model": model.state_dict(), "head": head.state_dict(),
                     "step": step, "dev_eer": eer, "args": vars(args)}
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
        avg_checkpoints(saved[-args.avg_last:], out / "avg.pt", args.base)
        P("wrote avg.pt from", [p.name for p in saved[-args.avg_last:]])
    P(f"done best_dev_eer={best:.3f}")


def avg_checkpoints(paths, dst, base):
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
    torch.save({"base_model": base, "model": sd, "step": "avg" + "+".join(p.stem for p in paths)}, dst)


if __name__ == "__main__":
    main()
