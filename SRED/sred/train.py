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


# ----------------------------------------------------------------------------
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

    items = read_train_list()
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
    elif ck.exists():
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

    frozen = freeze_before(model, args.train_from)
    if frozen:
        P("frozen:", frozen, f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

    teacher = None
    if args.kd > 0 or args.rkd > 0:
        teacher = (load_model(args.teacher) if args.teacher else copy.deepcopy(model)).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        P("teacher:", args.teacher or "frozen copy of base")

    aug = Augmenter(p_noise=0.6 * args.aug, p_reverb=0.3 * args.aug, p_band=0.2 * args.aug,
                    p_eq=0.3 * args.aug, p_mulaw=0.1 * args.aug) if args.aug > 0 else None

    ds = CropDataset(items, spk2id, lengths, args.crop_sec)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, sampler=EndlessSampler(items, args.spk_balanced, args.seed),
                                     num_workers=args.workers, drop_last=True, prefetch_factor=4, persistent_workers=True)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": head.parameters(), "lr": args.head_lr, "weight_decay": 0.0},
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
        if args.min_crop_sec:
            n = int(round(random.uniform(args.min_crop_sec, args.crop_sec) * 4)) * 4000  # 0.25 s grid (few cudnn shapes)
            off = random.randint(0, wav.shape[1] - n)
            wav = wav[:, off:off + n]
        clean = wav
        if aug is not None:
            wav = aug(wav)
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
            emb = model(wav).float()
            loss, acc = head(emb, lab)
            kd = torch.zeros((), device=device)
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
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"] + list(head.parameters()), 5.0)
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
