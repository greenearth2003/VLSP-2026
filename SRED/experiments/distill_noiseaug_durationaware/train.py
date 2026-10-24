"""Training for E0-E3 (NEW_PLAN §0): pretrained ReDimNet2-B4, two-view batches (dnd/data.py), SphereFace2 (E0) or
DAME (E1+) objective; teacher KD terms are added for E2 / E3 (cached clean-crop teacher embeddings).

    cd SRED && source env.sh
    .venv/bin/python experiments/distill_noiseaug_durationaware/train.py --config experiments/distill_noiseaug_durationaware/configs/e0.json --name E0
    ... --resume            # continue runs/<name>/ from last.pt
    ... --max-steps 30      # smoke test (schedule still uses cfg steps)

Each view is run in chunks of at most `max_chunk_audio_s` audio-seconds (6 s x 32 crops does not fit in 20 GB);
losses are per sample, so accumulating chunk gradients is exact. BatchNorm statistics stay frozen (freeze_bn), so
chunking does not change the forward pass either.

runs/<name>/: config.json (resolved train + data config, git commit, input hashes), log.txt, step<k>.pt (model only,
sred.models.load_model-compatible), last.pt (model + heads + optimizer + step, for --resume).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path.insert(0, str(SRED))
sys.path.insert(0, str(EXP))

from dnd.data import build_loader, load_config  # noqa: E402
from dnd.heads import Objective  # noqa: E402
from prep.head_init import load_or_build  # noqa: E402
from sred.data import TRAIN_DIR  # noqa: E402
from sred.models import _load_redimnet2_pt  # noqa: E402

BASE_NAME = "redimnet2_b4_vox2_pt"  # registry name in sred.models: lets load_model(step<k>.pt) rebuild the student


def git_commit() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXP, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=EXP, capture_output=True,
                               text=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def param_groups(model, objective, cfg):
    o = cfg["optim"]
    emb_ids = {id(p) for m in (model.pool, model.bn, model.linear) for p in m.parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in emb_ids]
    embedding = [p for p in model.parameters() if id(p) in emb_ids]
    return [
        {"name": "backbone", "params": backbone, "lr": o["backbone_lr"], "weight_decay": o["weight_decay"]},
        {"name": "embedding", "params": embedding, "lr": o["embedding_lr"], "weight_decay": o["weight_decay"]},
        {"name": "heads", "params": list(objective.parameters()), "lr": o["head_lr"], "weight_decay": 0.0},
    ]


def lr_mult(step, cfg, warmup=True, offset=0):
    """Linear warm-up, then cosine decay to min_lr_frac over the steps after `offset`. Heads skip the warm-up
    (head_warmup: false); the student's schedule starts after the head-only phase (offset = head_only_steps)."""
    o = cfg["optim"]
    step, total = step - offset, cfg["steps"] - offset
    if warmup and step < o["warmup_steps"]:
        return (step + 1) / o["warmup_steps"]
    p = (step - o["warmup_steps"]) / max(1, total - o["warmup_steps"])
    return o["min_lr_frac"] + (1 - o["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[], help="override a train-config key, e.g. optim.head_lr=5e-4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    dcfg = load_config(EXP / cfg["data_config"], cfg.get("data_overrides", []))

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    out = EXP / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def P(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    loader, info = build_loader(dcfg, seed=a.seed, workers=a.workers)
    n_cls = len(info["speakers"])
    model = _load_redimnet2_pt(EXP / cfg["student_ckpt"]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 7_000_000, n_params
    model.eval()
    with torch.no_grad():
        emb_dim = model(torch.zeros(1, 32000, device=dev)).shape[-1]
    cent = torch.from_numpy(load_or_build(model, info["manifest"], info["spk2id"], TRAIN_DIR, info["manifest_sha256"]))
    model.train()
    objective = Objective(cfg["loss"], n_cls, emb_dim, cent if cfg["loss"].get("centroid_init", True) else None).to(dev)
    opt = torch.optim.AdamW(param_groups(model, objective, cfg))
    base_lrs = [g["lr"] for g in opt.param_groups]
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg["amp"])

    start = 1
    if a.resume and (out / "last.pt").exists():
        r = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"])
        objective.load_state_dict(r["objective"])
        opt.load_state_dict(r["opt"])
        start = r["step"] + 1
        loader.batch_sampler.seed = a.seed + start  # fresh crop stream after resume
        P(f"resumed from step {r['step']}")
    else:
        (out / "config.json").write_text(json.dumps({
            "train": cfg, "data": dcfg, "args": vars(a), "git": git_commit(), "params": n_params,
            "speakers": n_cls, "utterances": len(info["manifest"]), "manifest_sha256": info["manifest_sha256"],
            "crop_bank_sha_manifest": str(info["bank"]["manifest_sha256"])}, indent=2))
    P(f"{a.name}: B4 {n_params:,} params, {n_cls} speakers, objective {cfg['loss']['type']} heads "
      f"{objective.dims}, steps {cfg['steps']}, git {git_commit()}")

    H = cfg["optim"].get("head_only_steps", 0)
    S = dcfg["speakers_per_step"]
    max_audio = cfg["max_chunk_audio_s"]
    last = cfg["steps"] if not a.max_steps else min(cfg["steps"], a.max_steps)
    run = {"loss": 0.0, "n": 0, "t0": time.time(), "per": {}}
    it = iter(loader)
    for step in range(start, last + 1):
        p = (step - 1) / cfg["steps"]
        b = next(it)
        # head-only phase: student frozen (lr 0, run without autograd) while the new heads fit its embeddings
        head_only = step <= H
        m = 0.0 if head_only else lr_mult(step - 1, cfg, offset=H)
        m_heads = lr_mult(step - 1, cfg, warmup=False) if not cfg["optim"]["head_warmup"] else lr_mult(step - 1, cfg)
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0 * (m_heads if g["name"] == "heads" else m)
        if cfg["freeze_bn"]:
            model.apply(set_bn_eval)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for view in ("short", "long"):
            v = b[view]
            wav, lab = v["wav"].to(dev, non_blocking=True), v["label"].to(dev, non_blocking=True)
            w = objective.view_weight(view, p)
            sec = wav.shape[1] / 16000
            n_chunks = max(1, math.ceil(S * sec / max_audio))
            for idx in torch.arange(S).chunk(n_chunks):
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    with torch.set_grad_enabled(not head_only):
                        z = model(wav[idx]).float()
                    per_sample, stats = objective.view_loss(z, lab[idx], view, p)
                    loss = w * per_sample.sum() / S
                loss.backward()
                step_loss += float(loss)
                for d, (ls, hits) in stats.items():
                    acc = run["per"].setdefault((view, d), [0.0, 0.0, 0])
                    acc[0] += ls
                    acc[1] += hits
                    acc[2] += len(idx)
        torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], cfg["grad_clip"])
        opt.step()
        run["loss"] += step_loss
        run["n"] += 1

        if step % cfg["log_every"] == 0 or step == last:
            n = run["n"]
            heads = " ".join(f"{vw[0]}{d}:{ls / k:.2f}/{h / k:.2f}" for (vw, d), (ls, h, k) in sorted(run["per"].items()))
            P(f"step {step}{' (heads only)' if head_only else ''} loss={run['loss'] / n:.3f} [{heads}] alpha={objective.alpha(p):.2f} "
              f"margin_x={objective.margin_scale(p):.2f} lr={opt.param_groups[0]['lr']:.2e} "
              f"bins={b['bins'][0]:g}/{b['bins'][1]:g}s {(time.time() - run['t0']) / n:.2f}s/step "
              f"mem={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G")
            run = {"loss": 0.0, "n": 0, "t0": time.time(), "per": {}}
        if step % cfg["save_every"] == 0 or step == last:
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"base_model": BASE_NAME, "model": sd, "step": step, "run": a.name}, out / f"step{step}.pt")
            torch.save({"base_model": BASE_NAME, "model": sd, "objective": objective.state_dict(),
                        "opt": opt.state_dict(), "step": step, "config": cfg}, out / "last.tmp")
            (out / "last.tmp").replace(out / "last.pt")
    P("done", out)


if __name__ == "__main__":
    main()
