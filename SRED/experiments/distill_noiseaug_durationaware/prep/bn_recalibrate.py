"""Re-estimate BatchNorm statistics of a trained model on our noisy two-view TRAIN crops, without changing any weight.

Training kept BN statistics frozen (freeze_bn), so a fine-tuned model still normalises with the pretrained model's
statistics (clean VoxCeleb-style audio). Here every BN layer gets a cumulative average (momentum=None) of its batch
statistics over `--steps` loader steps (both views, augmentation on), forward passes only.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/bn_recalibrate.py SRC.pt OUT.pt [--steps 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

from dnd.data import build_loader, load_config  # noqa: E402
from sred.models import load_model  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    st = torch.load(a.src, map_location="cpu", weights_only=False)
    model = load_model(a.src).cuda().eval()
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average over all batches
        m.train()
    loader, _ = build_loader(load_config(EXP / "configs/base.json"), seed=123, workers=8)
    it = iter(loader)
    for k in range(a.steps):
        b = next(it)
        for v in ("short", "long"):
            for chunk in b[v]["wav"].split(16):  # same 96-audio-s memory budget as training
                model(chunk.cuda())
        if k % 50 == 0:
            print(f"  step {k}/{a.steps}", flush=True)
    for m in bns:
        m.momentum = 0.1
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({**{k: v for k, v in st.items() if k != "model"}, "model": sd,
                "step": f"bnrecal{a.steps}({Path(a.src).name})"}, a.out)
    print(f"{len(bns)} BatchNorm layers re-estimated over {a.steps} steps -> {a.out}")


if __name__ == "__main__":
    main()
