"""Accuracy cost of low-precision deployment (simulated / fake quantization) for a checkpoint.

    python scripts/quant_eval.py --model exp/ft/W_K21_M/avg.pt --variants w8 w8a8 w4

w8    : weights INT8, symmetric per-output-channel, all Conv1d/Conv2d/Linear (activations float)
w4    : same with INT4 weights
w8a8  : w8 + activations INT8 at every Conv/Linear input (symmetric per-tensor, scale from calibration
        max over 256 train crops, fixed at test time -- static PTQ)
Each variant's quantized checkpoint is written next to the source and evaluated with sred.evaluate
(QMF back-end), so the numbers are directly comparable to the float model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.models import load_model  # noqa: E402
from sred.prune import load_calib_wavs  # noqa: E402
from sred.quant import QMODS, ActQuant, quant_weight, wrap_acts  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", nargs="+", default=["w8", "w8a8", "w4"])
    a = ap.parse_args()
    src = Path(a.model)
    state = torch.load(src, map_location="cpu", weights_only=False)
    for v in a.variants:
        model = load_model(str(src)).eval()
        bits = 4 if v.startswith("w4") else 8
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, QMODS):
                    m.weight.copy_(quant_weight(m.weight, bits))
        out = src.parent.parent / f"Q_{src.parent.name}_{v}" / "avg.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        if v.endswith("a8"):
            # static activation scales are model-structure dependent -> evaluate in-process via a wrapper ckpt
            wrap_acts(model)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(dev)
            with torch.no_grad():
                wavs = load_calib_wavs(256, sec=3.0, seed=1, list_path="lists/clean/train_clean.txt").to(dev)
                for i in range(0, len(wavs), 32):
                    model(wavs[i:i + 32])
            for m in model.modules():
                if isinstance(m, ActQuant):
                    m.calib = False
            torch.save({"kind": "fakequant_module", "module": model.cpu(), "base_model": state["base_model"]}, out)
        else:
            torch.save({"base_model": state["base_model"], "model": model.state_dict(), "step": f"{v}"}, out)
        print("wrote", out, flush=True)
        subprocess.run([sys.executable, "-m", "sred.evaluate", "--model", str(out), "--tag", f"quant_{src.parent.name}_{v}"],
                       check=True)


if __name__ == "__main__":
    main()
