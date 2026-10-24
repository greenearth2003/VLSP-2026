"""Uniform weight average of train.py checkpoints of ONE run (a post-hoc EMA / SWA of the end of training).
BatchNorm statistics are frozen during training (freeze_bn), so they are identical across the checkpoints and
need no recalibration.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/average_ckpts.py OUT.pt step5000.pt step6000.pt ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def main(out, srcs):
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in srcs]
    bases = {s["base_model"] for s in states}
    assert len(bases) == 1, f"checkpoints come from different models: {bases}"
    avg = {}
    for k, v in states[0]["model"].items():
        if v.is_floating_point():
            avg[k] = sum(s["model"][k].double() for s in states).div(len(states)).to(v.dtype)
        else:
            assert all(torch.equal(s["model"][k], v) for s in states), f"non-float buffer {k} differs"
            avg[k] = v.clone()
    torch.save({"base_model": bases.pop(), "model": avg, "step": "avg(" + ",".join(Path(p).stem for p in srcs) + ")",
                "sources": [str(p) for p in srcs]}, out)
    print(f"averaged {len(srcs)} checkpoints -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
