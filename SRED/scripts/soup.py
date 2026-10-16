"""Weight soup + WiSE-FT: average fine-tuned checkpoints of one run (or several seeds of the same base),
then interpolate with the pretrained base.

    python scripts/soup.py out.pt ckpt1.pt [ckpt2.pt ...] --alpha 0.5

theta = alpha * mean(theta_ckpts) + (1 - alpha) * theta_pretrained   (alpha=1: plain soup)
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sred.models import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    sts = [torch.load(c, map_location="cpu", weights_only=False) for c in a.ckpts]
    bases = {s["base_model"] for s in sts}
    assert len(bases) == 1, bases
    base = load_model(sts[0]["base_model"]).state_dict()
    sd = {}
    for k, v in sts[0]["model"].items():
        if v.is_floating_point():
            m = sum(s["model"][k].double() for s in sts) / len(sts)
            sd[k] = (a.alpha * m + (1 - a.alpha) * base[k].double()).to(v.dtype)
        else:
            sd[k] = v
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"base_model": sts[0]["base_model"], "model": sd, "step": f"soup{len(sts)}_a{a.alpha}"}, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
