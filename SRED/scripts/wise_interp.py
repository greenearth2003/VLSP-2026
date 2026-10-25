"""WiSE-FT: linear interpolation between a fine-tuned checkpoint and its pretrained base.

  python scripts/wise_interp.py exp/ft/F03_M_s3_aug_bnfz_var/best.pt exp/ft/W01_wise_F03_M/avg.pt --alpha 0.5

theta = alpha * theta_ft + (1 - alpha) * theta_pretrained   (alpha=0.5 for W01)
Non-float buffers (e.g. BatchNorm num_batches_tracked) are taken from the fine-tuned checkpoint.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sred.models import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ft_ckpt")
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the fine-tuned model")
    args = ap.parse_args()

    ft = torch.load(args.ft_ckpt, map_location="cpu", weights_only=False)
    from sred.expand import base_state  # expanded checkpoints (sred/expand.py) need the base in the same layout
    base = base_state(ft["base_model"], ft.get("expand")) if ft.get("expand") else load_model(ft["base_model"]).state_dict()
    a = args.alpha
    sd = {k: (a * v.double() + (1 - a) * base[k].double()).to(v.dtype) if v.is_floating_point() else v
          for k, v in ft["model"].items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"base_model": ft["base_model"], "model": sd, "step": f"wise{a}({Path(args.ft_ckpt)})",
                "expand": ft.get("expand"), "denoiser": ft.get("denoiser")}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
