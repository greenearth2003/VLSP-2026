"""Check that this server's data side is identical to the reference server (crop ids, teacher caches and checkpoints
from different servers are only comparable when these match). Hashes cover array CONTENTS, not the .npz container,
so a rebuild that produces the same data passes.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/verify_artifacts.py            # check
    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/verify_artifacts.py --write    # reference server only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
EXPECTED = EXP / "prep/expected_hashes.json"

FILES = {  # name -> path; .npz = hash of every array (sorted by key), else hash of the file bytes
    "manifest.jsonl": EXP / "artifacts/manifest.jsonl",
    "crop_bank.npz": EXP / "artifacts/crop_bank.npz",
    "lists/rir_bank.npz": SRED / "lists/rir_bank.npz",
    "lists/musan_bank.npz": SRED / "lists/musan_bank.npz",
    "lists/realnoise_bank.npz": SRED / "lists/realnoise_bank.npz",
    "pretrained/redimnet2/b4-vox2-lm.pt": EXP / "pretrained/redimnet2/b4-vox2-lm.pt",
    "pretrained/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt": EXP / "pretrained/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt",
    "pretrained/wavlm/wavlm_large.pth": EXP / "pretrained/wavlm/wavlm_large.pth",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    if path.suffix == ".npz":
        with np.load(path) as z:
            for k in sorted(z.files):
                a = z[k]
                h.update(k.encode())
                h.update(str(a.dtype).encode() + str(a.shape).encode())
                h.update(np.ascontiguousarray(a).tobytes())
    else:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="record this server's hashes as the reference")
    a = ap.parse_args()
    got = {}
    for name, p in FILES.items():
        got[name] = digest(p) if p.exists() else None
    if a.write:
        EXPECTED.write_text(json.dumps(got, indent=2) + "\n")
        print("wrote", EXPECTED)
    exp = json.loads(EXPECTED.read_text())
    bad = 0
    for name, h in got.items():
        status = "MISSING" if h is None else ("ok" if h == exp.get(name) else "MISMATCH")
        bad += status != "ok"
        print(f"{status:8s} {name:48s} {h or '-'}  (expected {exp.get(name)})")
    print("all identical to the reference server" if not bad else f"{bad} item(s) differ: see HANDOFF.md")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
