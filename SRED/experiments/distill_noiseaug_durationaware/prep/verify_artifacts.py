"""Check that this server trains on exactly the reference server's data (crop ids, teacher caches and checkpoints from
different servers are only comparable when these match), and that all of it is present.

1. inputs: manifest, crop bank, noise banks, pretrained weights. Hashes cover array CONTENTS, not the .npz container,
   so a rebuild that produces the same data passes.
2. train audio: every one of the 57,010 manifest wavs must exist, decode to the manifest's sample count / rate, and
   match the reference sha256 (artifacts/train_audio.sha256); also counts speakers / hours and the raw train.txt.
3. disk: free space for teacher caches and run outputs.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/verify_artifacts.py            # check (~2-5 min)
    .venv/bin/python experiments/distill_noiseaug_durationaware/prep/verify_artifacts.py --write    # reference server only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
EXPECTED = EXP / "prep/expected_hashes.json"
AUDIO_REF = EXP / "artifacts/train_audio.sha256"
MIN_FREE_GB = 10  # teacher caches (~0.7 GB) + checkpoints of a few runs (~0.4 GB each) + headroom

FILES = {  # name -> path; .npz = hash of every array (sorted by key), else hash of the file bytes
    "manifest.jsonl": EXP / "artifacts/manifest.jsonl",
    "train_audio.sha256": AUDIO_REF,
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


def _audio(job):
    path, n_expected = job
    if not os.path.exists(path):
        return "missing", None
    info = sf.info(path)
    if info.samplerate != 16000 or info.frames != n_expected:
        return f"bad shape ({info.samplerate} Hz, {info.frames} samples, expected {n_expected})", None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return "ok", h.hexdigest()[:16]


def check_train_audio(write: bool) -> int:
    """Returns the number of problems."""
    sys.path.insert(0, str(SRED))
    from sred.data import TRAIN_DIR, TRAIN_LIST

    man = [json.loads(ln) for ln in open(EXP / "artifacts/manifest.jsonl")]
    with Pool(min(32, os.cpu_count() or 4)) as pool:
        res = pool.map(_audio, [(str(TRAIN_DIR / m["audio_path"]), m["num_samples"]) for m in man], chunksize=64)
    if write:
        AUDIO_REF.write_text("".join(f"{h}  {m['audio_path']}\n" for m, (_, h) in zip(man, res)))
    ref = dict(ln.split()[::-1] for ln in open(AUDIO_REF))
    bad = [(m["audio_path"], st if st != "ok" else "content differs")
           for m, (st, h) in zip(man, res) if st != "ok" or ref.get(m["audio_path"]) != h]
    raw = [ln.split() for ln in open(TRAIN_LIST) if ln.strip()]
    hours = sum(m["num_samples"] for m in man) / 16000 / 3600
    print(f"train audio ({TRAIN_DIR}): {len(man) - len(bad)}/{len(man)} wavs identical to the reference; "
          f"{len({m['speaker_id'] for m in man})} speakers, {hours:.1f} h in the cleaned list; "
          f"raw train.txt {len(raw)} lines / {len({s for s, _ in raw})} speakers (expected 58404 / 1900)")
    for rel, why in bad[:20]:
        print(f"  BAD {rel}: {why}")
    if len(bad) > 20:
        print(f"  ... {len(bad) - 20} more")
    problems = len(bad) + (len(raw) != 58404)
    free = shutil.disk_usage(EXP).free / 1e9
    print(f"free disk at {EXP}: {free:.0f} GB (need >= {MIN_FREE_GB} GB for teacher caches and run outputs)")
    return problems + (free < MIN_FREE_GB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="record this server's hashes as the reference")
    ap.add_argument("--skip-audio", action="store_true", help="skip the train-audio check (inputs only)")
    a = ap.parse_args()
    audio_problems = 0 if a.skip_audio else check_train_audio(a.write)
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
    bad += audio_problems
    print("all identical to the reference server" if not bad else f"{bad} problem(s): see HANDOFF.md")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
