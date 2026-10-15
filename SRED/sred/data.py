from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DATA_ROOT = Path(os.environ.get("SRED_DATA", "/home/jovyan/data/VLSP2026/Vi-SRED26"))
TRAIN_DIR = DATA_ROOT / "train/data"
TRAIN_LIST = DATA_ROOT / "train/train.txt"
PUBLIC_DIR = DATA_ROOT / "public_test/data"
PUBLIC_TRIALS = DATA_ROOT / "public_test/public-test.txt"


def load_wav(path: str | Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == 16000, f"{path}: sr={sr}"
    return wav


def read_train_list(path: Path = TRAIN_LIST) -> list[tuple[str, str]]:
    """Returns [(speaker, relpath)]."""
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            spk, rel = line.split()
            out.append((spk, rel))
    return out


def read_trials(path: Path = PUBLIC_TRIALS) -> tuple[list[tuple[str, str]], np.ndarray | None]:
    """Returns pairs and labels (1 target / 0 nontarget, or None if unlabeled)."""
    pairs, labels = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        pairs.append((parts[0], parts[1]))
        if len(parts) > 2:
            labels.append(1 if parts[2] in ("target", "1") else 0)
    return pairs, (np.array(labels) if labels else None)


def unique_files(pairs) -> list[str]:
    seen = {}
    for a, b in pairs:
        seen.setdefault(a, None)
        seen.setdefault(b, None)
    return list(seen)


def cohort_subset(items: list[tuple[str, str]], per_spk: int = 5, seed: int = 0) -> list[tuple[str, str]]:
    by_spk = defaultdict(list)
    for spk, rel in items:
        by_spk[spk].append(rel)
    rng = random.Random(seed)
    out = []
    for spk in sorted(by_spk):
        rels = sorted(by_spk[spk])
        rng.shuffle(rels)
        out += [(spk, r) for r in rels[:per_spk]]
    return out


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, files: list[str], root: Path, vad: bool = False):
        self.files, self.root, self.vad = files, Path(root), vad

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        wav = load_wav(self.root / self.files[i])
        if self.vad:
            from .vad import vad_trim
            wav = np.ascontiguousarray(vad_trim(wav))
        return i, torch.from_numpy(wav)
