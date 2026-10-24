"""Two-view data loader (NEW_PLAN §0, README): every step = S speakers x (short view, long view), crops drawn from the
fixed crop bank (artifacts/crop_bank.npz), noise added on the fly in the workers.

  TwoViewSampler   per step: one short bin and one long bin (uniform within each group); S distinct speakers drawn
                   uniformly among those that have crops in both bins (speaker-uniform sampling); per speaker one long
                   crop and one short crop, from DIFFERENT utterances when it has another one in that bin, else a
                   non-overlapping crop of the same file, else any. Yields [short crops of the S speakers, then long
                   crops of the same S speakers, same order]: consecutive crops (2k, 2k+1) inside a view are always
                   different speakers, which the shared-background augmentation needs.
  BankCropDataset  reads one bank crop (start, length) from the source file; audio is never modified on disk.
  TwoViewCollate   splits the batch into its two views, stacks each, runs NoiseAugmenter per view (each view has one
                   length) and returns per view: wav, label, crop (bank index = row of any cached teacher embedding),
                   row (manifest row), rir_cat / noise_cat / snr, and clean (with return_clean=True).

Entry point:

    from dnd.data import build_loader, load_config
    cfg = load_config("configs/base.json", ["aug=null"])
    loader, info = build_loader(cfg, seed=0, workers=8)       # endless
    b = next(iter(loader))   # b["short"]["wav"] [S, T_short], b["long"]["wav"] [S, T_long], b["bins"] (short_s, long_s)
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 16000
EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]


def load_config(path, overrides=()):
    """JSON config; overrides "a.b=value" (value parsed as JSON when possible)."""
    cfg = json.loads(Path(path).read_text())
    for o in overrides:
        key, val = o.split("=", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        d = cfg
        *head, last = key.split(".")
        for k in head:
            d = d[k]
        assert last in d, f"--set {key}: no such key"
        d[last] = val
    return cfg


def load_manifest(path):
    return [json.loads(ln) for ln in open(path)]


class TwoViewSampler(torch.utils.data.Sampler):
    def __init__(self, bank, manifest, spk2id, short_bins, long_bins, speakers_per_step, seed=0):
        self.S, self.seed = speakers_per_step, seed
        self.start_step = 1  # train.py sets this on resume: batch k depends only on (seed, k)
        self.short_bins, self.long_bins = list(short_bins), list(long_bins)
        self.start, self.length = bank["start"], bank["length"]
        self.row = bank["row"]
        spk_of_row = np.array([spk2id[m["speaker_id"]] for m in manifest])
        # index[bin][speaker] -> {manifest row: [crop ids]}
        self.index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for c, (r, b) in enumerate(zip(bank["row"], bank["bin_sec"])):
            self.index[float(b)][int(spk_of_row[r])][int(r)].append(c)
        for b in self.short_bins + self.long_bins:
            assert b in self.index, f"bin {b} s has no crops in the bank"
        self.eligible = {(s, l): sorted(set(self.index[s]) & set(self.index[l]))
                         for s in self.short_bins for l in self.long_bins}
        self.fallbacks = {"same_file_disjoint": 0, "same_file_overlap": 0, "pairs": 0}

    def _overlaps(self, a, b):
        return self.row[a] == self.row[b] and self.start[a] < self.start[b] + self.length[b] \
            and self.start[b] < self.start[a] + self.length[a]

    def __iter__(self):
        step = self.start_step
        while True:
            rng = random.Random(f"{self.seed}:{step}")  # per-step stream: resuming at step k reproduces batch k
            sb, lb = rng.choice(self.short_bins), rng.choice(self.long_bins)
            spks = rng.sample(self.eligible[(sb, lb)], self.S)
            short, long_ = [], []
            for s in spks:
                l_utts = self.index[lb][s]
                lu = rng.choice(list(l_utts))
                lc = rng.choice(l_utts[lu])
                s_utts = self.index[sb][s]
                other = [u for u in s_utts if u != lu]
                if other:
                    sc = rng.choice(s_utts[rng.choice(other)])
                else:
                    cands = [c for c in s_utts[lu] if not self._overlaps(c, lc)]
                    key = "same_file_disjoint" if cands else "same_file_overlap"
                    self.fallbacks[key] += 1
                    sc = rng.choice(cands or s_utts[lu])
                self.fallbacks["pairs"] += 1
                short.append((sc, 0, step))
                long_.append((lc, 1, step))
            yield short + long_
            step += 1

    def __len__(self):  # endless; only for progress bars
        return 10 ** 12


class BankCropDataset(torch.utils.data.Dataset):
    def __init__(self, bank, manifest, spk2id, root):
        self.bank, self.root = bank, root
        self.paths = [m["audio_path"] for m in manifest]
        self.labels = [spk2id[m["speaker_id"]] for m in manifest]

    def __len__(self):
        return len(self.bank["row"])

    def __getitem__(self, key):
        c, view, step = key
        r = int(self.bank["row"][c])
        wav, _ = sf.read(str(self.root / self.paths[r]), start=int(self.bank["start"][c]),
                         frames=int(self.bank["length"][c]), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
        return wav, self.labels[r], c, r, view, step


class TwoViewCollate:
    def __init__(self, augmenter=None, return_clean=False, seed=0):
        self.augmenter, self.return_clean, self.seed = augmenter, return_clean, seed

    def _view(self, items, rng):
        wav = np.stack([it[0] for it in items])
        out = {"label": torch.tensor([it[1] for it in items]), "crop": torch.tensor([it[2] for it in items]),
               "row": torch.tensor([it[3] for it in items])}
        if self.return_clean:
            out["clean"] = torch.from_numpy(wav.copy())
        if self.augmenter is not None:
            wav, meta = self.augmenter(wav, rng)
            out.update({k: torch.from_numpy(v) for k, v in meta.items()})
        out["wav"] = torch.from_numpy(wav)
        return out

    def __call__(self, batch):
        # augmentation randomness from (seed, step) only: independent of worker count and scheduling, so a resumed
        # run gets bit-identical inputs
        step = batch[0][5]
        assert all(b[5] == step for b in batch)
        rng = np.random.default_rng([self.seed, step])
        short = [b for b in batch if b[4] == 0]
        long_ = [b for b in batch if b[4] == 1]
        out = {"short": self._view(short, rng), "long": self._view(long_, rng)}
        out["bins"] = (out["short"]["wav"].shape[1] / SR, out["long"]["wav"].shape[1] / SR)
        out["step"] = step
        return out


def build_loader(cfg, seed=0, workers=8, return_clean=False):
    """Endless DataLoader from a config; cfg["aug"] = null gives clean crops. The bank, manifest and noise banks load
    here, before the workers fork, so the workers share them. Returns (loader, info)."""
    if str(SRED) not in sys.path:
        sys.path.insert(0, str(SRED))
    from sred.data import TRAIN_DIR
    from .noise import NoiseAugmenter

    manifest = load_manifest(EXP / cfg["manifest"])
    bank = dict(np.load(EXP / cfg["crop_bank"]))
    man_hash = (EXP / cfg["manifest"]).with_suffix(".sha256").read_text().split()[0]
    assert str(bank["manifest_sha256"]) == man_hash, "crop bank was built from another manifest: rebuild it"
    spks = sorted({m["speaker_id"] for m in manifest})
    spk2id = {s: i for i, s in enumerate(spks)}
    augmenter = None
    if cfg.get("aug"):
        b = cfg["banks"]
        augmenter = NoiseAugmenter(cfg["aug"], SRED / b["rir"], SRED / b["noise"], SRED / b["train_bg"])
    d = cfg["views"]
    sampler = TwoViewSampler(bank, manifest, spk2id, d["short_bins"], d["long_bins"], cfg["speakers_per_step"], seed)
    ds = BankCropDataset(bank, manifest, spk2id, TRAIN_DIR)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=TwoViewCollate(augmenter, return_clean, seed),
                                         num_workers=workers, prefetch_factor=4 if workers else None,
                                         persistent_workers=workers > 0, pin_memory=True)
    info = {"manifest": manifest, "bank": bank, "speakers": spks, "spk2id": spk2id, "augmenter": augmenter,
            "sampler": sampler, "manifest_sha256": man_hash}
    return loader, info
