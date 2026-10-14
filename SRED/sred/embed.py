"""Embedding extraction with on-disk caching."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import WavDataset

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "exp/emb"


def _segments(wav: torch.Tensor, crop: int, hop: int) -> list[torch.Tensor]:
    n = wav.shape[-1]
    if n <= crop:
        return [wav]
    starts = list(range(0, n - crop + 1, hop))
    if starts[-1] + crop < n:
        starts.append(n - crop)
    return [wav[s:s + crop] for s in starts]


@torch.inference_mode()
def embed_wav(model, wav: torch.Tensor, mode: str = "full", crop_sec: float = 3.0,
              hop_sec: float = 1.5, min_sec: float = 0.5, max_sec: float = 60.0) -> torch.Tensor:
    """wav: 1-D tensor on device. Returns an un-normalised embedding [D].

    mode:
      full       - one forward pass on whole utterance
      crop       - mean of L2-normalised embeddings of overlapping crops
      full+crop  - mean of normalised full embedding and normalised crop mean
    """
    sr = 16000
    wav = wav[: int(max_sec * sr)]
    if wav.shape[-1] < int(min_sec * sr):  # repeat-pad very short audio
        reps = int(np.ceil(min_sec * sr / max(1, wav.shape[-1])))
        wav = wav.repeat(reps)[: int(min_sec * sr)]
    outs = []
    if mode in ("full", "full+crop"):
        outs.append(F.normalize(model(wav[None]), dim=-1)[0])
    if mode in ("crop", "full+crop"):
        segs = _segments(wav, int(crop_sec * sr), int(hop_sec * sr))
        e = F.normalize(model(torch.stack(segs)) if len(segs) > 1 else model(segs[0][None]), dim=-1)
        outs.append(F.normalize(e.mean(0), dim=-1))
    return torch.stack(outs).mean(0)


def _length_batches(lengths: np.ndarray, tol: float = 0.03, max_samples: int = 16000 * 240,
                    max_bs: int = 64) -> list[list[int]]:
    """Group indices so each batch's shortest item is within `tol` of its longest."""
    order = np.argsort(lengths)
    batches, cur = [], []
    for i in order:
        if cur and (lengths[i] * (1 - tol) > lengths[cur[0]] or
                    lengths[i] * (len(cur) + 1) > max_samples or len(cur) >= max_bs):
            batches.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        batches.append(cur)
    return batches


class _BatchDataset(torch.utils.data.Dataset):
    def __init__(self, files, root, batches, max_len):
        self.ds, self.batches, self.max_len = WavDataset(files, root), batches, max_len

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, j):
        wavs = [self.ds[i][1][: self.max_len] for i in self.batches[j]]
        n = min(w.shape[-1] for w in wavs)
        return self.batches[j], torch.stack([w[:n] for w in wavs])


@torch.inference_mode()
def extract_batched(model, files: list[str], root: Path, device: str = "cuda", num_workers: int = 6,
                    max_sec: float = 60.0, min_sec: float = 0.5, tol: float = 0.03) -> np.ndarray:
    """Full-utterance embeddings, batching utterances of near-equal length (tail-trim <= tol)."""
    import soundfile as sf

    model = model.to(device).eval()
    max_len = int(max_sec * 16000)
    lengths = np.array([min(sf.info(str(Path(root) / f)).frames, max_len) for f in files])
    batches = _length_batches(lengths, tol=tol)
    dl = torch.utils.data.DataLoader(_BatchDataset(files, root, batches, max_len), batch_size=None,
                                     num_workers=num_workers, prefetch_factor=4)
    embs = np.zeros((len(files), 0), dtype=np.float32)
    for idx, wav in tqdm(dl, total=len(batches), mininterval=30, desc="embed-b"):
        wav = wav.to(device)
        if wav.shape[-1] < int(min_sec * 16000):
            reps = int(np.ceil(min_sec * 16000 / wav.shape[-1]))
            wav = wav.repeat(1, reps)[:, : int(min_sec * 16000)]
        e = model(wav).float().cpu().numpy()
        if embs.shape[1] == 0:
            embs = np.zeros((len(files), e.shape[1]), dtype=np.float32)
        embs[idx] = e
    return embs


def extract(model, files: list[str], root: Path, device: str = "cuda", num_workers: int = 6,
            **kw) -> np.ndarray:
    if kw.pop("batched", False):
        return extract_batched(model, files, root, device, num_workers)
    model = model.to(device).eval()
    ds = WavDataset(files, root)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers, prefetch_factor=8)
    embs = [None] * len(files)
    for i, wav in tqdm(dl, total=len(ds), mininterval=30, desc="embed"):
        embs[i] = embed_wav(model, wav.to(device), **kw).float().cpu().numpy()
    return np.stack(embs)


def cache_key(model_name: str, split: str, files: list[str], **kw) -> Path:
    h = hashlib.md5(("\n".join(files) + repr(sorted(kw.items()))).encode()).hexdigest()[:8]
    tag = "_".join(f"{k}{v}" for k, v in sorted(kw.items())) or "default"
    # checkpoints live at exp/ft/<run>/best.pt -> key by "<run>_best"
    safe = f"{Path(model_name).parent.name}_{Path(model_name).stem}" if "/" in model_name else model_name
    return EMB_DIR / safe / f"{split}_{tag}_{h}.npy"


def extract_cached(model_name: str, model_fn, split: str, files: list[str], root: Path, **kw) -> np.ndarray:
    path = cache_key(model_name, split, files, **kw)
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = extract(model_fn(), files, root, **kw)
    np.save(path, emb)
    return emb
