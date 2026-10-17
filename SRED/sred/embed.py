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


def vad_trim(wav: torch.Tensor, pad_frames: int = 15, min_keep_sec: float = 1.0) -> torch.Tensor:
    """Energy VAD (same rule as scripts/data_features.py): keep 25 ms frames above
    max(p10 + min(12, (p95-p10)/2), p95 - 35) dB, dilated by `pad_frames` (10 ms hop) on both sides,
    and concatenate them. Falls back to the input if less than `min_keep_sec` would remain."""
    fl, hop = 400, 160
    if wav.shape[-1] < fl * 4:
        return wav
    fr = wav.unfold(0, fl, hop)
    db = 10 * torch.log10(fr.pow(2).mean(-1) + 1e-10)
    p10, p95 = torch.quantile(db, 0.10), torch.quantile(db, 0.95)
    keep = db > torch.maximum(p10 + torch.clamp((p95 - p10) / 2, max=12.0), p95 - 35)
    keep = F.max_pool1d(keep.float()[None, None], 2 * pad_frames + 1, 1, pad_frames)[0, 0] > 0
    mask = torch.zeros_like(wav, dtype=torch.bool)
    idx = keep.nonzero().squeeze(1)
    if len(idx) == 0:
        return wav
    starts = idx * hop
    for st in starts.tolist():  # frames overlap; mark hop-sized blocks + tail
        mask[st:st + fl] = True
    out = wav[mask]
    return out if out.shape[-1] >= int(min_keep_sec * 16000) else wav


@torch.inference_mode()
def dominant_speaker_wav(model, wav: torch.Tensor, win_sec: float = 1.0, min_total_sec: float = 3.0,
                         min_minor_frac: float = 0.2, split_cos: float = 0.5) -> torch.Tensor:
    """Keep only the dominant voice of a clip: VAD-trim, cut into `win_sec` windows, embed each, 2-means on
    cosine; if the two clusters are clearly different voices (centroid cosine < split_cos) and the minority is
    not tiny, return the concatenated windows of the larger cluster. Otherwise return the VAD-trimmed clip."""
    sr = 16000
    w = vad_trim(wav)
    win = int(win_sec * sr)
    n = w.shape[-1] // win
    if w.shape[-1] < int(min_total_sec * sr) or n < 3:
        return w
    segs = w[: n * win].view(n, win)
    e = F.normalize(model(segs), dim=-1)
    S = e @ e.T
    i, j = divmod(int(torch.argmin(S)), n)  # farthest pair as seeds
    cent = torch.stack([e[i], e[j]])
    for _ in range(10):
        a = (e @ cent.T).argmax(1)
        if a.min() == a.max():
            return w
        cent = F.normalize(torch.stack([e[a == k].mean(0) for k in (0, 1)]), dim=-1)
    a = (e @ cent.T).argmax(1)
    n0 = int((a == 0).sum())
    if min(n0, n - n0) < max(1, int(round(min_minor_frac * n))) or float(cent[0] @ cent[1]) >= split_cos:
        return w
    dom = 0 if n0 >= n - n0 else 1
    return segs[a == dom].reshape(-1)


@torch.inference_mode()
def loudest_speaker_wav(model, wav: torch.Tensor, win_sec: float = 1.0, hop_sec: float = 0.5, min_total_sec: float = 3.0,
                        split_cos: float = 0.45, min_db_gap: float = 6.0, min_minor_win: int = 2):
    """Keep only the LOUDER of two voices in a clip. VAD-trim, windows of win_sec (hop hop_sec), embed each window,
    2-means on cosine. Cut only if the two clusters are clearly different voices (centroid cosine < split_cos) AND
    one is clearly louder (median window RMS gap >= min_db_gap). Returns (wav, cut_flag)."""
    sr = 16000
    w = vad_trim(wav)
    win, hop = int(win_sec * sr), int(hop_sec * sr)
    if w.shape[-1] < int(min_total_sec * sr):
        return w, False
    segs = w.unfold(0, win, hop)  # [n, win]
    n = segs.shape[0]
    if n < 4:
        return w, False
    e = F.normalize(model(segs), dim=-1)
    db = 10 * torch.log10(segs.pow(2).mean(1) + 1e-10)
    i, j = divmod(int(torch.argmin(e @ e.T)), n)
    cent = torch.stack([e[i], e[j]])
    for _ in range(10):
        a = (e @ cent.T).argmax(1)
        if a.min() == a.max():
            return w, False
        cent = F.normalize(torch.stack([e[a == k].mean(0) for k in (0, 1)]), dim=-1)
    a = (e @ cent.T).argmax(1)
    n0 = int((a == 0).sum())
    if min(n0, n - n0) < min_minor_win or float(cent[0] @ cent[1]) >= split_cos:
        return w, False
    l0, l1 = float(db[a == 0].median()), float(db[a == 1].median())
    if abs(l0 - l1) < min_db_gap:
        return w, False
    keep = 0 if l0 > l1 else 1
    # rebuild audio from the kept windows (hop-sized hops, no duplicated samples)
    mask = torch.zeros(w.shape[-1], dtype=torch.bool, device=w.device)
    for k in torch.nonzero(a == keep).squeeze(1).tolist():
        mask[k * hop: k * hop + win] = True
    out = w[mask]
    return (out, True) if out.shape[-1] >= sr else (w, False)


@torch.inference_mode()
def embed_shift(model, wav: torch.Tensor, frac: float = 0.6, n_shift: int = 5, min_win_sec: float = 3.0,
                full_weight: float = 2.0, min_sec: float = 0.5, max_sec: float = 60.0) -> torch.Tensor:
    sr = 16000
    wav = wav[: int(max_sec * sr)]
    if wav.shape[-1] < int(min_sec * sr):
        reps = int(np.ceil(min_sec * sr / max(1, wav.shape[-1])))
        wav = wav.repeat(reps)[: int(min_sec * sr)]
    T = wav.shape[-1]
    e_full = F.normalize(model(wav[None]), dim=-1)[0]
    L = int(max(min_win_sec * sr, frac * T))
    if L >= T:
        return e_full
    starts = np.linspace(0, T - L, n_shift).astype(int)
    segs = torch.stack([wav[s:s + L] for s in starts])
    e_seg = F.normalize(model(segs), dim=-1)
    return F.normalize(full_weight * e_full + e_seg.sum(0), dim=-1)


@torch.inference_mode()
def embed_wav(model, wav: torch.Tensor, mode: str = "full", crop_sec: float = 3.0,
              hop_sec: float = 1.5, min_sec: float = 0.5, max_sec: float = 60.0) -> torch.Tensor:
    """wav: 1-D tensor on device. Returns an un-normalised embedding [D].

    mode:
      full       - one forward pass on whole utterance
      crop       - mean of L2-normalised embeddings of overlapping crops
      full+crop  - mean of normalised full embedding and normalised crop mean
      vad        - full, after energy-VAD silence removal (vad_trim)
      shift      - TTA with long shifted segments: full embedding (weight 2) + `n_shift` windows of length
                   max(min_win_sec, frac*T) evenly shifted across the utterance, all L2-normalised, averaged
      dom        - full, after dominant-speaker extraction
    """
    sr = 16000
    if mode == "loud":
        wav, mode = loudest_speaker_wav(model, wav)[0], "full"
    if mode == "raw":  # un-normalised full-utterance embedding (its norm is a quality measure)
        return model(wav[: int(max_sec * 16000)][None])[0]
    if mode == "shift":
        return embed_shift(model, wav, min_sec=min_sec, max_sec=max_sec)
    if mode == "vad":
        wav, mode = vad_trim(wav), "full"
    elif mode == "dom":
        wav, mode = dominant_speaker_wav(model, wav), "full"
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
