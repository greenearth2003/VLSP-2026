"""Per-sample noise / reverb / channel augmentation on CPU (numpy), run inside DataLoader workers.

Sources (all prebuilt by SRED/scripts/make_ext_banks.py and make_noise_bank.py):
  rir_bank.npz       RIRS_NOISES impulse responses, cats sim_small / sim_medium / sim_large / real
  musan_bank.npz     MUSAN noise / music / speech + RIRS_NOISES point-source / isotropic noise
  realnoise_bank.npz quiet frames of train recordings ("train_bg"): the only source spectrally close to private

Order per crop: reverb (source in a room) -> additive noise (one category) -> channel (band-limit, mic EQ, mu-law).
Pairs (2k, 2k+1) are different speakers; with p_shared a pair gets the SAME train_bg segment (each at its own SNR)
instead of individual noise, so "same background" stops meaning "same speaker" (the X2 / C2X trick).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

SR = 16000
NONE = -1


def fill_track(audio, off, keys, T, rng, fade=800):
    """T samples from several different bank segments (indices in `keys`), random starts, 50 ms crossfades, no looping."""
    out = np.zeros(T + fade, np.float32)
    pos, ramp = 0, np.linspace(0, 1, fade, dtype=np.float32)
    while pos < T:
        k = int(keys[rng.integers(len(keys))])
        seg = audio[off[k]:off[k + 1]].astype(np.float32)
        seg = seg[int(rng.integers(max(1, len(seg) // 2))):]
        seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-8)
        L = min(len(seg), T + fade - pos)
        piece = seg[:L].copy()
        if pos > 0 and L > fade:
            piece[:fade] *= ramp
            out[pos:pos + fade] *= ramp[::-1]
        out[pos:pos + L] += piece
        pos += max(L - fade, 1)
    return out[:T]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-8)


def mix_at_snr(x, n, snr_db):
    return x + n * (rms(x) / (rms(n) * 10 ** (snr_db / 20)))


class NoiseAugmenter:
    """cfg = the "aug" block of the experiment config. Banks are loaded once in the main process (fork shares them)."""

    def __init__(self, cfg: dict, rir_bank: str, noise_bank: str, train_bg_bank: str):
        self.cfg = cfg
        rb = np.load(rir_bank)
        self.rir, rir_cat = rb["rir"], rb["cat"]
        self.rir_cats = [c for c, w in cfg["rir_cats"].items() if w > 0]
        self.rir_w = np.array([cfg["rir_cats"][c] for c in self.rir_cats], float)
        self.rir_w /= self.rir_w.sum()
        self.rir_keys = [np.flatnonzero(rir_cat == c) for c in self.rir_cats]

        nb, bg = dict(np.load(noise_bank)), dict(np.load(train_bg_bank))  # npz members load on access: read once
        self.src = {}  # cat -> (audio, offsets, segment keys); all MUSAN cats share one audio array
        for c in sorted(set(nb["cat"].tolist())):
            self.src[c] = (nb["audio"], nb["offsets"], np.flatnonzero(nb["cat"] == c))
        self.src["train_bg"] = (bg["audio"], bg["offsets"], np.arange(len(bg["offsets"]) - 1))
        self.noise_cats = [c for c, w in cfg["noise_cats"].items() if w > 0]
        missing = [c for c in self.noise_cats if c not in self.src]
        assert not missing, f"unknown noise categories {missing}; have {sorted(self.src)}"
        self.noise_w = np.array([cfg["noise_cats"][c] for c in self.noise_cats], float)
        self.noise_w /= self.noise_w.sum()
        # stable ids for logging: rir cats 0.., noise cats 0.. (+ "shared" = len(noise_cats))
        self.SHARED = len(self.noise_cats)

    # -- sources --------------------------------------------------------------------------------------------------
    def noise_track(self, cat, T, rng):
        audio, off, keys = self.src[cat]
        if cat == "musan_speech":  # babble of 3-7 talkers
            lo, hi = self.cfg.get("babble_talkers", [3, 7])
            return sum(fill_track(audio, off, keys, T, rng) for _ in range(int(rng.integers(lo, hi + 1))))
        return fill_track(audio, off, keys, T, rng)

    def reverb(self, x, rng):
        k = int(rng.choice(len(self.rir_cats), p=self.rir_w))
        h = self.rir[rng.choice(self.rir_keys[k])].astype(np.float32)
        y = fftconvolve(x, h)[: len(x)].astype(np.float32)
        return y * (rms(x) / rms(y)), k

    # -- channel ---------------------------------------------------------------------------------------------------
    @staticmethod
    def band_limit(x, rng):
        f = np.fft.rfftfreq(len(x), 1 / SR)
        lo, hi = rng.uniform(100, 400), rng.uniform(3000, 4000)
        resp = 1 / np.sqrt(1 + (f / hi) ** 8) / np.sqrt(1 + (lo / np.maximum(f, 1)) ** 4)
        return np.fft.irfft(np.fft.rfft(x) * resp, n=len(x)).astype(np.float32)

    @staticmethod
    def mic_eq(x, rng):
        logf = np.log(np.maximum(np.fft.rfftfreq(len(x), 1 / SR), 20))
        g = np.zeros_like(logf)
        for _ in range(2):  # two peaking bands +-8 dB, plus a tilt
            fc, bw, gain = rng.uniform(np.log(150), np.log(6000)), rng.uniform(0.3, 1.2), rng.uniform(-8, 8)
            g += gain * np.exp(-0.5 * ((logf - fc) / bw) ** 2)
        g += rng.uniform(-3, 3) * (logf - np.log(1000)) / np.log(2)
        return np.fft.irfft(np.fft.rfft(x) * 10 ** (g / 20), n=len(x)).astype(np.float32)

    @staticmethod
    def mu_law(x, mu=255):
        peak = max(float(np.abs(x).max()), 1e-6)
        y = x / peak
        q = np.round((np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu) + 1) / 2 * mu)
        y = (q / mu) * 2 - 1
        return (np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu * peak).astype(np.float32)

    # -- batch -----------------------------------------------------------------------------------------------------
    def __call__(self, wavs: np.ndarray, rng: np.random.Generator):
        """wavs [B, T] float32 clean crops, pairs (2k, 2k+1) = different speakers. Returns (augmented [B, T],
        meta dict of int/float arrays: rir_cat, noise_cat, snr). Category id -1 = not applied."""
        c = self.cfg
        B, T = wavs.shape
        out = wavs.copy()
        meta = {"rir_cat": np.full(B, NONE), "noise_cat": np.full(B, NONE), "snr": np.full(B, np.nan, np.float32)}

        for i in range(B):
            if rng.random() < c["p_reverb"]:
                out[i], meta["rir_cat"][i] = self.reverb(out[i], rng)

        shared = np.zeros(B, bool)
        if c["p_shared"] > 0:
            for i in range(0, B - 1, 2):
                if rng.random() < c["p_shared"]:
                    n = fill_track(*self.src["train_bg"], T, rng)
                    for j in (i, i + 1):
                        snr = rng.uniform(*c["snr_shared"])
                        out[j] = mix_at_snr(out[j], n, snr)
                        meta["noise_cat"][j], meta["snr"][j], shared[j] = self.SHARED, snr, True

        for i in range(B):
            if (shared[i] and c.get("shared_exclusive", True)) or rng.random() >= c["p_noise"]:
                continue
            k = int(rng.choice(len(self.noise_cats), p=self.noise_w))
            cat = self.noise_cats[k]
            snr = rng.uniform(*c["snr"][cat])
            out[i] = mix_at_snr(out[i], self.noise_track(cat, T, rng), snr)
            meta["noise_cat"][i], meta["snr"][i] = k, snr

        ch = c.get("channel", {})
        for i in range(B):
            if rng.random() < ch.get("p_band", 0):
                out[i] = self.band_limit(out[i], rng)
            if rng.random() < ch.get("p_eq", 0):
                out[i] = self.mic_eq(out[i], rng)
            if rng.random() < ch.get("p_mulaw", 0):
                out[i] = self.mu_law(out[i])
        return out, meta
