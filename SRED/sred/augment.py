"""On-GPU waveform augmentation for robustness to noise / channel / edge mics / voice variation.

No external corpora needed: babble comes from other utterances in the batch,
noise is synthetic coloured noise, reverberation uses synthetic exponential-decay
RIRs, and channel effects are simulated with band-limiting, random EQ and mu-law.
If MUSAN / RIR corpora are available, pass their waveforms via `noise_bank`.

Voice / prosody augmentations (label-preserving, mild ranges; at most one per sample):
  speed  -- resample by r (pitch and duration change together, Kaldi-style speed perturbation)
  pitch  -- shift pitch by +-n semitones, duration unchanged (phase-vocoder stretch + resample)
  tempo  -- change speaking rate by r, pitch unchanged (phase vocoder)
Other: gain (level), time masking (zeroed chunks), frequency masking (band-stop), clipping.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF  # mu-law

SR = 16000
# SNR ranges (dB) per external-noise category; musan_speech is a 3-7 speaker babble (Kaldi / WeSpeaker recipe)
MUSAN_SNR = {"musan_noise": (0, 15), "musan_music": (5, 15), "musan_speech": (13, 20),
             "rirs_pointsource": (0, 15), "rirs_isotropic": (0, 15)}


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean(-1, keepdim=True).clamp_min(1e-10).sqrt()


def _mix_at_snr(x, n, snr_db):
    scale = _rms(x) / (_rms(n) * (10 ** (snr_db / 20)))
    return x + n * scale


def coloured_noise(shape, device, beta: float) -> torch.Tensor:
    """1/f^beta noise: beta=0 white, 1 pink, 2 brown."""
    n = shape[-1]
    spec = torch.randn(*shape[:-1], n // 2 + 1, dtype=torch.cfloat, device=device)
    f = torch.arange(n // 2 + 1, device=device).clamp_min(1).float()
    spec = spec / f.pow(beta / 2)
    return torch.fft.irfft(spec, n=n)


def synth_rir(batch: int, device, rt60_range=(0.2, 0.9), length_s: float = 0.6) -> torch.Tensor:
    L = int(length_s * SR)
    t = torch.arange(L, device=device).float() / SR
    rt60 = torch.empty(batch, 1, device=device).uniform_(*rt60_range)
    decay = torch.exp(-6.9078 * t[None] / rt60)  # 60 dB decay
    rir = torch.randn(batch, L, device=device) * decay
    rir[:, 0] = 1.0  # direct path
    pre = int(0.0025 * SR)
    rir[:, 1:pre] = 0
    return rir / rir.norm(dim=-1, keepdim=True)


def fft_conv(x, h):
    n = x.shape[-1] + h.shape[-1] - 1
    nfft = 1 << (n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(h, nfft), nfft)
    return y[..., : x.shape[-1]]


def _fit_length(y: torch.Tensor, T: int) -> torch.Tensor:
    """Crop or wrap-pad [B, T'] to [B, T]."""
    if y.shape[-1] >= T:
        return y[..., :T]
    reps = math.ceil(T / y.shape[-1])
    return y.repeat(1, reps)[..., :T]


def resample_rate(x: torch.Tensor, r: float) -> torch.Tensor:
    """Play back r times faster (r>1: shorter + higher pitch). Linear interpolation (fine for |r-1|<~0.2)."""
    n = max(2, int(round(x.shape[-1] / r)))
    return F.interpolate(x[:, None], size=n, mode="linear", align_corners=False)[:, 0]


def time_stretch(x: torch.Tensor, r: float, n_fft: int = 512, hop: int = 128) -> torch.Tensor:
    """Change duration by 1/r without changing pitch (phase vocoder)."""
    win = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft, hop, window=win, return_complex=True)
    adv = torch.linspace(0, math.pi * hop, spec.shape[-2], device=x.device)[..., None]
    st = AF.phase_vocoder(spec, r, adv)
    return torch.istft(st, n_fft, hop, window=win, length=int(round(x.shape[-1] / r)))


def pitch_shift(x: torch.Tensor, semitones: float) -> torch.Tensor:
    """Shift pitch by `semitones`, keeping duration: stretch by 1/f, then resample by f."""
    f = 2 ** (semitones / 12)
    return resample_rate(time_stretch(x, 1 / f), f)


def fill_from_bank(audio, off, T, rng, fade=800, keys=None):
    """A noise track of exactly T samples built from several DIFFERENT bank segments joined by 50 ms crossfades,
    starting at random points: no segment is looped. `keys` restricts the draw to those segment indices."""
    out = np.zeros(T + fade, np.float32)
    pos, ramp = 0, np.linspace(0, 1, fade, dtype=np.float32)
    while pos < T:
        k = int(rng.integers(len(off) - 1)) if keys is None else int(keys[rng.integers(len(keys))])
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


def rumble_v2(nb, T, dev):
    """Steady low drone (< ~150 Hz) + hum with harmonics (f0 45-120 Hz, fading up to ~4 kHz), generated over the whole
    clip; level drifts with a smooth RANDOM envelope (no periodic swell, nothing looped). Matches the 'wave'
    backgrounds of private clips: a bright constant band below 150 Hz and horizontal harmonic lines."""
    f = torch.fft.rfftfreq(T, 1 / SR).to(dev)[None]
    fc = torch.empty(nb, 1, device=dev).uniform_(60, 160)
    drone = torch.fft.irfft(torch.randn(nb, f.shape[1], dtype=torch.cfloat, device=dev) / torch.sqrt(1 + (f / fc) ** 8), n=T)
    drone = drone / _rms(drone)
    t = torch.arange(T, device=dev).float()[None] / SR
    f0 = torch.empty(nb, 1, device=dev).uniform_(45, 120)
    hum = torch.zeros(nb, T, device=dev)
    decay = torch.empty(nb, 1, device=dev).uniform_(0.25, 0.7)
    for h in range(1, 40):
        amp = decay ** (h - 1) * (f0 * h < 4000).float()
        hum = hum + amp * torch.sin(2 * math.pi * f0 * h * (1 + 0.002 * torch.randn(nb, 1, device=dev)) * t
                                    + torch.empty(nb, 1, device=dev).uniform_(0, 6.283))
    hum = hum / _rms(hum)
    mix = torch.empty(nb, 1, device=dev).uniform_(0.2, 0.8)
    n = mix * drone + (1 - mix) * hum
    # smooth random level drift: low-passed noise (< 0.5 Hz), +-3 dB
    env = torch.fft.irfft(torch.randn(nb, f.shape[1], dtype=torch.cfloat, device=dev) * (f < 0.5).float(), n=T)
    env = env / (env.abs().amax(-1, keepdim=True) + 1e-8)
    return n * 10 ** (3 * env / 20)


class Augmenter:
    def __init__(self, p_noise=0.6, p_reverb=0.3, p_band=0.2, p_eq=0.3, p_mulaw=0.1,
                 snr_babble=(13, 20), snr_noise=(5, 20), noise_bank: torch.Tensor | None = None,
                 snr_bank=(0, 15),
                 p_speed=0.0, speed_range=(0.9, 1.1), p_pitch=0.0, pitch_range=(-2.0, 2.0),
                 p_tempo=0.0, tempo_range=(0.85, 1.15), p_gain=0.0, gain_db=(-20.0, 10.0),
                 p_timemask=0.0, timemask=(1, 3, 0.05, 0.25), p_freqmask=0.0, p_clip=0.0, n_groups=4,
                 p_overlap=0.0, snr_overlap=(5.0, 15.0), overlap_frac=(0.3, 1.0), noise_beta=(0.0, 2.0),
                 p_matched=0.0, matched: dict | None = None, matched_snr_shift=0.0, matched_hiss_frac=0.2,
                 p_shared=0.0, shared_bank: dict | None = None, snr_shared=(3.0, 18.0), p_shared_eq=0.0,
                 p_rumble=0.0, snr_rumble=(-3.0, 8.0), rumble_version=1,
                 rir_bank: dict | None = None, p_rir_bank=1.0,
                 musan_bank: dict | None = None, p_musan=0.0, musan_cats=None, snr_musan: dict | None = None):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        # real RIRs (lists/rir_bank.npz): replace the synthetic RIR in a share p_rir_bank of the reverb draws
        self.rir_t = torch.from_numpy(rir_bank["rir"].astype(np.float32)) if rir_bank is not None else None
        # MUSAN / RIRS_NOISES noise (lists/musan_bank.npz): replace babble/coloured noise in a share p_musan of the
        # noise draws; category drawn uniformly from musan_cats, SNR from snr_musan (Kaldi / WeSpeaker ranges)
        if musan_bank is not None:
            cats = musan_cats or sorted(set(musan_bank["cat"].tolist()))
            self.musan_keys = {c: np.flatnonzero(musan_bank["cat"] == c) for c in cats}
            assert all(len(k) for k in self.musan_keys.values()), f"unknown musan category in {cats}"
            self.snr_musan = {**MUSAN_SNR, **(snr_musan or {})}

    def _rand(self, b, device):
        return torch.rand(b, device=device)

    def musan_track(self, cat, T, rng):
        """One external-noise track of T samples; musan_speech sums 3-7 different talkers (babble)."""
        audio, off, keys = self.musan_bank["audio"], self.musan_bank["offsets"], self.musan_keys[cat]
        if cat != "musan_speech":
            return fill_from_bank(audio, off, T, rng, keys=keys)
        return sum(fill_from_bank(audio, off, T, rng, keys=keys) for _ in range(int(rng.integers(3, 8))))

    def mix_musan(self, x, cat=None, snr=None):
        """Mix external noise into x [B, T]: category uniform over musan_cats unless given, SNR from its range."""
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,))))
        cats = [cat or list(self.musan_keys)[int(rng.integers(len(self.musan_keys)))] for _ in range(len(x))]
        n = torch.from_numpy(np.stack([self.musan_track(c, x.shape[1], rng) for c in cats])).to(x.device)
        s = [snr if snr is not None else rng.uniform(*self.snr_musan[c]) for c in cats]
        return _mix_at_snr(x, n, torch.tensor(s, device=x.device, dtype=x.dtype)[:, None])

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] float on GPU. Returns augmented copy (same shape)."""
        B, T = x.shape
        dev = x.device
        x = x.clone()

        # voice / prosody first (it is a property of the source): at most one of speed / pitch / tempo.
        # Samples are grouped into a few shared factors so each op runs batched.
        u = self._rand(B, dev)
        ps, pp, pt = self.p_speed, self.p_pitch, self.p_tempo
        ops = ((u < ps, "speed"), ((u >= ps) & (u < ps + pp), "pitch"), ((u >= ps + pp) & (u < ps + pp + pt), "tempo"))
        for m, kind in ops:
            idx = m.nonzero().squeeze(1)
            if len(idx) == 0:
                continue
            g = torch.randint(0, self.n_groups, (len(idx),))
            for k in range(self.n_groups):
                sel = idx[g == k]
                if len(sel) == 0:
                    continue
                if kind == "speed":
                    y = resample_rate(x[sel], float(torch.empty(1).uniform_(*self.speed_range)))
                elif kind == "pitch":
                    y = pitch_shift(x[sel], float(torch.empty(1).uniform_(*self.pitch_range)))
                else:
                    y = time_stretch(x[sel], float(torch.empty(1).uniform_(*self.tempo_range)))
                x[sel] = _fit_length(y, T)

        # matched noise: Gaussian noise coloured by measured noise spectra (average background or hiss), slow
        # +-4 dB level fluctuation, SNR drawn from an empirical SNR distribution (+ shift; negative = harder)
        if self.p_matched > 0 and self.matched is not None:
            m = self._rand(B, dev) < self.p_matched
            if m.any():
                idx = m.nonzero().squeeze(1)
                f = torch.fft.rfftfreq(T, 1 / SR).to(dev)
                fz = torch.as_tensor(self.matched["freqs"], device=dev, dtype=torch.float32)
                g_avg = torch.sqrt(torch.as_tensor(np.interp(f.cpu().numpy(), self.matched["freqs"], self.matched["avg"]), device=dev, dtype=torch.float32))
                g_hiss = torch.sqrt(torch.as_tensor(np.interp(f.cpu().numpy(), self.matched["freqs"], self.matched["hiss"]), device=dev, dtype=torch.float32))
                use_hiss = (torch.rand(len(idx), device=dev) < self.matched_hiss_frac)[:, None]
                g = torch.where(use_hiss, g_hiss[None], g_avg[None])
                spec = torch.randn(len(idx), f.numel(), dtype=torch.cfloat, device=dev) * g
                n = torch.fft.irfft(spec, n=T)
                t = torch.arange(T, device=dev).float() / SR
                fr = torch.empty(len(idx), 1, device=dev).uniform_(0.3, 2.0)
                ph = torch.empty(len(idx), 1, device=dev).uniform_(0, 6.283)
                n = n * 10 ** (4 * torch.sin(2 * math.pi * fr * t[None] + ph) / 20)
                snrs = torch.as_tensor(np.random.choice(self.matched["snr"], len(idx)), device=dev, dtype=torch.float32)
                snr = (snrs + self.matched_snr_shift).clamp(-5, 40)[:, None]
                x[idx] = _mix_at_snr(x[idx], n, snr)

        # overlapping secondary speaker (TV-style cross-talk): one other utterance of the batch, quieter than
        # the main speaker by snr_overlap dB, on a random contiguous stretch of overlap_frac of the crop.
        # The label stays the main speaker's -> the model learns to follow the dominant voice.
        m = self._rand(B, dev) < self.p_overlap
        if m.any():
            idx = m.nonzero().squeeze(1)
            other = (idx + torch.randint(1, B, (len(idx),), device=dev)) % B  # never itself
            for i, j in zip(idx.tolist(), other.tolist()):
                L = int(T * float(torch.empty(1).uniform_(*self.overlap_frac)))
                st = int(torch.randint(0, T - L + 1, (1,)))
                seg = x[i, st:st + L]
                n = x[j, :L] if x.shape[1] >= L else x[j]
                snr = float(torch.empty(1).uniform_(*self.snr_overlap))
                scale = _rms(seg[None]) / (_rms(n[None]) * (10 ** (snr / 20)))
                x[i, st:st + L] = seg + n * scale[0]

        # reverberation first (source in room), then additive noise
        m = self._rand(B, dev) < self.p_reverb
        if m.any():
            rir = synth_rir(int(m.sum()), dev)
            if self.rir_t is not None:
                use = torch.rand(len(rir)) < self.p_rir_bank
                if use.any():
                    rir = F.pad(rir, (0, self.rir_t.shape[1] - rir.shape[1]))
                    rir[use.to(dev)] = self.rir_t[torch.randint(0, len(self.rir_t), (int(use.sum()),))].to(dev)
            y = fft_conv(x[m], rir)
            x[m] = y * (_rms(x[m]) / _rms(y))

        m = self._rand(B, dev) < self.p_noise
        if m.any():
            idx = m.nonzero().squeeze(1)
            if self.musan_bank is not None and self.p_musan > 0:
                mu = (torch.rand(len(idx)) < self.p_musan).to(dev)
                if mu.any():
                    sel = idx[mu]
                    x[sel] = self.mix_musan(x[sel])
                idx = idx[~mu]
            kind = torch.randint(0, 3 if self.noise_bank is not None else 2, (len(idx),), device=dev)
            for k in range(3):
                sel = idx[kind == k]
                if len(sel) == 0:
                    continue
                if k == 0:  # babble: 1-3 other utterances from the batch
                    n = torch.zeros(len(sel), T, device=dev)
                    for _ in range(int(torch.randint(1, 4, (1,)))):
                        perm = torch.randint(0, B, (len(sel),), device=dev)
                        n += x[perm] / _rms(x[perm])
                    snr = torch.empty(len(sel), 1, device=dev).uniform_(*self.snr_babble)
                elif k == 1:  # coloured noise
                    beta = float(torch.empty(1).uniform_(*self.noise_beta))  # <0: blue / hiss-like
                    n = coloured_noise((len(sel), T), dev, beta)
                    snr = torch.empty(len(sel), 1, device=dev).uniform_(*self.snr_noise)
                else:  # real noise bank
                    nb = self.noise_bank
                    j = torch.randint(0, nb.shape[0], (len(sel),))
                    off = int(torch.randint(0, max(1, nb.shape[1] - T), (1,)))
                    n = nb[j, off:off + T].to(dev)
                    if n.shape[1] < T:
                        n = F.pad(n, (0, T - n.shape[1]))
                    snr = torch.empty(len(sel), 1, device=dev).uniform_(*self.snr_bank)
                x[sel] = _mix_at_snr(x[sel], n, snr)

        m = self._rand(B, dev) < self.p_band  # telephone-like band limit
        if m.any():
            n = int(m.sum())
            lo = torch.empty(n, 1, device=dev).uniform_(100, 400)
            hi = torch.empty(n, 1, device=dev).uniform_(3000, 4000)
            f = torch.fft.rfftfreq(T, 1 / SR).to(dev)[None]
            # smooth 4th-order-like Butterworth magnitude responses
            resp = 1 / torch.sqrt(1 + (f / hi) ** 8) * (1 / torch.sqrt(1 + (lo / f.clamp_min(1)) ** 4))
            x[m] = torch.fft.irfft(torch.fft.rfft(x[m]) * resp, n=T)

        m = self._rand(B, dev) < self.p_eq  # random mic colouration (2 peaking bands, +-8 dB)
        if m.any():
            n = int(m.sum())
            logf = torch.log(torch.fft.rfftfreq(T, 1 / SR).to(dev).clamp_min(20))[None]
            gain_db = torch.zeros(n, logf.shape[1], device=dev)
            for _ in range(2):
                fc = torch.empty(n, 1, device=dev).uniform_(math.log(150), math.log(6000))
                bw = torch.empty(n, 1, device=dev).uniform_(0.3, 1.2)
                g = torch.empty(n, 1, device=dev).uniform_(-8, 8)
                gain_db = gain_db + g * torch.exp(-0.5 * ((logf - fc) / bw) ** 2)
            tilt = torch.empty(n, 1, device=dev).uniform_(-3, 3)  # dB per octave-ish tilt
            gain_db = gain_db + tilt * (logf - math.log(1000)) / math.log(2)
            x[m] = torch.fft.irfft(torch.fft.rfft(x[m]) * 10 ** (gain_db / 20), n=T)

        m = self._rand(B, dev) < self.p_mulaw
        if m.any():
            y = x[m]
            peak = y.abs().amax(-1, keepdim=True).clamp_min(1e-6)
            q = AF.mu_law_encoding(y / peak, 256)
            x[m] = AF.mu_law_decoding(q, 256) * peak

        m = self._rand(B, dev) < self.p_freqmask  # SpecAugment-style frequency masking (1-2 band-stops)
        if m.any():
            n = int(m.sum())
            f = torch.fft.rfftfreq(T, 1 / SR).to(dev)[None]
            resp = torch.ones(n, f.shape[1], device=dev)
            for _ in range(int(torch.randint(1, 3, (1,)))):
                fc = torch.exp(torch.empty(n, 1, device=dev).uniform_(math.log(200), math.log(7000)))
                bw = fc * torch.empty(n, 1, device=dev).uniform_(0.1, 0.4)
                resp = resp * (1 - 0.97 * torch.exp(-0.5 * ((f - fc) / bw) ** 2))
            x[m] = torch.fft.irfft(torch.fft.rfft(x[m]) * resp, n=T)

        m = self._rand(B, dev) < self.p_clip  # overdriven mic: drive into hard clipping
        if m.any():
            y = x[m]
            lvl = y.abs().amax(-1, keepdim=True).clamp_min(1e-6) * torch.empty(len(y), 1, device=dev).uniform_(0.2, 0.6)
            x[m] = torch.maximum(torch.minimum(y, lvl), -lvl) / lvl * y.abs().amax(-1, keepdim=True)

        m = self._rand(B, dev) < self.p_timemask  # SpecAugment-style time masking: zero 1-3 chunks
        if m.any():
            nmin, nmax, fmin, fmax = self.timemask
            for i in m.nonzero().squeeze(1).tolist():
                for _ in range(int(torch.randint(nmin, nmax + 1, (1,)))):
                    w = int(T * float(torch.empty(1).uniform_(fmin, fmax)) / nmax)
                    st = int(torch.randint(0, max(1, T - w), (1,)))
                    x[i, st:st + w] = 0

        m = self._rand(B, dev) < self.p_gain  # level perturbation
        if m.any():
            g = torch.empty(int(m.sum()), 1, device=dev).uniform_(*self.gain_db)
            x[m] = x[m] * 10 ** (g / 20)

        # shared background (same-channel hard negatives): consecutive crops (0,1), (2,3), ... are different speakers
        # (random sampler); with prob p_shared a pair gets the IDENTICAL real background segment, each at its own SNR
        # draw, so "same background" stops meaning "same speaker". Bank: lists/realnoise_bank.npz.
        if self.p_shared > 0 and self.shared_bank is not None and B >= 2:
            # vectorised: one noise segment per chosen pair (tiled to T), both crops mixed in one call
            audio, off = self.shared_bank["audio"], self.shared_bank["offsets"]
            pairs = torch.arange(0, B - 1, 2)
            pairs = pairs[torch.rand(len(pairs)) < self.p_shared]
            if len(pairs):
                rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,))))
                segs = [fill_from_bank(audio, off, T, rng) for _ in range(len(pairs))]  # no looping
                n = torch.from_numpy(np.stack(segs)).to(dev).repeat_interleave(2, 0)
                idx = torch.stack([pairs, pairs + 1], 1).reshape(-1).to(dev)
                snr = torch.empty(len(idx), 1, device=dev).uniform_(*self.snr_shared)
                x[idx] = _mix_at_snr(x[idx], n, snr)
                # same channel, not only the same room noise: one mic colouration per pair, applied to both crops
                eq = pairs[torch.rand(len(pairs)) < self.p_shared_eq] if self.p_shared_eq > 0 else pairs[:0]
                if len(eq):
                    logf = torch.log(torch.fft.rfftfreq(T, 1 / SR).to(dev).clamp_min(20))[None]
                    gain_db = torch.zeros(len(eq), logf.shape[1], device=dev)
                    for _ in range(2):
                        fc = torch.empty(len(eq), 1, device=dev).uniform_(math.log(150), math.log(6000))
                        bw = torch.empty(len(eq), 1, device=dev).uniform_(0.3, 1.2)
                        g = torch.empty(len(eq), 1, device=dev).uniform_(-8, 8)
                        gain_db = gain_db + g * torch.exp(-0.5 * ((logf - fc) / bw) ** 2)
                    gain_db = gain_db + torch.empty(len(eq), 1, device=dev).uniform_(-3, 3) * (logf - math.log(1000)) / math.log(2)
                    j = torch.stack([eq, eq + 1], 1).reshape(-1).to(dev)
                    x[j] = torch.fft.irfft(torch.fft.rfft(x[j]) * 10 ** (gain_db.repeat_interleave(2, 0) / 20), n=T)

        # low-frequency rumble (13.8% of private clips, 0.2% of train): low-passed noise, cutoff 80-300 Hz, a slow
        # 0.1-2 Hz swell, sometimes a 40-120 Hz hum; full-band SNR -3..8 dB (the flagged private backgrounds sit
        # ~12 dB above the usual floor with 95-99% of their energy below 500 Hz)
        m = self._rand(B, dev) < self.p_rumble if self.p_rumble > 0 else None
        if m is not None and m.any() and self.rumble_version == 2:
            idx = m.nonzero().squeeze(1)
            x[idx] = _mix_at_snr(x[idx], rumble_v2(len(idx), T, dev),
                                 torch.empty(len(idx), 1, device=dev).uniform_(*self.snr_rumble))
        elif m is not None and m.any():
            idx = m.nonzero().squeeze(1)
            nb = len(idx)
            f = torch.fft.rfftfreq(T, 1 / SR).to(dev)[None]
            fc = torch.empty(nb, 1, device=dev).uniform_(80, 300)
            resp = 1 / torch.sqrt(1 + (f / fc) ** 8)
            n = torch.fft.irfft(torch.randn(nb, f.shape[1], dtype=torch.cfloat, device=dev) * resp, n=T)
            t = torch.arange(T, device=dev).float()[None] / SR
            hum = (torch.rand(nb, 1, device=dev) < 0.3).float()
            n = n / _rms(n) + hum * 0.7 * torch.sin(2 * math.pi * torch.empty(nb, 1, device=dev).uniform_(40, 120) * t)
            fm = torch.empty(nb, 1, device=dev).uniform_(0.1, 2.0)
            depth = torch.empty(nb, 1, device=dev).uniform_(0.3, 0.9)
            n = n * (1 + depth * torch.sin(2 * math.pi * fm * t + torch.empty(nb, 1, device=dev).uniform_(0, 6.283)))
            snr = torch.empty(nb, 1, device=dev).uniform_(*self.snr_rumble)
            x[idx] = _mix_at_snr(x[idx], n, snr)

        # keep amplitude in a sane range
        peak = x.abs().amax(-1, keepdim=True)
        x = torch.where(peak > 1.0, x / peak * 0.95, x)
        return torch.nan_to_num(x)
