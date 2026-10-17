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

import torch
import torch.nn.functional as F
import torchaudio.functional as AF  # mu-law

SR = 16000


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


class Augmenter:
    def __init__(self, p_noise=0.6, p_reverb=0.3, p_band=0.2, p_eq=0.3, p_mulaw=0.1,
                 snr_babble=(13, 20), snr_noise=(5, 20), noise_bank: torch.Tensor | None = None,
                 snr_bank=(0, 15),
                 p_speed=0.0, speed_range=(0.9, 1.1), p_pitch=0.0, pitch_range=(-2.0, 2.0),
                 p_tempo=0.0, tempo_range=(0.85, 1.15), p_gain=0.0, gain_db=(-20.0, 10.0),
                 p_timemask=0.0, timemask=(1, 3, 0.05, 0.25), p_freqmask=0.0, p_clip=0.0, n_groups=4,
                 p_overlap=0.0, snr_overlap=(5.0, 15.0), overlap_frac=(0.3, 1.0)):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def _rand(self, b, device):
        return torch.rand(b, device=device)

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
            y = fft_conv(x[m], rir)
            x[m] = y * (_rms(x[m]) / _rms(y))

        m = self._rand(B, dev) < self.p_noise
        if m.any():
            idx = m.nonzero().squeeze(1)
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
                    beta = float(torch.empty(1).uniform_(0, 2))
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

        # keep amplitude in a sane range
        peak = x.abs().amax(-1, keepdim=True)
        x = torch.where(peak > 1.0, x / peak * 0.95, x)
        return torch.nan_to_num(x)
