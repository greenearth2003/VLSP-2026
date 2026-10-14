"""On-GPU waveform augmentation for robustness to noise / channel / edge mics.

No external corpora needed: babble comes from other utterances in the batch,
noise is synthetic coloured noise, reverberation uses synthetic exponential-decay
RIRs, and channel effects are simulated with band-limiting, random EQ and mu-law.
If MUSAN / RIR corpora are available, pass their waveforms via `noise_bank`.
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


class Augmenter:
    def __init__(self, p_noise=0.6, p_reverb=0.3, p_band=0.2, p_eq=0.3, p_mulaw=0.1,
                 snr_babble=(13, 20), snr_noise=(5, 20), noise_bank: torch.Tensor | None = None,
                 snr_bank=(0, 15)):
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

        # keep amplitude in a sane range
        peak = x.abs().amax(-1, keepdim=True)
        x = torch.where(peak > 1.0, x / peak * 0.95, x)
        return torch.nan_to_num(x)
