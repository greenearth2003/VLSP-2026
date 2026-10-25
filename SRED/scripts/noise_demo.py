"""Listening page: one clean train utterance passed through every noise / reverb / channel type the augmenter can apply,
each at a mid-range training level. Writes exp/analysis/listen/noise_demo/{index.html, audio/*.mp3, spec/*.jpg}.

    python scripts/noise_demo.py [--rel id00129/0020.wav]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sred.augment import (SR, Augmenter, _mix_at_snr, _rms, coloured_noise, fft_conv,  # noqa: E402
                          fill_from_bank, synth_rir)

TRAIN = ROOT / "data/Vi-SRED26/train/data"
OUT = ROOT / "exp/analysis/listen/noise_demo"
OFF = dict(p_noise=0, p_reverb=0, p_band=0, p_eq=0, p_mulaw=0)


def load(rel, T=None):
    x, sr = sf.read(TRAIN / rel, dtype="float32")
    assert sr == SR
    return torch.from_numpy(x[:T] if T else x)[None]


def reverb(x, h):
    y = fft_conv(x, h)
    return y * (_rms(x) / _rms(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", default="id00129/0020.wav")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = load(a.rel)
    T = x.shape[1]

    # babble: three other train speakers with clips at least as long
    items = [ln.split() for ln in open(ROOT / "lists/clean/train_clean.txt")]
    spk = a.rel.split("/")[0]
    others, used = [], {spk}
    for i in rng.permutation(len(items)):
        s, rel = items[i]
        if s not in used and sf.info(TRAIN / rel).frames >= T:
            others.append(load(rel, T))
            used.add(s)
        if len(others) == 3:
            break
    babble = sum(o / _rms(o) for o in others)

    rb = dict(np.load(ROOT / "lists/rir_bank.npz"))
    mb = dict(np.load(ROOT / "lists/musan_bank.npz"))
    real_bg = dict(np.load(ROOT / "lists/realnoise_bank.npz"))
    mus = Augmenter(**OFF, musan_bank=mb)

    def bank_rir(cat):
        idx = np.flatnonzero(rb["cat"] == cat)
        return torch.from_numpy(rb["rir"][rng.choice(idx)].astype(np.float32))[None]

    def aug(**kw):
        return Augmenter(**{**OFF, **kw})(x)

    G = {"cur": "In training now (C2X recipe)", "off": "Built, switched off in C2X",
         "rir": "New: RIRS_NOISES reverb (--rir-bank)", "mus": "New: MUSAN and RIRS_NOISES noise (--musan-bank)",
         "chan": "Channel effects (not noise, for reference)", "mix": "New: a typical crop with both"}
    rows = [
        ("orig", "cur", "Original", "Clean train utterance, as stored", "—", "—", "7.1% / 2.8% of crops stay fully clean", x),
        ("white", "cur", "Coloured noise, white (β 0)", "Synthetic Gaussian noise, flat spectrum", "12.5 dB", "5–20 dB, β 0–2",
         "30% of crops (all β)", _mix_at_snr(x, coloured_noise((1, T), "cpu", 0.0), 12.5)),
        ("pink", "cur", "Coloured noise, pink (β 1)", "Synthetic 1/f noise", "12.5 dB", "5–20 dB, β 0–2", "in the 30% above",
         _mix_at_snr(x, coloured_noise((1, T), "cpu", 1.0), 12.5)),
        ("brown", "cur", "Coloured noise, brown (β 2)", "Synthetic 1/f² noise, low rumble-like", "12.5 dB", "5–20 dB, β 0–2",
         "in the 30% above", _mix_at_snr(x, coloured_noise((1, T), "cpu", 2.0), 12.5)),
        ("babble", "cur", "Babble", "Three other train speakers summed (in-batch utterances)", "16.5 dB", "13–20 dB, 1–3 talkers",
         "30% of crops", _mix_at_snr(x, babble, 16.5)),
        ("srir_dry", "cur", "Synthetic RIR, small room", "Exponential-decay noise RIR, RT60 0.3 s (synth_rir)", "RT60 0.3 s",
         "RT60 0.2–0.9 s", "30% of crops (all RT60)", reverb(x, synth_rir(1, "cpu", (0.3, 0.3)))),
        ("srir_wet", "cur", "Synthetic RIR, large room", "Same model, RT60 0.8 s", "RT60 0.8 s", "RT60 0.2–0.9 s",
         "in the 30% above", reverb(x, synth_rir(1, "cpu", (0.8, 0.8)))),
        ("realbg", "cur", "Real train background (shared pair)", "Quiet frames harvested from train recordings "
         "(lists/realnoise_bank.npz); both crops of a pair get the same one", "10.5 dB", "3–18 dB", "50% / 80% of crops",
         _mix_at_snr(x, torch.from_numpy(fill_from_bank(real_bg["audio"], real_bg["offsets"], T, rng))[None], 10.5)),
        ("matched", "off", "Matched noise", "Gaussian noise coloured by private's measured background spectrum, ±4 dB "
         "swell (M1–M4 runs)", "drawn from private", "private's SNR distribution", "0% (rejected)",
         aug(p_matched=1.0, matched=dict(np.load(ROOT / "lists/matched_noise.npz")), matched_hiss_frac=0.0)),
        ("rumble1", "off", "Rumble v1", "Low-passed swelling noise, 80–300 Hz, sometimes a hum (R1 run)", "2.5 dB",
         "−3–8 dB", "0%", aug(p_rumble=1.0, snr_rumble=(2.5, 2.5), rumble_version=1)),
        ("rumble2", "off", "Rumble v2", "Steady drone plus harmonic hum", "2.5 dB", "−3–8 dB", "0%",
         aug(p_rumble=1.0, snr_rumble=(2.5, 2.5), rumble_version=2)),
        ("rir_small", "rir", "Simulated RIR, small room", "RIRS_NOISES simulated_rirs/smallroom (1–10 m rooms)",
         "one RIR", "1,500 sampled", "≈9.3% of crops", reverb(x, bank_rir("sim_small"))),
        ("rir_medium", "rir", "Simulated RIR, medium room", "simulated_rirs/mediumroom (10–30 m)", "one RIR", "1,500 sampled",
         "≈9.3% of crops", reverb(x, bank_rir("sim_medium"))),
        ("rir_large", "rir", "Simulated RIR, large room", "simulated_rirs/largeroom (30–50 m)", "one RIR", "1,500 sampled",
         "≈9.3% of crops", reverb(x, bank_rir("sim_large"))),
        ("rir_real", "rir", "Real recorded RIR", "real_rirs_isotropic_noises: RWCP, AIR, REVERB 2014 rooms (channel 0)",
         "one RIR", "325 RIRs", "≈2.0% of crops", reverb(x, bank_rir("real"))),
        ("mus_noise", "mus", "MUSAN noise", "Free-sound and sound-bible noises: machines, traffic, rain, beeps…", "7.5 dB",
         "0–15 dB", "≈3.75% of crops", mus.mix_musan(x, "musan_noise", 7.5)),
        ("mus_music", "mus", "MUSAN music", "Instrumental and vocal music (Jamendo, free-sound…)", "10 dB", "5–15 dB",
         "≈3.75% of crops", mus.mix_musan(x, "musan_music", 10.0)),
        ("mus_speech", "mus", "MUSAN speech babble", "3–7 MUSAN talkers summed (mostly English read speech)", "16.5 dB",
         "13–20 dB", "0% (off by default)", mus.mix_musan(x, "musan_speech", 16.5)),
        ("pointsrc", "mus", "RIRS_NOISES point-source noise", "Single foreground/background noise sources (from MUSAN-like "
         "free-sound clips)", "7.5 dB", "0–15 dB", "≈3.75% of crops", mus.mix_musan(x, "rirs_pointsource", 7.5)),
        ("isotropic", "mus", "RIRS_NOISES isotropic noise", "Diffuse room noise recorded with the real RIRs (REVERB, RWCP)",
         "7.5 dB", "0–15 dB", "≈3.75% of crops", mus.mix_musan(x, "rirs_isotropic", 7.5)),
        ("band", "chan", "Band-limit", "Telephone-like 100–400 Hz to 3–4 kHz band", "—", "—", "20% of crops", aug(p_band=1.0)),
        ("eq", "chan", "Mic EQ", "Two random ±8 dB peaking bands plus tilt", "—", "—", "30% of crops", aug(p_eq=1.0)),
        ("mulaw", "chan", "μ-law", "8-bit μ-law codec round trip", "—", "—", "10% of crops", aug(p_mulaw=1.0)),
        ("combo", "mix", "Real RIR + MUSAN noise", "Medium simulated room, then MUSAN noise: what one MUS1 crop can look like",
         "7.5 dB", "—", "≈4.5% of crops", mus.mix_musan(reverb(x, bank_rir("sim_medium")), "musan_noise", 7.5)),
    ]

    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "spec").mkdir(parents=True, exist_ok=True)
    peak = max(float(r[-1].abs().max()) for r in rows)
    gain = 0.89 / peak  # one gain for every clip, so loudness differences are real
    win = torch.hann_window(512)
    ref = None
    meta = []
    for key, grp, title, what, used, rng_, share, y in rows:
        y = (y[0] * gain).clamp(-1, 1)
        wav = OUT / f"audio/{key}.wav"
        sf.write(wav, y.numpy(), SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "96k",
                        str(OUT / f"audio/{key}.mp3")], check=True)
        wav.unlink()
        S = 20 * torch.log10(torch.stft(y, 512, 160, window=win, return_complex=True).abs() + 1e-6)
        ref = float(S.max()) if ref is None else ref  # the original sets the colour scale for all
        fig = plt.figure(figsize=(9, 1.5), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(S.numpy(), origin="lower", aspect="auto", cmap="magma", vmin=ref - 80, vmax=ref)
        ax.axis("off")
        fig.savefig(OUT / f"spec/{key}.jpg", pil_kwargs={"quality": 85})
        plt.close(fig)
        meta.append(dict(key=key, group=grp, title=title, what=what, used=used, range=rng_, share=share))
    tpl = (Path(__file__).parent / "noise_demo_template.html").read_text()
    page = tpl.replace("/*GROUPS*/{}", json.dumps(G)).replace("/*ROWS*/[]", json.dumps(meta, ensure_ascii=False))
    page = page.replace("{{REL}}", a.rel).replace("{{DUR}}", f"{T / SR:.1f}")
    (OUT / "index.html").write_text(page)
    print(OUT / "index.html", len(meta), "clips")


if __name__ == "__main__":
    main()
