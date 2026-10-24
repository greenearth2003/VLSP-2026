"""Loader and restore checks (run before any training run on any server):

  1 correctness: every crop is exactly bank (file, start, length) on disk, its label is its manifest speaker, the
    two views hold the same speakers in the same order, pairs (2k, 2k+1) are different speakers, lengths = bins
  2 noise math: with reverb and channel effects off, the realised SNR equals the logged SNR, and both crops of a
    shared-background pair carry the same noise track
  3 restore, inputs: a loader restarted at step k yields bit-identical batches k, k+1, ... to an uninterrupted
    loader, with any number of workers
  4 restore, model: a saved checkpoint reloads to identical weights and identical embeddings on the same input

    .venv/bin/python experiments/distill_noiseaug_durationaware/tests/test_loader.py [--ckpt runs/E0/step8000.pt]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

EXP = Path(__file__).resolve().parents[1]
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

from dnd.data import build_loader, load_config  # noqa: E402
from dnd.noise import NoiseAugmenter, rms  # noqa: E402
from sred.data import TRAIN_DIR  # noqa: E402

CFG = EXP / "configs/base.json"
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def digest(b):
    h = hashlib.sha256()
    for v in ("short", "long"):
        for k in ("wav", "label", "crop", "row", "rir_cat", "noise_cat", "snr"):
            if k in b[v]:
                h.update(b[v][k].numpy().tobytes())
    return h.hexdigest()[:16]


def test_correctness(steps=4):
    print("1 correctness")
    cfg = load_config(CFG)
    dl, info = build_loader(cfg, seed=3, workers=2, return_clean=True)
    man, bank, spk2id = info["manifest"], info["bank"], info["spk2id"]
    it = iter(dl)
    for _ in range(steps):
        b = next(it)
        s, l = b["short"], b["long"]
        check(torch.equal(s["label"], l["label"]), f"step {b['step']}: both views hold the same speakers, same order")
        lab = s["label"].numpy()
        check(len(set(lab)) == len(lab), f"step {b['step']}: {len(lab)} distinct speakers")
        check(bool((lab[0::2] != lab[1::2]).all()), f"step {b['step']}: pairs (2k, 2k+1) are different speakers")
        for v, sec in (("short", b["bins"][0]), ("long", b["bins"][1])):
            d = b[v]
            ok_len, ok_lab, ok_row, ok_audio = True, True, True, True
            for i in range(len(d["label"])):
                c, r = int(d["crop"][i]), int(d["row"][i])
                ok_row &= int(bank["row"][c]) == r
                ok_len &= int(bank["length"][c]) == d["wav"].shape[1] == int(round(sec * 16000))
                ok_lab &= spk2id[man[r]["speaker_id"]] == int(d["label"][i])
                x, _ = sf.read(str(TRAIN_DIR / man[r]["audio_path"]), start=int(bank["start"][c]),
                               frames=int(bank["length"][c]), dtype="float32")
                ok_audio &= np.array_equal(x, d["clean"][i].numpy())
            check(ok_row and ok_len, f"step {b['step']} {v}: crop ids, rows and lengths match the bank ({sec:g} s)")
            check(ok_lab, f"step {b['step']} {v}: labels = manifest speakers")
            check(ok_audio, f"step {b['step']} {v}: clean crops bit-identical to the source wav at (start, length)")
            nc, snr = d["noise_cat"].numpy(), d["snr"].numpy()
            check(bool(((nc < 0) == np.isnan(snr)).all()), f"step {b['step']} {v}: SNR logged iff noise added")


def test_noise_math():
    print("2 noise math (reverb and channel off)")
    cfg = load_config(CFG)
    a = dict(cfg["aug"], p_reverb=0.0, p_noise=1.0, p_shared=0.5, channel={})
    b = cfg["banks"]
    aug = NoiseAugmenter(a, SRED / b["rir"], SRED / b["noise"], SRED / b["train_bg"])
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((32, 32000)) * 0.05).astype(np.float32)
    y, meta = aug(x, rng)
    n = y - x
    snr_real = 20 * np.log10(np.array([rms(x[i]) / rms(n[i]) for i in range(len(x))]))
    err = np.abs(snr_real - meta["snr"])
    check(float(err.max()) < 0.01, f"realised SNR = logged SNR for all 32 crops (max error {err.max():.4f} dB)")
    shared = np.flatnonzero(meta["noise_cat"] == aug.SHARED)
    pairs = [i for i in shared[::2] if i + 1 in shared and i % 2 == 0]
    corr = [float(np.corrcoef(n[i], n[i + 1])[0, 1]) for i in pairs]
    check(len(pairs) > 0 and min(corr) > 0.9999,
          f"{len(pairs)} shared-background pairs carry the same noise track (min corr {min(corr) if corr else 0:.6f})")
    cats = {aug.noise_cats[c] if 0 <= c < aug.SHARED else ("shared" if c == aug.SHARED else "none")
            for c in meta["noise_cat"]}
    print(f"       categories in this batch: {sorted(cats)}")


def test_restore_inputs():
    print("3 restore: inputs")
    cfg = load_config(CFG)
    ref = {}
    dl, _ = build_loader(cfg, seed=11, workers=3)
    it = iter(dl)
    for _ in range(6):
        b = next(it)
        ref[b["step"]] = digest(b)
    for workers in (0, 2):
        dl, info = build_loader(cfg, seed=11, workers=workers)
        info["sampler"].start_step = 4
        it = iter(dl)
        got = {}
        for _ in range(3):
            b = next(it)
            got[b["step"]] = digest(b)
        same = all(got[k] == ref[k] for k in (4, 5, 6))
        check(same, f"restart at step 4 with {workers} workers reproduces steps 4-6 bit-identically")
    dl, _ = build_loader(cfg, seed=12, workers=0)
    check(digest(next(iter(dl))) != ref[1], "a different seed gives a different batch 1")


def test_restore_model(ckpt):
    print("4 restore: model")
    from sred.models import load_model
    p = EXP / ckpt
    m1, m2 = load_model(str(p)).eval(), load_model(str(p)).eval()
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    check(all(torch.equal(sd1[k], sd2[k]) for k in sd1), f"{ckpt}: two reloads have identical weights")
    saved = torch.load(p, map_location="cpu", weights_only=False)["model"]
    check(all(torch.equal(saved[k], sd1[k]) for k in saved), f"{ckpt}: reloaded weights = saved weights")
    torch.manual_seed(0)
    x = torch.randn(4, 32000)
    with torch.no_grad():
        e1, e2 = m1(x), m2(x)
    check(torch.equal(e1, e2), f"{ckpt}: identical embeddings on the same input (CPU, fp32)")
    last = p.parent / "last.pt"
    if last.exists() and p.name == f"step{torch.load(last, map_location='cpu', weights_only=False)['step']}.pt":
        lm = torch.load(last, map_location="cpu", weights_only=False)
        check(all(torch.equal(lm["model"][k], saved[k]) for k in saved) and "opt" in lm and "objective" in lm,
              "last.pt holds the same weights plus optimizer and heads (resume state)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/E0/step8000.pt")
    a = ap.parse_args()
    test_correctness()
    test_noise_math()
    test_restore_inputs()
    if (EXP / a.ckpt).exists():
        test_restore_model(a.ckpt)
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)
