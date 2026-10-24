"""Deep-dive on the two-view data loader: draw N steps exactly as training does and report what the model sees.

    .venv/bin/python experiments/distill_noiseaug_durationaware/inspect_loader.py --steps 300 [--set aug=null ...]

Writes runs/_inspect/report.txt and example crops (clean / augmented wav, both views of one speaker) to
runs/_inspect/examples/.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

from dnd.data import SR, build_loader, load_config  # noqa: E402


def q(v, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    return " ".join(f"{x:5.2f}" for x in np.quantile(v, qs)) if len(v) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "configs/base.json"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--examples", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = EXP / "runs/_inspect"
    (out / "examples").mkdir(parents=True, exist_ok=True)
    for old in (out / "examples").glob("*.wav"):  # examples from an earlier inspection
        old.unlink()

    t = time.time()
    dl, info = build_loader(cfg, seed=0, workers=a.workers, return_clean=True)
    t_build = time.time() - t
    aug, bank, man, sampler = info["augmenter"], info["bank"], info["manifest"], info["sampler"]

    bins, spk_count = Counter(), Counter()
    stats = {v: {"rir": Counter(), "noise": Counter(), "snr": {}, "frac": []} for v in ("short", "long")}
    distinct_ok = pair_ok = diff_utt = n_pairs = 0
    it = iter(dl)
    next(it)  # worker start-up
    t = time.time()
    for k in range(a.steps):
        b = next(it)
        sb, lb = b["bins"]
        bins[("short", sb)] += 1
        bins[("long", lb)] += 1
        s, l = b["short"], b["long"]
        lab = s["label"].numpy()
        assert (lab == l["label"].numpy()).all(), "views not aligned by speaker"
        distinct_ok += len(set(lab)) == len(lab)
        pair_ok += int((lab[0::2] != lab[1::2]).all())
        diff_utt += int((s["row"] != l["row"]).sum())
        n_pairs += len(lab)
        spk_count.update(lab.tolist())
        for v, d in (("short", s), ("long", l)):
            st = stats[v]
            st["frac"].extend(bank["speech_frac"][d["crop"].numpy()].tolist())
            if aug is None:
                continue
            for i in range(len(lab)):
                rc, nc, snr = int(d["rir_cat"][i]), int(d["noise_cat"][i]), float(d["snr"][i])
                st["rir"]["none" if rc < 0 else aug.rir_cats[rc]] += 1
                name = "none" if nc < 0 else ("shared train_bg (pair)" if nc == aug.SHARED else aug.noise_cats[nc])
                st["noise"][name] += 1
                if nc >= 0:
                    st["snr"].setdefault(name, []).append(snr)
        if k < a.examples:
            for v, d in (("short", s), ("long", l)):
                tag = f"{k:02d}_spk{lab[0]}_{v}_{d['wav'].shape[1] / SR:.2f}s"
                sf.write(out / f"examples/{tag}_clean.wav", d["clean"][0].numpy(), SR)
                sf.write(out / f"examples/{tag}_aug.wav", d["wav"][0].numpy(), SR)
    dt = time.time() - t

    S = cfg["speakers_per_step"]
    L = []
    P = L.append
    P(f"config {a.config} {' '.join(a.set)}")
    P(f"manifest {info['manifest_sha256'][:12]}: {len(man)} utts, {len(info['speakers'])} speakers; "
      f"bank {len(bank['row'])} crops; loader built in {t_build:.1f}s")
    P(f"{a.steps} steps x {S} speakers x 2 views = {a.steps * S * 2} crops; {a.workers} workers: "
      f"{a.steps / dt:.1f} steps/s ({dt / a.steps * 1000:.0f} ms/step)")
    P(f"checks: distinct speakers in {distinct_ok}/{a.steps} steps; different-speaker pairs (2k,2k+1) in "
      f"{pair_ok}/{a.steps}; short and long view from different utterances {diff_utt / n_pairs * 100:.2f}% "
      f"(same file, disjoint {sampler.fallbacks['same_file_disjoint']}, overlapping {sampler.fallbacks['same_file_overlap']})")
    c = np.array([spk_count.get(i, 0) for i in range(len(info["speakers"]))])
    P(f"speaker-uniform: draws per speaker mean {c.mean():.2f} sd {c.std():.2f} (Poisson sd {np.sqrt(c.mean()):.2f}); "
      f"never drawn {int((c == 0).sum())}")
    P("")
    for v in ("short", "long"):
        bs = sorted((b, n) for (vv, b), n in bins.items() if vv == v)
        P(f"{v} view bins: " + "  ".join(f"{b:g}s {n / a.steps * 100:.0f}%" for b, n in bs))
        P(f"  speech fraction of crops (active intervals) quantiles 10/25/50/75/90: {q(stats[v]['frac'])}")
    if aug is not None:
        for v in ("short", "long"):
            st, n = stats[v], a.steps * S
            P("")
            P(f"[{v}] reverb: " + "  ".join(f"{k} {x / n:.3f}" for k, x in sorted(st["rir"].items(), key=lambda z: -z[1])))
            P(f"[{v}] additive noise (share of crops) and SNR dB quantiles 10/25/50/75/90")
            for k, x in sorted(st["noise"].items(), key=lambda z: -z[1]):
                snr = f"  SNR {q(np.array(st['snr'][k]))}" if k in st["snr"] else ""
                P(f"    {k:24s} {x / n:6.3f}{snr}")
    else:
        P("augmentation: off (aug=null)")
    report = "\n".join(L)
    (out / "report.txt").write_text(report + "\n")
    print(report)
    print("examples:", out / "examples")


if __name__ == "__main__":
    main()
