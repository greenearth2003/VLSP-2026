"""Public Test benchmark (clean official audio, official labels): model selection without Silver and without AS-norm.

    .venv/bin/python experiments/distill_noiseaug_durationaware/public_bench.py PUB_<name> [more tags]

Each tag is a runs/_calib/<tag>/posttrain.npz made by posttrain_embed.py with SILVER_DIR pointing at the clean Public
Test audio (public_bench.sh does this). Back-ends, all fitted on train-cohort trials only:
  cosine          whole-clip embeddings
  qmf2            logistic calibrator on cosine + speech duration + SNR
  qmf2_norm_speech  + raw embedding norm + speech-view embedding  (the submitted back-end minus AS-norm; allowed)
  qmf2_..._asnorm   + AS-norm (the submitted back-end; reference only, disallowed)
Reports Public Test EER (%) on all 49,993 trials and on the cleaned 47,932, minDCF(0.01), and the natural buckets of
the shorter side's speech. Writes runs/_eval/public_bench.json (merged per tag) and prints a table sorted by the
selection metric: qmf2_norm_speech EER on all trials.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
sys.path[:0] = [str(EXP.parents[1]), str(EXP)]

import evaluate as ev  # noqa: E402
from posttrain_eval import l2n, run  # noqa: E402
from sred.metrics import compute_min_dcf  # noqa: E402

BACKENDS = {"qmf2": {}, "qmf2_norm_speech": dict(norm=True, speech=True),
            "qmf2_norm_speech_asnorm": dict(norm=True, speech=True, asnorm=True)}
OUT = EXP / "runs/_eval/public_bench.json"


def metrics(s, tr, y, cl):
    r = {"all": ev.eer(s, y), "cleaned": ev.eer(s[cl], y[cl]), "minDCF": float(compute_min_dcf(s, y)),
         "minDCF_cleaned": float(compute_min_dcf(s[cl], y[cl]))}
    for bk in ["<0.75s", "0.75-1s", "1-2s", "2-3s", ">3s"]:
        k = (tr.bucket == bk).to_numpy()
        r[bk] = ev.eer(s[k], y[k])
    return r


def bench(tag):
    D = dict(np.load(EXP / "runs/_calib" / tag / "posttrain.npz"))
    tr = pd.read_csv(ev.DEV / "trials.csv")
    y = tr.label.to_numpy()
    cl = np.load(ev.DEV / "silver_weights.npz")["in_cleaned"]
    pos = {f: i for i, f in enumerate(D["silver_files"])}
    ia, ib = np.array([pos[f] for f in tr.a]), np.array([pos[f] for f in tr.b])
    se = l2n(D["silver_whole"])
    res = {"cosine": metrics(np.sum(se[ia] * se[ib], 1), tr, y, cl)}
    ones = np.ones(len(y))
    for name, kw in BACKENDS.items():
        r = run(D, tr, ia, ib, y, ones, cl, **kw)
        res[name] = {"all": r["silver_all"], "cleaned": r["silver_cleaned"], "minDCF": r["silver_minDCF"],
                     **{k.replace("natural ", ""): v for k, v in r.items() if k.startswith("natural ")}}
    return res


def main():
    allr = json.loads(OUT.read_text()) if OUT.exists() else {}
    for tag in sys.argv[1:]:
        allr[tag] = bench(tag)
        OUT.write_text(json.dumps(allr, indent=1))
        print(tag, " ".join(f"{b}={allr[tag][b]['all']:.3f}/{allr[tag][b]['cleaned']:.3f}" for b in allr[tag]), flush=True)
    sel = "qmf2_norm_speech"
    print(f"\n{'model':32s} {'cosine all/cl':>15s} {'qmf2 all/cl':>15s} {sel + ' all/cl':>26s} {'minDCF':>7s}"
          f" {'(+asnorm ref)':>15s}")
    for tag, r in sorted(allr.items(), key=lambda kv: kv[1][sel]["all"]):
        print(f"{tag:32s} {r['cosine']['all']:7.3f}/{r['cosine']['cleaned']:.3f} {r['qmf2']['all']:7.3f}/"
              f"{r['qmf2']['cleaned']:.3f} {r[sel]['all']:18.3f}/{r[sel]['cleaned']:.3f} {r[sel]['minDCF']:7.3f}"
              f" {r['qmf2_norm_speech_asnorm']['all']:9.3f}/{r['qmf2_norm_speech_asnorm']['cleaned']:.3f}")


if __name__ == "__main__":
    main()
