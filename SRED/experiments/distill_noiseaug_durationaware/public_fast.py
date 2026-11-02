"""Fast Public Test model selection: plain cosine of whole-clip embeddings on the clean official audio, many models at once.

    .venv/bin/python experiments/distill_noiseaug_durationaware/public_fast.py embed --shard I --nshards N --group G [--models a,b]
    .venv/bin/python experiments/distill_noiseaug_durationaware/public_fast.py rank
    .venv/bin/python experiments/distill_noiseaug_durationaware/public_fast.py backend NAME CALIB_TAG

embed: every model in MODELS on one shard of the 17,850 Public Test files; each file is read once. For the models in
SPEECH it also stores the speech-only view, and for all files the qualities (speech s, SNR dB), so the calibrated
back-end can be applied without another pass. --models splits the list across servers (missing WiSE files are built).
-> runs/_pubfast/<group>/shard{I}.npz
rank: Public Test EER (all 49,993 / cleaned 47,932) and minDCF(0.01) of plain cosine per model -> runs/_eval/public_fast.json
backend: train-cohort calibrator for NAME, re-using the cohort embeddings of runs/_calib/CALIB_TAG/posttrain.npz (same
model), with and without AS-norm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(EXP)]

import evaluate as ev  # noqa: E402
from calibrate_eval import CAP, quality  # noqa: E402
from posttrain_embed import raw, speech_only  # noqa: E402
from sred.data import PUBLIC_DIR  # noqa: E402
from sred.metrics import compute_min_dcf  # noqa: E402

R = EXP / "runs"
MODELS = {
    "E5s4000": R / "E5-b6-lowrank-kd/step4000.pt",
    "E5s6000": R / "E5-b6-lowrank-kd/step6000.pt",
    "E5s8000": R / "E5-b6-lowrank-kd/step8000.pt",
    "E5s10000": R / "E5-b6-lowrank-kd/step10000.pt",
    "E5s6000_wise0.7": R / "E5-b6-lowrank-kd/wise0.7_step6000.pt",
    "E5s8000_wise0.7": R / "_calib/_wise_pub/E5s8000_wise0.7.pt",
    "E5s10000_wise0.7": R / "_calib/_wise_pub/E5s10000_wise0.7.pt",
    "E8s5000": R / "E8-b6-lowrank-kd-continue/step5000.pt",
    "E8s5000_wise0.7": R / "_calib/E8s5000_wise0.7_A/wise0.7.pt",
}
SPEECH = {"E8s5000_wise0.7"}
OUT = R / "_pubfast"


def files():
    tr = pd.read_csv(ev.DEV / "trials.csv")
    return tr, sorted(set(tr.a) | set(tr.b))


WISE = {"E5s8000_wise0.7": "E5-b6-lowrank-kd/step8000.pt", "E5s10000_wise0.7": "E5-b6-lowrank-kd/step10000.pt"}


def embed(shard, nshards, group, names):
    from backend import wise
    torch.set_num_threads(2)
    _, fs = files()
    fs = fs[shard::nshards]
    models = {}
    for k in names:
        if not MODELS[k].exists() and k in WISE:
            MODELS[k].parent.mkdir(parents=True, exist_ok=True)
            wise(R / WISE[k], EXP / "artifacts/e5_init_lowrank_step6000.pt", MODELS[k], 0.7)
        models[k] = ev.load_student(str(MODELS[k]))[0].cuda().eval()
    W = {k: [] for k in models}
    S = {k: [] for k in SPEECH if k in models}
    Q = []
    for i, fn in enumerate(fs):
        x, _ = sf.read(str(PUBLIC_DIR / fn), dtype="float32", always_2d=False)
        x = (x if x.ndim == 1 else x.mean(1))[:CAP]
        sp = speech_only(x)
        Q.append((*quality(x), *quality(sp)))
        for k, m in models.items():
            W[k].append(raw(m, x))
            if k in SPEECH:
                S[k].append(raw(m, sp))
        if i % 1000 == 0:
            print(f"shard {shard}: {i}/{len(fs)}", flush=True)
    (OUT / group).mkdir(parents=True, exist_ok=True)
    np.savez(OUT / group / f"shard{shard}.npz", files=np.array(fs), q=np.array(Q, np.float32),
             **{f"w_{k}": np.stack(v) for k, v in W.items()}, **{f"s_{k}": np.stack(v) for k, v in S.items()})
    print(f"shard {shard} done", flush=True)


def load_all():
    tr, fs = files()
    D = {"files": np.array(fs)}
    for g in sorted(d for d in OUT.iterdir() if d.is_dir()):
        parts = [dict(np.load(p)) for p in sorted(g.glob("shard*.npz"))]
        G = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.argsort(G["files"])  # fs is sorted: align every group to it
        assert list(G["files"][order]) == fs, f"{g.name}: incomplete shards"
        D.update({k: v[order] for k, v in G.items() if k != "files"})
    pos = {f: i for i, f in enumerate(D["files"])}
    ia, ib = np.array([pos[f] for f in tr.a]), np.array([pos[f] for f in tr.b])
    return D, tr, ia, ib


def mets(s, y, cl):
    return {"all": ev.eer(s, y), "cleaned": ev.eer(s[cl], y[cl]), "minDCF": float(compute_min_dcf(s, y)),
            "minDCF_cleaned": float(compute_min_dcf(s[cl], y[cl]))}


def rank():
    D, tr, ia, ib = load_all()
    y = tr.label.to_numpy()
    cl = np.load(ev.DEV / "silver_weights.npz")["in_cleaned"]
    res = {}
    for k in [m for m in MODELS if f"w_{m}" in D]:
        e = D[f"w_{k}"] / np.linalg.norm(D[f"w_{k}"], axis=1, keepdims=True)
        res[k] = mets(np.sum(e[ia] * e[ib], 1), y, cl)
    (R / "_eval/public_fast.json").write_text(json.dumps(res, indent=1))
    print(f"{'model (plain cosine, clean Public Test)':40s} {'EER all':>8s} {'cleaned':>8s} {'minDCF':>7s} {'(cl)':>6s}")
    for k, r in sorted(res.items(), key=lambda kv: kv[1]["all"]):
        print(f"{k:40s} {r['all']:8.3f} {r['cleaned']:8.3f} {r['minDCF']:7.3f} {r['minDCF_cleaned']:6.3f}")


def backend(name, calib_tag):
    from posttrain_eval import run
    D, tr, ia, ib = load_all()
    y = tr.label.to_numpy()
    cl = np.load(ev.DEV / "silver_weights.npz")["in_cleaned"]
    C = dict(np.load(R / "_calib" / calib_tag / "posttrain.npz"))
    w = D[f"w_{name}"]
    q = D["q"]
    P = {k: v for k, v in C.items() if k.startswith("coh_") or k.startswith("q_coh")}
    has_sp = f"s_{name}" in D
    P.update(silver_whole=w, silver_speech=D[f"s_{name}"] if has_sp else w, silver_files=D["files"],
             q_silver_whole=np.c_[q[:, :2]], q_silver_speech=np.c_[q[:, 2:4]])
    out = {}
    variants = [("qmf2_norm (no AS-norm, no speech view)", dict(norm=True))]
    if has_sp:
        variants += [("qmf2_norm_speech (no AS-norm)", dict(norm=True, speech=True)),
                     ("qmf2_norm_speech_asnorm (submitted, reference)", dict(norm=True, speech=True, asnorm=True))]
    for lab, kw in variants:
        r = run(P, tr, ia, ib, y, np.ones(len(y)), cl, **kw)
        out[lab] = {"all": r["silver_all"], "cleaned": r["silver_cleaned"], "minDCF": r["silver_minDCF"]}
        print(f"{name} {lab}: EER all {r['silver_all']:.3f}  cleaned {r['silver_cleaned']:.3f}  minDCF {r['silver_minDCF']:.3f}")
    (R / f"_eval/public_fast_backend__{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "embed":
        a = sys.argv
        names = a[a.index("--models") + 1].split(",") if "--models" in a else list(MODELS)
        embed(int(a[a.index("--shard") + 1]), int(a[a.index("--nshards") + 1]), a[a.index("--group") + 1], names)
    elif c == "rank":
        rank()
    else:
        backend(sys.argv[2], sys.argv[3])
