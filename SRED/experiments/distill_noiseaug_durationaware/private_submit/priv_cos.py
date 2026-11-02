"""Private Test plain-cosine scoring, same per-file computation as public_fast.py embed (whole clip capped at CAP).

    cd SRED && .venv/bin/python priv_cos.py embed --shard i --nshards N --models a,b
    cd SRED && .venv/bin/python priv_cos.py score --nshards N --models a,b
-> runs/_submit_cos/shards/shard{i}.npz ; runs/_submit_cos/results_cos_<model>.txt
"""
import sys
from pathlib import Path

import numpy as np

EXP = Path("experiments/distill_noiseaug_durationaware").resolve()
sys.path[:0] = [str(EXP.parents[1]), str(EXP)]
import public_fast as pf  # noqa: E402

DATA = Path(__import__("os").environ["SRED_DATA"]) / "private_test"
OUT = pf.R / "_submit_cos"


def trials():
    lines = [ln.split()[:2] for ln in (DATA / "private-test.txt").read_text().splitlines() if ln.strip()]
    return lines, sorted({x for ln in lines for x in ln})


TAG = "shards"


def arg(a, k):
    return a[a.index(k) + 1]


def embed(shard, nshards, names):
    import soundfile as sf
    import torch
    torch.set_num_threads(2)
    _, fs = trials()
    fs = fs[shard::nshards]
    models = {k: pf.ev.load_student(str(pf.MODELS[k]))[0].cuda().eval() for k in names}
    W = {k: [] for k in models}
    S = {k: [] for k in pf.SPEECH if k in models}
    Q = []
    for i, fn in enumerate(fs):
        x, _ = sf.read(str(DATA / "data" / fn), dtype="float32", always_2d=False)
        x = (x if x.ndim == 1 else x.mean(1))[:pf.CAP]
        sp = pf.speech_only(x)
        Q.append((*pf.quality(x), *pf.quality(sp)))
        for k, m in models.items():
            W[k].append(pf.raw(m, x))
            if k in S:
                S[k].append(pf.raw(m, sp))
        if i % 1000 == 0:
            print(f"shard {shard}: {i}/{len(fs)}", flush=True)
    (OUT / TAG).mkdir(parents=True, exist_ok=True)
    np.savez(OUT / TAG / f"shard{shard}.npz", files=np.array(fs), q=np.array(Q, np.float32),
             **{f"w_{k}": np.stack(v) for k, v in W.items()}, **{f"s_{k}": np.stack(v) for k, v in S.items()})
    print(f"shard {shard} done", flush=True)


def score(nshards, names):
    lines, fs = trials()
    parts = [dict(np.load(OUT / TAG / f"shard{i}.npz")) for i in range(nshards)]
    G = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(G["files"])
    assert list(G["files"][order]) == fs, "shards do not cover the private files exactly"
    pos = {f: i for i, f in enumerate(fs)}
    ia, ib = np.array([pos[a] for a, _ in lines]), np.array([pos[b] for _, b in lines])
    for k in names:
        e = G[f"w_{k}"][order]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        s = np.sum(e[ia] * e[ib], 1)
        assert np.isfinite(s).all()
        out = OUT / f"results_cos_{k}.txt"
        out.write_text("".join(f"{a} {b} {v:.6f}\n" for (a, b), v in zip(lines, s)))
        print(f"wrote {out} ({len(lines):,} trials)")


if __name__ == "__main__":
    a = sys.argv
    names = arg(a, "--models").split(",")
    TAG = arg(a, "--tag") if "--tag" in a else "shards"
    if a[1] == "embed":
        embed(int(arg(a, "--shard")), int(arg(a, "--nshards")), names)
    else:
        score(int(arg(a, "--nshards")), names)
