"""Central validation for E0-E3 (reference server only; NEW_PLAN §7, §15): plain cosine scoring of full 192-D
embeddings on the silver audio (public labels), no calibration, no TTA.

    .venv/bin/python experiments/distill_noiseaug_durationaware/evaluate.py runs/E0/step8000.pt [more checkpoints]
    .venv/bin/python experiments/distill_noiseaug_durationaware/evaluate.py b4-pretrained        # zero-shot reference
    .venv/bin/python experiments/distill_noiseaug_durationaware/evaluate.py --table             # Pareto table of all results

Reports EER (%) for: silver all / cleaned / private-like weighting, the natural buckets of the shorter side's speech
(<0.75, 0.75-1, 1-2, 2-3, >3 s), and the controlled conditions full-full, full-3/2/1/0.75/0.5 s, 2s-2s, 1s-1s
(artifacts/duration_eval), plus minDCF(0.01) on silver all. Results: runs/_eval/<run>__<ckpt>[<suffix>].json.

Needs only artifacts/duration_eval/ (trials, crops, conditions, silver_weights.npz) and the silver audio
(default SRED/data/public_rumble_v2; override with SILVER_DIR=...). Another server passes --suffix __<server> so its
JSONs never overwrite the reference server's.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parent
SRED = EXP.parents[1]
sys.path[:0] = [str(SRED), str(SRED / "scripts"), str(EXP)]

from prep.build_eval_weights import pairs_hash  # noqa: E402
from sred.metrics import compute_min_dcf  # noqa: E402

SILVER = Path(os.environ.get("SILVER_DIR", SRED / "data/public_rumble_v2"))
DEV = EXP / "artifacts/duration_eval"
OUT = EXP / "runs/_eval"
BS = 64


def eer(s, y, w=None):
    o = np.argsort(s)
    yo = y[o]
    wo = np.ones(len(y)) if w is None else w[o]
    frr = np.cumsum(wo * yo) / (wo * yo).sum()
    far = 1 - np.cumsum(wo * (1 - yo)) / (wo * (1 - yo)).sum()
    k = np.argmin(np.abs(frr - far))
    return float(100 * (frr[k] + far[k]) / 2)


def load_student(spec: str):
    from sred.models import _load_redimnet2_pt, load_model
    if spec == "b4-pretrained":
        return _load_redimnet2_pt(EXP / "pretrained/redimnet2/b4-vox2-lm.pt"), "b4-pretrained"
    p = Path(spec)
    p = p if p.is_absolute() or p.exists() else EXP / spec
    return load_model(str(p)), f"{p.parent.name}__{p.stem}"


@torch.inference_mode()
def embed(model, items, dev="cuda"):
    """items: list of (key, file, start, length or None = whole file) -> {key: normalized embedding}."""
    out = {}
    groups = defaultdict(list)
    for it in items:
        groups[it[3]].append(it)
    for L, its in groups.items():
        if L is None:  # whole files: one at a time (lengths differ)
            for key, f, _, _ in its:
                x, _ = sf.read(str(SILVER / f), dtype="float32")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out[key] = F.normalize(model(torch.from_numpy(x)[None].to(dev)).float(), dim=1)[0].cpu().numpy()
            continue
        for i in range(0, len(its), BS):
            ch = its[i:i + BS]
            x = np.stack([sf.read(str(SILVER / f), start=s, frames=L, dtype="float32")[0] for _, f, s, _ in ch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = F.normalize(model(torch.from_numpy(x).to(dev)).float(), dim=1).cpu().numpy()
            out.update({k: v for (k, _, _, _), v in zip(ch, e)})
    return out


def evaluate(spec, suffix=""):
    model, tag = load_student(spec)
    model = model.cuda().eval()
    tr = pd.read_csv(DEV / "trials.csv")
    crops = pd.read_csv(DEV / "crops.csv")
    conds = json.loads((DEV / "conditions.json").read_text())
    y = tr.label.to_numpy()
    files = sorted(set(tr.a) | set(tr.b))
    items = [(("full", f), f, 0, None) for f in files]
    items += [((n, f), f, int(s), int(L)) for f, n, s, L in crops[["file", "N", "start", "length"]].itertuples(index=False)]
    E = embed(model, items)

    def score(ea, eb, idx):  # ea / eb: "full" or a crop length in seconds
        a = np.stack([E[(ea, f)] for f in tr.a.to_numpy()[idx]])
        b = np.stack([E[(eb, f)] for f in tr.b.to_numpy()[idx]])
        return (a * b).sum(1)

    sw = np.load(DEV / "silver_weights.npz")  # frozen private-like weights + cleaned flags (prep/build_eval_weights.py)
    assert str(sw["pairs_sha256"]) == pairs_hash(tr.a, tr.b), "silver_weights.npz does not match trials.csv"
    w, cl = sw["private_like"], sw["in_cleaned"]

    every = np.arange(len(tr))
    s_full = score("full", "full", every)
    tag += suffix
    r = {"tag": tag, "host": socket.gethostname(), "silver_all": eer(s_full, y), "silver_cleaned": eer(s_full[cl], y[cl]),
         "silver_private_like": eer(s_full, y, w), "silver_minDCF": float(compute_min_dcf(s_full, y))}
    for name, c in conds.items():
        idx = np.array(c["trials"], int)
        if not len(idx):
            continue
        key = lambda v: "full" if v is None else float(v)  # noqa: E731
        s = s_full[idx] if c["kind"] == "natural" else score(key(c["enrol"]), key(c["test"]), idx)
        r[name] = eer(s, y[idx])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(r, indent=2))
    return r


COLS = ["silver_all", "silver_cleaned", "silver_private_like", "natural <0.75s", "natural 0.75-1s", "natural 1-2s",
        "natural 2-3s", "natural >3s", "full-full", "full-3s", "full-2s", "full-1s", "full-0.75s", "full-0.5s",
        "2s-2s", "1s-1s", "silver_minDCF"]


def table():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    df = pd.DataFrame(rows).set_index("tag").reindex(columns=COLS)
    df.columns = [c.replace("silver_", "").replace("natural ", "") for c in COLS]
    print(df.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--suffix", default="", help="appended to the result name, e.g. __l40s on another server")
    a = ap.parse_args()
    for c in a.ckpts:
        r = evaluate(c, a.suffix)
        print(r["tag"], " ".join(f"{k}={v:.3f}" for k, v in r.items() if k not in ("tag", "host")), flush=True)
    if a.table or not a.ckpts:
        table()


if __name__ == "__main__":
    main()
