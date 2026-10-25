"""Quick screen: a fixed random sample of held-out noisy trials to reject weak ideas early (and to review
step checkpoints during training) at ~30 % of the cost of the full three-set evaluation.

    python scripts/quick_screen.py build [--n 8000]          # once: lists/quick_screen_public_noisy_n<N>.txt
    python scripts/quick_screen.py validate [--n 8000]       # agreement with the full evaluation (cached embeddings)
    python scripts/quick_screen.py score CKPT [CKPT ...] [--n 8000] [--wise 0.5]   # any checkpoint, incl. stepN.pt

Set: N trials (half target) drawn at random (seed 0) from P-hold (cleaned trials, held-out speaker half) of the
synthetic-noise stand-in, the proxy that predicted the private score. Metric: cosine EER on the set and its paired
difference vs E2b on the same trials (E2b from its cached embeddings).

Validated 2026-10-19 on 6 evaluated models (20 random draws each):
  - "hardest pairs" sets (closest to a reference threshold) do NOT rank models like the full set: Spearman -0.10
    (n=100) / 0.20 (n=500) -> not used.
  - random n=100-1000 is too noisy for the differences that matter here (0.1-0.7 EER); n=8000 (~5,300 of 17,850
    files) tracks the full deltas within ~0.15 for 5 of 6 models, but under-reads D01 (+0.24 vs +0.71 full): part
    of D01's loss is in QMF calibration, which this cosine screen does not include.
  - real-noise set: even its full P-hold half ranks models poorly (Spearman 0.46) -> not part of the screen.
Use: reject when the screen delta vs E2b is >= +0.3; anything else goes to the full evaluation.
--wise A: score A * ckpt + (1 - A) * pretrained base instead (what c_sweep's W_ models are), for step checkpoints.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.data import read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.protocol import _half  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["public_noisy"]
E2B = "exp/ft/W_E2b_sf2/avg.pt"
VALIDATE = ["exp/ft/W_E2b_sf2/avg.pt", "exp/ft/W_N10_sf2_noisy/avg.pt", "exp/ft/W_M1_p50/avg.pt", "exp/ft/W_M2_p80/avg.pt", "exp/ft/W_M3_p80_hard/avg.pt",
            "exp/ft/D01_dn48/avg.pt", "exp/ft/D03_dn48_aux5/avg.pt", "exp/ft/D02b_dn48_ex/avg.pt",
            "exp/ft/D00_ctrl/avg.pt", "exp/ft/D00c_ctrl_precrash/avg.pt"]


def public_trials():
    pairs, y = read_trials()
    files = unique_files(pairs)
    ident = json.loads((ROOT / "lists/clean/public_identity.json").read_text())
    keep = {tuple(l.split()[:2]) for l in open(ROOT / "lists/clean/public_trials_clean.txt")}
    hold = np.array([p in keep and _half(ident[p[0].split("/")[0]]) == 1 and _half(ident[p[1].split("/")[0]]) == 1
                     for p in pairs])
    return pairs, y, files, hold


def cos_from_cache(model, split, pairs, files):
    pos = {f: i for i, f in enumerate(files)}
    E = l2n(np.load(cache_key(model, split, files, mode="full")))
    ia = np.array([pos[a] for a, _ in pairs])
    ib = np.array([pos[b] for _, b in pairs])
    return (E[ia] * E[ib]).sum(1)


def auc(s, y):
    t, n = s[y == 1], s[y == 0]
    return float((t[:, None] > n[None, :]).mean() + 0.5 * (t[:, None] == n[None, :]).mean())


def list_path(split, n):
    return ROOT / f"lists/quick_screen_{split}_n{n}.txt"


def read_list(split, n):
    rows = [l.split() for l in list_path(split, n).read_text().splitlines()]
    return [(a, b) for a, b, _ in rows], np.array([int(t == "target") for _, _, t in rows])


def build(n):
    pairs, y, files, hold = public_trials()
    idx = np.where(hold)[0]
    r = np.random.default_rng(0)
    pick = sorted(np.concatenate([r.choice(idx[y[idx] == 1], n // 2, replace=False),
                                  r.choice(idx[y[idx] == 0], n // 2, replace=False)]))
    for split in SPLITS:
        list_path(split, n).write_text("".join(f"{pairs[i][0]} {pairs[i][1]} {'target' if y[i] else 'nontarget'}\n"
                                               for i in pick))
        print(f"{split}: {len(pick)} trials ({len({f for i in pick for f in pairs[i]})} files) from {len(idx)} P-hold trials")


def e2b_eer(split, qp, qy):
    pairs, _, files, _ = public_trials()
    pos = {p: i for i, p in enumerate(pairs)}
    return eer(cos_from_cache(E2B, split, pairs, files)[[pos[p] for p in qp]], qy)


def eer(s, y):
    from sred.metrics import compute_eer
    return compute_eer(s, y)


def full_eer(model, split):
    tag = Path(model).parent.name
    best = None
    for l in open(ROOT / "exp/results.jsonl"):
        d = json.loads(l)
        if d.get("model") == model and d.get("split") == split and d.get("mode", "full") == "full":
            best = d["results"]["qmf_dur"]["eer"]
    return best


def validate(n):
    pairs, y, files, _ = public_trials()
    pos = {p: i for i, p in enumerate(pairs)}
    for split in SPLITS:
        qp, qy = read_list(split, n)
        sel = [pos[p] for p in qp]
        q_ref, f_ref = eer(cos_from_cache(E2B, split, pairs, files)[sel], qy), full_eer(E2B, split)
        print(f"{split} (n={n}): model | screen delta vs E2b | full delta vs E2b")
        for m in VALIDATE[1:]:
            try:
                s = cos_from_cache(m, split, pairs, files)
            except FileNotFoundError:
                continue
            e = full_eer(m, split)
            if e is not None:
                print(f"  {Path(m).parent.name:22s} {eer(s[sel], qy) - q_ref:+.2f}   {e - f_ref:+.2f}")


def score(ckpts, n, wise):
    import torch
    from sred.embed import extract
    from sred.models import load_model
    data = ROOT / "data"
    for c in ckpts:
        model = load_model(c)
        if wise is not None:
            st = torch.load(c, map_location="cpu", weights_only=False)
            from sred.expand import base_state
            base = base_state(st["base_model"], st.get("expand")) if st.get("expand") else load_model(st["base_model"]).state_dict()
            sd = {k: (wise * v.double() + (1 - wise) * base[k].double()).to(v.dtype) if v.is_floating_point() else v
                  for k, v in model.state_dict().items()}
            model.load_state_dict(sd)
        out = []
        for split in SPLITS:
            qp, qy = read_list(split, n)
            qf = sorted({f for p in qp for f in p})
            E = l2n(extract(model, qf, data / split, num_workers=4))
            p = {f: i for i, f in enumerate(qf)}
            s = np.array([(E[p[a]] * E[p[b]]).sum() for a, b in qp])
            e, r = eer(s, qy), e2b_eer(split, qp, qy)
            out.append(f"{split} screen EER {e:.2f} (E2b {r:.2f}, delta {e - r:+.2f}{'  -> REJECT' if e - r >= 0.3 else ''})")
        print(f"{c}{'' if wise is None else f' (wise {wise})'}: " + "   ".join(out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "validate", "score"])
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--wise", type=float, default=None)
    a = ap.parse_args()
    {"build": lambda: build(a.n), "validate": lambda: validate(a.n), "score": lambda: score(a.ckpts, a.n, a.wise)}[a.cmd]()
