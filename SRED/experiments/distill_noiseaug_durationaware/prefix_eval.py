"""Debug: score silver with embedding PREFIXES z[:d] (d = 24, 48, 96, 192) for one checkpoint, on the natural buckets
and controlled-duration conditions. Tells whether duration information sits in the leading dimensions (DAME) and
whether the full-embedding cosine dilutes it.

    .venv/bin/python experiments/distill_noiseaug_durationaware/prefix_eval.py runs/E1/step8000.pt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
EXP = Path(__file__).resolve().parent
sys.path[:0] = [str(EXP)]
import evaluate as ev  # noqa: E402

DIMS = [24, 48, 96, 192]
CONDS = ["natural <0.75s", "natural 0.75-1s", "natural 1-2s", "natural >3s", "full-full", "full-1s", "full-0.5s", "1s-1s"]


def main(spec):
    model, tag = ev.load_student(spec)
    model = model.cuda().eval()
    tr = pd.read_csv(ev.DEV / "trials.csv")
    crops = pd.read_csv(ev.DEV / "crops.csv")
    conds = json.loads((ev.DEV / "conditions.json").read_text())
    y = tr.label.to_numpy()
    files = sorted(set(tr.a) | set(tr.b))
    items = [(("full", f), f, 0, None) for f in files]
    items += [((n, f), f, int(s), int(L)) for f, n, s, L in crops[["file", "N", "start", "length"]].itertuples(index=False)]
    E = ev.embed(model, items, normalize=False)
    out = {"tag": tag}
    for d in DIMS:
        def emb(key, f):
            v = E[(key, f)][:d]
            return v / (np.linalg.norm(v) + 1e-9)
        for name in CONDS:
            c = conds[name]
            idx = np.array(c["trials"], int)
            ka = "full" if c["enrol"] is None else float(c["enrol"])
            kb = "full" if c["test"] is None else float(c["test"])
            s = np.array([emb(ka, a) @ emb(kb, b) for a, b in zip(tr.a.to_numpy()[idx], tr.b.to_numpy()[idx])])
            out[f"{name}@{d}"] = ev.eer(s, y[idx])
    (ev.OUT / f"prefix__{tag}.json").write_text(json.dumps(out, indent=1))
    print(tag)
    print(f"  {'condition':16s}" + "".join(f"{d:>9d}" for d in DIMS))
    for name in CONDS:
        print(f"  {name:16s}" + "".join(f"{out[f'{name}@{d}']:9.3f}" for d in DIMS))


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        main(spec)
