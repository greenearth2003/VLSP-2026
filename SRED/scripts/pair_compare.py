"""Trial-level comparison of two systems on one public split: which pairs does CKPT fix / break vs REF?

    python scripts/pair_compare.py REF_CKPT CKPT [SPLIT]     # SPLIT: public | public_noisy | public_realnoise

QMF(dur) scores from cached embeddings (run sred.evaluate on SPLIT first). Decisions at each system's own EER
threshold. Prints: decision transitions, EER by pair SNR / speech-duration bins (features_<split>.csv,
measured on that split's audio), and the pairs with the largest fixes and breaks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sred.calibrate import QMF  # noqa: E402
from sred.data import cohort_subset, read_train_list, read_trials, unique_files  # noqa: E402
from sred.embed import cache_key  # noqa: E402
from sred.metrics import compute_eer  # noqa: E402
from sred.scoring import l2n  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ref, new = sys.argv[1], sys.argv[2]
split = sys.argv[3] if len(sys.argv) > 3 else "public_noisy"
pairs, y = read_trials()
files = unique_files(pairs)
pos = {f: i for i, f in enumerate(files)}
ia = np.array([pos[a] for a, _ in pairs])
ib = np.array([pos[b] for _, b in pairs])
feat = pd.read_csv(ROOT / f"exp/analysis/features_{split}.csv").set_index("rel").loc[files]
dur = feat.dur.values
ci = cohort_subset(read_train_list(), per_spk=5)
lengths = json.loads((ROOT / "lists/train_lengths.json").read_text())
cdur = np.array([lengths[r] / 16000 for _, r in ci])


def scores(model):
    emb = l2n(np.load(cache_key(model, split, files, mode="full")))
    coh = l2n(np.load(cache_key(model, "traincohort5", [r for _, r in ci], mode="full")))
    return QMF(use_asnorm=False).fit(coh, [s for s, _ in ci], cdur).score(emb, dur, ia, ib)


def eer_thr(s):
    o = np.argsort(s)
    frr = np.cumsum(y[o]) / y.sum()
    far = 1 - np.cumsum(1 - y[o]) / (1 - y).sum()
    return s[o][np.argmin(np.abs(frr - far))]


name = lambda m: Path(m).parent.name  # noqa: E731
sr, sn = scores(ref), scores(new)
tr, tn = eer_thr(sr), eer_thr(sn)
ok_r = (sr > tr) == (y == 1)
ok_n = (sn > tn) == (y == 1)
print(f"# {name(new)} vs {name(ref)} on {split}  ({len(y)} trials)\n")
print(f"EER: {name(ref)} {compute_eer(sr, y):.3f}  ->  {name(new)} {compute_eer(sn, y):.3f}\n")

lab = np.where(y == 1, "target", "non-target")
t = pd.DataFrame({"type": lab, "fixed": ~ok_r & ok_n, "broken": ok_r & ~ok_n, "both_wrong": ~ok_r & ~ok_n})
print("Decisions at each system's own EER threshold (errors fixed / introduced by the new system):")
print(t.groupby("type")[["fixed", "broken", "both_wrong"]].sum().to_string(), "\n")

snr = feat.snr_db.values
sp = feat.speech_dur.values
d = pd.DataFrame({"y": y, "ref": sr, "new": sn,
                  "min_snr": np.minimum(snr[ia], snr[ib]), "min_speech": np.minimum(sp[ia], sp[ib])})
for col, unit, bins in (("min_snr", "dB", [-99, 5, 10, 15, 20, 30, 99]), ("min_speech", "s", [0, 1, 2, 4, 8, 999])):
    d["bin"] = pd.cut(d[col], bins)
    rows = []
    for b, g in d.groupby("bin", observed=True):
        if g.y.sum() < 20 or (1 - g.y).sum() < 20:
            continue
        rows.append(dict(bin=str(b), trials=len(g), targets=int(g.y.sum()), **{
            f"EER {name(ref)}": compute_eer(g.ref.values, g.y.values), f"EER {name(new)}": compute_eer(g.new.values, g.y.values)}))
    r = pd.DataFrame(rows)
    r["delta"] = r.iloc[:, -1] - r.iloc[:, -2]
    print(f"EER by {col} of the pair ({unit}):")
    print(r.round(3).to_string(index=False), "\n")

# margin to own threshold, signed so that > 0 = correct side; improvement = new margin - ref margin
m_r = np.where(y == 1, sr - tr, tr - sr)
m_n = np.where(y == 1, sn - tn, tn - sn)
ex = pd.DataFrame({"file1": [a for a, _ in pairs], "file2": [b for _, b in pairs], "type": lab,
                   "snr1": snr[ia].round(1), "snr2": snr[ib].round(1), "speech1": sp[ia].round(1), "speech2": sp[ib].round(1),
                   "margin_ref": m_r.round(2), "margin_new": m_n.round(2), "gain": (m_n - m_r).round(2)})
for title, sel, asc in (("Largest fixes (wrong -> right)", ~ok_r & ok_n, False), ("Largest breaks (right -> wrong)", ok_r & ~ok_n, True)):
    for ty in ("target", "non-target"):
        print(f"{title}, {ty} pairs (margin > 0 = correct side of the system's threshold):")
        print(ex[sel & (lab == ty)].sort_values("gain", ascending=asc).head(8).to_string(index=False), "\n")
