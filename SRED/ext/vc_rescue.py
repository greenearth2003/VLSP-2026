"""Rescue VC speakers flagged as mixed-voice on 10 samples: embed ALL their utterances (pretrained M),
2-means; keep the dominant cluster if it holds >= 60 % of the utterances, else drop the speaker.
Appends kept utterances to lists/train_vlsp_vc.txt (idempotent: rewrites from base + clean VC + rescued)."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sred.embed import cache_key, extract_cached  # noqa: E402
from sred.models import load_model  # noqa: E402
from sred.scoring import l2n  # noqa: E402

VC = Path("/home/jovyan/data/ext/vietnam_celeb/wav")
rows = []
for split in ["train", "test"]:
    for line in (ROOT / f"lists/vc_{split}.txt").read_text().splitlines():
        if line.strip():
            s, rel = line.split(); rows.append((s, str(VC / rel)))
by = defaultdict(list)
for s, p in rows:
    by[s].append(p)
rng = random.Random(0); sample = []
for s in sorted(by):
    ps = sorted(by[s]); rng.shuffle(ps); sample += [(s, p) for p in ps[:10]]
E10 = l2n(np.load(cache_key("redimnet_M_mix", "vc_sample10", [p for _, p in sample], mode="full")))
sspk = np.array([s for s, _ in sample])
flag = []
for s in sorted(by):
    X = E10[sspk == s]
    if len(X) < 8: continue
    km = KMeans(2, n_init=3, random_state=0).fit(X); c = l2n(km.cluster_centers_)
    if c[0] @ c[1] < 0.5 and np.bincount(km.labels_).min() / len(X) >= 0.15:
        flag.append(s)
stats = json.loads((ROOT / "lists/train_vlsp_vc_stats.json").read_text())
leak = set(stats.get("leak_all", []))
files = [p for s in flag for p in by[s]]
E = l2n(extract_cached("redimnet_M_mix", lambda: load_model("redimnet_M_mix"), "vc_flagged_all", files, Path("/"), mode="full"))
fspk = np.array([s for s in flag for _ in by[s]])
rescued, n_resc, dropped = [], 0, 0
for s in flag:
    m = fspk == s; X = E[m]; F = np.array(files)[m]
    km = KMeans(2, n_init=3, random_state=0).fit(X); cnt = np.bincount(km.labels_, minlength=2)
    if cnt.max() / len(X) >= 0.6:
        rescued += [f"{s} {p}" for p in F[km.labels_ == cnt.argmax()]]; n_resc += 1
    else:
        dropped += 1
lst = ROOT / "lists/train_vlsp_vc.txt"
lines = [l for l in lst.read_text().splitlines() if l.strip()]
have = set(lines)
lines += [r for r in rescued if r not in have]
lst.write_text("\n".join(lines) + "\n")
stats.update(rescue_flagged=len(flag), rescue_kept_speakers=n_resc, rescue_dropped_speakers=dropped,
             rescue_kept_utts=len(rescued), total_utts_after_rescue=len(lines))
(ROOT / "lists/train_vlsp_vc_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps({k: v for k, v in stats.items() if k != "leak_examples"}, indent=1))
