"""Quick EER on a fixed random subset of public trials (cosine). Usage: quick_subset_eval.py MODEL [N]"""
import sys, time
import numpy as np
from sred.data import PUBLIC_DIR, read_trials, unique_files
from sred.embed import extract
from sred.metrics import compute_eer
from sred.models import load_model
from sred.scoring import cosine_scores

name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
pairs, lab = read_trials()
sel = np.random.default_rng(0).choice(len(pairs), n, replace=False)
pairs = [pairs[i] for i in sel]
lab = lab[sel]
files = unique_files(pairs)
pos = {f: i for i, f in enumerate(files)}
ia = np.array([pos[a] for a, _ in pairs])
ib = np.array([pos[b] for _, b in pairs])
t = time.time()
e = extract(load_model(name), files, PUBLIC_DIR)
print(name, "subset", n, "EER", round(compute_eer(cosine_scores(e, ia, ib), lab), 3), "sec", round(time.time() - t))
