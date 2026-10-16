#!/bin/bash
# two-teacher long->short KD: targets = normalised mean of W_G05_B6 and W_C01_B6 full-utterance embeddings
cd /home/jovyan/work/thuc/VLSP-2026/SRED && source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/clean_data.py --models exp/ft/W_C01_B6/avg.pt --train-models exp/ft/W_C01_B6/avg.pt --extract-only > exp/logs/extract_trainfull_W_C01_B6.log 2>&1
PYTHONPATH=. .venv/bin/python - <<'PY'
import numpy as np
from sred.data import read_train_list
from sred.embed import cache_key
files = [r for _, r in read_train_list()]
E = []
for m in ["exp/ft/W_G05_B6/avg.pt", "exp/ft/W_C01_B6/avg.pt"]:
    e = np.load(cache_key(m, "trainfull", files, mode="full"))
    E.append(e / np.linalg.norm(e, axis=1, keepdims=True))
np.save("exp/emb/teacher2_mean_trainfull.npy", ((E[0] + E[1]) / 2).astype(np.float32))
print("cos between teachers (mean):", float((E[0] * E[1]).sum(1).mean()))
PY
WISE_STEP=12000 scripts/c_sweep.sh redimnet_M_mix K25_M stem lists/clean/train_clean.txt \
  --teacher-emb exp/emb/teacher2_mean_trainfull.npy --min-crop-sec 0.75 --temb-kd 2.0 --temb-rkd 20 \
  --steps 18000 --save-every 2000 > exp/logs/sweep_K25_M.out 2>&1
