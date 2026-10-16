#!/bin/bash
# Download doof-ferb/Vietnam-Celeb (unofficial mirror, 44 parquet shards, ~21.5 GB) and unpack to wav.
# Output: $OUT/wav/<split>/<idXXXXX>/<NNNNN>.wav  and  lists/vc_<split>.txt ("vc_<id> <split>/<id>/<file>")
set -e
cd "$(dirname "$0")/.."
OUT=/home/jovyan/data/ext/vietnam_celeb
mkdir -p $OUT
.venv/bin/python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); out = "/home/jovyan/data/ext/vietnam_celeb"
files = sorted(f for f in api.list_repo_files("doof-ferb/Vietnam-Celeb", repo_type="dataset")
               if f.endswith((".parquet", ".csv", ".md")))
for i, f in enumerate(files):
    hf_hub_download("doof-ferb/Vietnam-Celeb", f, repo_type="dataset", local_dir=out)
    print(f"[{i+1}/{len(files)}] {f}", flush=True)
PY
/opt/conda/bin/python - <<'PY'
import glob, os, pyarrow.parquet as pq
out = "/home/jovyan/data/ext/vietnam_celeb"
lists = {"train": [], "test": []}
for p in sorted(glob.glob(f"{out}/data/*.parquet")):
    split = os.path.basename(p).split("-")[0]
    for batch in pq.ParquetFile(p).iter_batches(batch_size=256):
        for a, rel in zip(batch.column("audio").to_pylist(), batch.column("original_file_path").to_pylist()):
            dst = f"{out}/wav/{split}/{rel}"
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                open(dst, "wb").write(a["bytes"])
            lists[split].append(f"vc_{rel.split('/')[0]} {split}/{rel}")
    print("unpacked", p, flush=True)
lst = "/home/jovyan/code/VLSP-2026/SRED/lists"
for s, rows in lists.items():
    open(f"{lst}/vc_{s}.txt", "w").write("\n".join(rows) + "\n")
    print(s, len(rows), "utts", len({r.split()[0] for r in rows}), "speakers")
print("VC_DONE")
PY
