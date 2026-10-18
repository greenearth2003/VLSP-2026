#!/bin/bash
# Upload the staged handoff folder to the private HF repo, verify every file against SHA256SUMS (presence + size),
# and delete the staged competition zips only if verification passes.
cd /home/jovyan/work/thuc/hf_stage
export HF_HOME=/home/jovyan/work/thuc/.gitcredentials/hf HF_XET_HIGH_PERFORMANCE=1
LOG=/home/jovyan/work/thuc/VLSP-2026/SRED/exp/logs
PY=/home/jovyan/work/thuc/VLSP-2026/SRED/.venv/bin/python
FLAG=/tmp/claude-1000/-home-jovyan-work-thuc/55cd9b09-1eae-4e94-ad6d-46471080ed00/scratchpad/verify_ok
/home/jovyan/work/thuc/VLSP-2026/SRED/.venv/bin/hf upload thucth/edge-device-speaker-recognition . . --repo-type model \
  --exclude ".cache/*" --exclude "data/derived/*" --commit-message "VLSP 2026 SRED handoff: code, data, key checkpoints, caches, submissions" \
  > $LOG/hf_upload.log 2>&1
echo "upload exit $?"
$PY - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info("thucth/edge-device-speaker-recognition", files_metadata=True)
remote = {s.rfilename: (s.size or 0) for s in info.siblings}
local = {}
for line in open("SHA256SUMS"):
    _, p = line.split(None, 1)
    p = p.strip()
    p = p[2:] if p.startswith("./") else p
    local[p] = os.path.getsize(p)
local["SHA256SUMS"] = os.path.getsize("SHA256SUMS")
missing = [p for p in local if p not in remote]
bad = [p for p in local if p in remote and remote[p] != local[p]]
print(f"VERIFY remote {len(remote)} local {len(local)} missing {len(missing)} size_mismatch {len(bad)}")
for p in missing[:30]:
    print("  MISSING", p)
for p in bad[:30]:
    print("  SIZE", p)
open("/tmp/claude-1000/-home-jovyan-work-thuc/55cd9b09-1eae-4e94-ad6d-46471080ed00/scratchpad/verify_ok", "w").write(
    "1" if not missing and not bad else "0")
PYEOF
if [ "$(cat $FLAG 2>/dev/null)" = "1" ]; then
  rm -f data/Vi-SRED26-*.zip && echo "DELETED staged zips"
else
  echo "NOT deleted (verification failed)"
fi
df -h /home/jovyan/work | tail -1
