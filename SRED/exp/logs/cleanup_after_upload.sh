#!/bin/bash
# delete the staged zips only if the HF verification reports 0 missing and 0 size mismatches
OUT=/tmp/claude-1000/-home-jovyan-work-thuc/55cd9b09-1eae-4e94-ad6d-46471080ed00/tasks/bqnuqi28v.output
until grep -q "remote files" $OUT 2>/dev/null; do sleep 60; done
if grep -q "missing on remote 0, size mismatches 0" $OUT; then
  rm -f /home/jovyan/work/thuc/hf_stage/data/Vi-SRED26-*.zip && echo "DELETED staged zips" 
else
  echo "NOT deleted: verification failed"; grep -E "MISSING|SIZE|remote files" $OUT
fi
df -h /home/jovyan/work | tail -1
