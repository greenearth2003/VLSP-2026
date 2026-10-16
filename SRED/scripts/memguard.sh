#!/bin/bash
# Log memory every 30 s; if available RAM < 2 GB, kill the largest python process (training is resumable).
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  cg=$(( $(cat /sys/fs/cgroup/memory.current) / 1048576 ))
  echo "$(date +%T) avail_MB=$avail cgroup_MB=$cg nproc_py=$(pgrep -c python)" >> exp/logs/mem.log
  if [ "$avail" -lt 2048 ]; then
    pid=$(ps -eo pid,rss,cmd --sort=-rss | awk '/python/ && !/jupyter|ipykernel/ {print $1; exit}')
    echo "$(date +%T) LOW MEMORY -> kill $pid" >> exp/logs/mem.log; kill -9 $pid
  fi
  sleep 30
done
