#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-gmemory@10.176.40.144}"
ROOT="${ROOT:-/home/gmemory/lxy/计算神经学final_task}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PART_DIR="$LOCAL_ROOT/data_cn"
LOG_DIR="$LOCAL_ROOT/logs"

mkdir -p "$LOG_DIR"

ssh -o ClearAllForwardings=yes "$REMOTE" "mkdir -p '$ROOT/data_parts' '$ROOT/logs'"

upload_part() {
  local file="$1"
  local base
  local local_size
  local remote_size
  local final_size

  base="$(basename "$file")"
  local_size="$(stat -f%z "$file")"
  remote_size="$(ssh -o ClearAllForwardings=yes "$REMOTE" "stat -c%s '$ROOT/data_parts/$base' 2>/dev/null || echo 0")"

  if [ "$remote_size" -gt "$local_size" ]; then
    echo "[$(date)] $base remote file is larger than local; removing remote copy"
    ssh -o ClearAllForwardings=yes "$REMOTE" "rm -f '$ROOT/data_parts/$base'"
    remote_size=0
  fi

  echo "[$(date)] $base remote $remote_size / local $local_size"
  if [ "$remote_size" -lt "$local_size" ]; then
    tail -c +"$((remote_size + 1))" "$file" |
      ssh -o ClearAllForwardings=yes "$REMOTE" "cat >> '$ROOT/data_parts/$base'"
  fi

  final_size="$(ssh -o ClearAllForwardings=yes "$REMOTE" "stat -c%s '$ROOT/data_parts/$base'")"
  echo "[$(date)] $base final $final_size / local $local_size"
  if [ "$final_size" -ne "$local_size" ]; then
    echo "Size mismatch for $base" >&2
    exit 1
  fi
}

for part in "$PART_DIR"/cn_project.tar.part_*; do
  upload_part "$part"
done

echo "[$(date)] All parts uploaded; running full server pipeline"
ssh -o ClearAllForwardings=yes "$REMOTE" "cd '$ROOT' && nohup scripts/run_full_after_upload.sh > logs/full_pipeline.nohup.log 2>&1 & echo \$! > logs/full_pipeline.pid"
echo "[$(date)] Server pipeline started. Monitor with:"
echo "ssh -o ClearAllForwardings=yes $REMOTE 'tail -f $ROOT/logs/full_run.log'"
