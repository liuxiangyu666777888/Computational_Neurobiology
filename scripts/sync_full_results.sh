#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-gmemory@10.176.40.144}"
ROOT="${ROOT:-/home/gmemory/lxy/计算神经学final_task}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$LOCAL_ROOT"
mkdir -p runs

rsync -av -e 'ssh -o ClearAllForwardings=yes' "$REMOTE:$ROOT/report_final.md" ./
rsync -av -e 'ssh -o ClearAllForwardings=yes' "$REMOTE:$ROOT/report_final.pdf" ./ || true
rsync -av -e 'ssh -o ClearAllForwardings=yes' "$REMOTE:$ROOT/splits/" ./splits/
rsync -av -e 'ssh -o ClearAllForwardings=yes' "$REMOTE:$ROOT/runs/denoise_unet/" ./runs/denoise_unet/
rsync -av -e 'ssh -o ClearAllForwardings=yes' "$REMOTE:$ROOT/logs/" ./logs/server/
