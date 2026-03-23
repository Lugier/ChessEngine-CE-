#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/Chess"
OUT_DIR="${OUT_DIR:-$ROOT/runpod/out-full}"
LOG="${LOG:-$OUT_DIR/monitor-100s-forever.log}"
INTERVAL_SEC="${INTERVAL_SEC:-100}"

mkdir -p "$OUT_DIR"
echo "=== monitor_forever start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"

tick=0
while true; do
  tick=$((tick + 1))
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "[$ts] tick=$tick"
    echo "procs:"
    ps -eo pid,stat,etime,%cpu,%mem,cmd | awk '/prepare_lichess_quiet.py|train_nnue.py|runpod\/train_30h.sh|verify.sh|cutechess|bayeselo/ && !/awk/ {print}'
    echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits | tr -d '\r')"
    gp="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d '\r' || true)"
    if [[ -z "${gp// }" ]]; then
      gp="none"
    fi
    echo "gpu_procs: $gp"
    echo "sizes:"
    du -sh "$ROOT/data/processed-full" "$OUT_DIR" 2>/dev/null || true
    echo "checkpoints:"
    ls -1 "$OUT_DIR/checkpoints" 2>/dev/null | tail -n 20 || true
    echo "training.log tail:"
    tail -n 30 "$OUT_DIR/training.log" 2>/dev/null || true
    echo "---"
  } >> "$LOG"
  sleep "$INTERVAL_SEC"
done
