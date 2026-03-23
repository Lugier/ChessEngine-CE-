#!/usr/bin/env bash
set -euo pipefail

LOG=/workspace/Chess/runpod/out-full/monitor-100s.log
mkdir -p /workspace/Chess/runpod/out-full

echo "=== monitor start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
for i in $(seq 1 144); do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$ts] tick=$i/144" >> "$LOG"
  ps -eo pid,stat,etime,%cpu,%mem,cmd | awk '/prepare_lichess_quiet.py|train_nnue.py|runpod\/train_30h.sh/ && !/awk/ {print}' >> "$LOG"
  echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits | tr -d '\r')" >> "$LOG"
  gp=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d '\r')
  if [[ -z "$gp" ]]; then
    gp="none"
  fi
  echo "gpu_procs: $gp" >> "$LOG"
  echo "sizes:" >> "$LOG"
  du -sh /workspace/Chess/data/processed-full /workspace/Chess/runpod/out-full 2>/dev/null >> "$LOG"
  echo "processed-full files:" >> "$LOG"
  ls -lah /workspace/Chess/data/processed-full 2>/dev/null >> "$LOG"
  echo "checkpoints tail:" >> "$LOG"
  ls -1 /workspace/Chess/runpod/out-full/checkpoints 2>/dev/null | tail -n 12 >> "$LOG"
  echo "log tail:" >> "$LOG"
  tail -n 20 /workspace/Chess/runpod/out-full/training.log 2>/dev/null >> "$LOG"
  echo "---" >> "$LOG"
  sleep 100
done
echo "=== monitor end $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
