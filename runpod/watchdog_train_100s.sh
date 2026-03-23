#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/Chess"
OUT_DIR="${OUT_DIR:-$ROOT/runpod/out-full}"
TRAIN_LOG="${TRAIN_LOG:-$OUT_DIR/launcher-maxeff4.log}"
WATCH_LOG="${WATCH_LOG:-$OUT_DIR/watchdog-100s.log}"
INTERVAL_SEC="${INTERVAL_SEC:-100}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_COUNT_FILE="${RESTART_COUNT_FILE:-$OUT_DIR/watchdog-restarts.count}"

mkdir -p "$OUT_DIR"
touch "$WATCH_LOG"
if [[ ! -f "$RESTART_COUNT_FILE" ]]; then
  echo 0 > "$RESTART_COUNT_FILE"
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$WATCH_LOG"
}

restart_training() {
  local count
  count="$(<"$RESTART_COUNT_FILE")"
  if [[ "$count" -ge "$MAX_RESTARTS" ]]; then
    log "restart limit reached ($count/$MAX_RESTARTS) - not restarting"
    return 1
  fi
  count=$((count + 1))
  echo "$count" > "$RESTART_COUNT_FILE"

  pkill -f "$ROOT/trainer/train_nnue.py" || true
  pkill -f "$ROOT/runpod/train_30h.sh" || true
  sleep 2

  log "restarting training (attempt $count/$MAX_RESTARTS)"
  nohup env \
    PYTHON_BIN="$ROOT/.venv/bin/python" \
    DATA_PYTHON_BIN="$ROOT/.venv/bin/python" \
    STRICT_FULL_RUN=1 \
    AUTO_PREP=0 \
    INSTALL_RUNTIME_DEPS=0 \
    DATA_DIR="$ROOT/data/processed-full" \
    OUT_DIR="$OUT_DIR" \
    FEATURE_MODE=kingbucket \
    EPOCHS=120 \
    SAVE_EVERY=5 \
    BATCH_SIZE=4096 \
    NUM_WORKERS=32 \
    MAX_TRAIN_ROWS=3000000 \
    VAL_MAX_ROWS=100000 \
    PREFETCH_FACTOR=4 \
    LOG_INTERVAL=100 \
    VAL_EVERY=10 \
    PREP_WORKERS=64 \
    "$ROOT/runpod/train_30h.sh" > "$TRAIN_LOG" 2>&1 < /dev/null &
  log "restart launched pid=$!"
}

has_train_proc() {
  pgrep -f "$ROOT/trainer/train_nnue.py" >/dev/null 2>&1
}

log "watchdog started interval=${INTERVAL_SEC}s"
while true; do
  if has_train_proc; then
    gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | tr -d '\r')"
    log "ok train_proc=1 gpu=${gpu_line}"
  else
    log "train process missing - checking log for failure"
    if [[ -f "$TRAIN_LOG" ]] && rg -n "Killed|Traceback|RuntimeError|CUDA out of memory|BrokenPipeError" "$TRAIN_LOG" >/dev/null 2>&1; then
      log "failure pattern detected in train log"
      restart_training || true
    else
      log "no failure pattern found; restarting defensively"
      restart_training || true
    fi
  fi
  sleep "$INTERVAL_SEC"
done
