#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/runpod/out-loop}"
STATE_DIR="${STATE_DIR:-$OUT_DIR/state}"
LOG="${LOG:-$OUT_DIR/continuous-loop.log}"
WARMSTART_DONE_FILE="${WARMSTART_DONE_FILE:-$STATE_DIR/warmstart.done}"
CYCLE_FILE="${CYCLE_FILE:-$STATE_DIR/cycle.txt}"
MAX_CYCLES="${MAX_CYCLES:-0}"
SLEEP_BETWEEN_CYCLES_SEC="${SLEEP_BETWEEN_CYCLES_SEC:-10}"

mkdir -p "$OUT_DIR" "$STATE_DIR"
echo "== continuous loop start $(date -u) ==" | tee -a "$LOG"

run_warmstart() {
  if [[ -f "$WARMSTART_DONE_FILE" ]]; then
    echo "warmstart already completed; skipping" | tee -a "$LOG"
    return 0
  fi

  echo "[phase] warmstart on static mixed data" | tee -a "$LOG"
  POSITION_POLICY="${POSITION_POLICY:-mixed}" \
  MIX_ALL="${MIX_ALL:-0.50}" \
  MIX_TACTICAL="${MIX_TACTICAL:-0.30}" \
  MIX_QUIET="${MIX_QUIET:-0.20}" \
  AUTO_POST_EVAL="${WARMSTART_AUTO_POST_EVAL:-0}" \
  AUTO_POST_VERIFY="${WARMSTART_AUTO_POST_VERIFY:-1}" \
  OUT_DIR="${WARMSTART_OUT_DIR:-$ROOT/runpod/out-warmstart}" \
  "$ROOT/runpod/train_30h.sh" | tee -a "$LOG"

  date -u +%Y-%m-%dT%H:%M:%SZ > "$WARMSTART_DONE_FILE"
  echo "warmstart completed" | tee -a "$LOG"
}

next_cycle() {
  local current=0
  if [[ -f "$CYCLE_FILE" ]]; then
    current="$(<"$CYCLE_FILE")"
  fi
  current=$((current + 1))
  echo "$current" > "$CYCLE_FILE"
  echo "$current"
}

run_cycle() {
  local cycle="$1"
  local cycle_out="$OUT_DIR/cycle-$cycle"
  mkdir -p "$cycle_out"
  echo "[phase] selfplay cycle=$cycle" | tee -a "$LOG"
  OUT_DIR="$cycle_out" \
  BASE_NET="${BASE_NET:-$ROOT/engine/cortex.nnue}" \
  CAND_NET="$cycle_out/cortex-cand.nnue" \
  SPRT_PROMOTE_POLICY="${SPRT_PROMOTE_POLICY:-strict}" \
  "$ROOT/runpod/train_4h_selfplay.sh" | tee -a "$LOG"
  echo "[phase] cycle=$cycle completed" | tee -a "$LOG"
}

run_warmstart

while true; do
  cycle="$(next_cycle)"
  run_cycle "$cycle"
  if [[ "$MAX_CYCLES" -gt 0 && "$cycle" -ge "$MAX_CYCLES" ]]; then
    echo "max cycles reached ($MAX_CYCLES), exiting" | tee -a "$LOG"
    break
  fi
  sleep "$SLEEP_BETWEEN_CYCLES_SEC"
done

echo "== continuous loop done $(date -u) ==" | tee -a "$LOG"
