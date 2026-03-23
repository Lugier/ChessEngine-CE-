#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/runpod/out-4h}"
SELFPLAY_PGN="${SELFPLAY_PGN:-$OUT_DIR/selfplay.pgn}"
SELFPLAY_BIN="${SELFPLAY_BIN:-$OUT_DIR/selfplay.binpack}"
TRAIN_BIN="${TRAIN_BIN:-$ROOT/data/processed-full/train.binpack}"
VAL_BIN="${VAL_BIN:-$ROOT/data/processed-full/val.binpack}"
BASE_NET="${BASE_NET:-$ROOT/engine/cortex.nnue}"
CAND_NET="${CAND_NET:-$OUT_DIR/cortex-selfplay.nnue}"
CUTECHESS_BIN="${CUTECHESS_BIN:-cutechess-cli}"
PY="${PYTHON_BIN:-python3}"
LOG="$OUT_DIR/train-4h.log"

SELFPLAY_GAMES="${SELFPLAY_GAMES:-1200}"
SELFPLAY_TC="${SELFPLAY_TC:-1+0.01}"
SELFPLAY_CONCURRENCY="${SELFPLAY_CONCURRENCY:-48}"
SELFPLAY_THREADS="${SELFPLAY_THREADS:-1}"
SELFPLAY_HASH="${SELFPLAY_HASH:-64}"
SELFPLAY_POSITIONS="${SELFPLAY_POSITIONS:-400000}"
SELFPLAY_RATIO="${SELFPLAY_RATIO:-0.25}"

mkdir -p "$OUT_DIR"
echo "== 4h selfplay run start $(date -u) ==" | tee -a "$LOG"

if [[ ! -x "$ROOT/engine/cortex" ]]; then
  echo "missing engine binary: $ROOT/engine/cortex" | tee -a "$LOG"
  exit 1
fi
if [[ ! -f "$TRAIN_BIN" || ! -f "$VAL_BIN" ]]; then
  echo "missing train/val binpacks ($TRAIN_BIN, $VAL_BIN)" | tee -a "$LOG"
  exit 1
fi

echo "[stage] selfplay generation" | tee -a "$LOG"
OUT="$SELFPLAY_PGN" \
GAMES="$SELFPLAY_GAMES" \
TC="$SELFPLAY_TC" \
CONCURRENCY="$SELFPLAY_CONCURRENCY" \
BASE_THREADS="$SELFPLAY_THREADS" \
CAND_THREADS="$SELFPLAY_THREADS" \
BASE_HASH="$SELFPLAY_HASH" \
CAND_HASH="$SELFPLAY_HASH" \
BASE_NET="$BASE_NET" \
CAND_NET="$BASE_NET" \
BASE_ENGINE="$ROOT/engine/cortex" \
CAND_ENGINE="$ROOT/engine/cortex" \
"$ROOT/scripts/cutechess_match.sh" | tee -a "$LOG"

echo "[stage] pgn -> selfplay binpack" | tee -a "$LOG"
"$PY" "$ROOT/data/pgn_to_binpack.py" \
  --pgn "$SELFPLAY_PGN" \
  --out "$SELFPLAY_BIN" \
  --max-positions "$SELFPLAY_POSITIONS" | tee -a "$LOG"

echo "[stage] 4h finetune" | tee -a "$LOG"
timeout 3h55m "$PY" -u "$ROOT/trainer/train_nnue.py" \
  --data "$TRAIN_BIN" \
  --val-data "$VAL_BIN" \
  --aux-data "$SELFPLAY_BIN" \
  --aux-ratio "$SELFPLAY_RATIO" \
  --epochs "${EPOCHS:-48}" \
  --batch-size "${BATCH_SIZE:-6144}" \
  --num-workers "${NUM_WORKERS:-28}" \
  --max-rows "${MAX_TRAIN_ROWS:-1200000}" \
  --val-max-rows "${VAL_MAX_ROWS:-150000}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --scheduler "${SCHEDULER:-cosine}" \
  --loss "${LOSS_FN:-kldiv}" \
  --feature-mode "${FEATURE_MODE:-kingbucket}" \
  --out "$CAND_NET" | tee -a "$LOG"

cp "$CAND_NET" "$ROOT/engine/cortex.nnue"

echo "[stage] quick sprt gate" | tee -a "$LOG"
CAND_NET="$ROOT/engine/cortex.nnue" \
CUTECHESS_BIN="$CUTECHESS_BIN" \
MATCH_OUT="$OUT_DIR/match-4h.pgn" \
GAMES="${EVAL_GAMES:-200}" \
TC="${EVAL_TC:-40/4+0.04}" \
ALPHA="${SPRT_ALPHA:-0.05}" \
BETA="${SPRT_BETA:-0.05}" \
ELO0="${SPRT_ELO0:-0}" \
ELO1="${SPRT_ELO1:-5}" \
"$ROOT/scripts/sprt.sh" | tee -a "$LOG"

echo "== 4h selfplay run done $(date -u) ==" | tee -a "$LOG"
