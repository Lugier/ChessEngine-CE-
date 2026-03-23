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
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi
LOG="$OUT_DIR/train-4h.log"
SPRT_LOG="$OUT_DIR/sprt-4h.log"
BASELINE_NET="$OUT_DIR/baseline-before-4h.nnue"

SELFPLAY_GAMES="${SELFPLAY_GAMES:-20000}"
SELFPLAY_TC="${SELFPLAY_TC:-1+0.01}"
SELFPLAY_CONCURRENCY="${SELFPLAY_CONCURRENCY:-48}"
SELFPLAY_THREADS="${SELFPLAY_THREADS:-1}"
SELFPLAY_HASH="${SELFPLAY_HASH:-64}"
SELFPLAY_POSITIONS="${SELFPLAY_POSITIONS:-2000000}"
SELFPLAY_RATIO="${SELFPLAY_RATIO:-0.40}"
SELFPLAY_FILTER="${SELFPLAY_FILTER:-all}"
SELFPLAY_MIN_PLY="${SELFPLAY_MIN_PLY:-4}"
SELFPLAY_MAX_PLY="${SELFPLAY_MAX_PLY:-200}"
SELFPLAY_PLY_STEP="${SELFPLAY_PLY_STEP:-1}"
SELFPLAY_MIN_OUTPUT_POSITIONS="${SELFPLAY_MIN_OUTPUT_POSITIONS:-200000}"
SPRT_PROMOTE_POLICY="${SPRT_PROMOTE_POLICY:-strict}"

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
if [[ ! -f "$BASE_NET" ]]; then
  echo "missing base net: $BASE_NET" | tee -a "$LOG"
  exit 1
fi

cp "$BASE_NET" "$BASELINE_NET"
echo "baseline snapshot: $BASELINE_NET" | tee -a "$LOG"

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
  --position-filter "$SELFPLAY_FILTER" \
  --min-ply "$SELFPLAY_MIN_PLY" \
  --max-ply "$SELFPLAY_MAX_PLY" \
  --ply-step "$SELFPLAY_PLY_STEP" \
  --strict-legal 1 \
  --min-output-positions "$SELFPLAY_MIN_OUTPUT_POSITIONS" \
  --max-positions "$SELFPLAY_POSITIONS" | tee -a "$LOG"

echo "[stage] 4h finetune" | tee -a "$LOG"
# GNU timeout on some images rejects "3h55m"; seconds are portable (default 6h).
FINETUNE_TIMEOUT_SEC="${FINETUNE_TIMEOUT_SEC:-21600}"
timeout "$FINETUNE_TIMEOUT_SEC" "$PY" -u "$ROOT/trainer/train_nnue.py" \
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

echo "[stage] quick sprt gate" | tee -a "$LOG"
set +e
BASE_NET="$BASELINE_NET" \
CAND_NET="$CAND_NET" \
CUTECHESS_BIN="$CUTECHESS_BIN" \
MATCH_OUT="$OUT_DIR/match-4h.pgn" \
GAMES="${EVAL_GAMES:-200}" \
TC="${EVAL_TC:-40/4+0.04}" \
ALPHA="${SPRT_ALPHA:-0.05}" \
BETA="${SPRT_BETA:-0.05}" \
ELO0="${SPRT_ELO0:-0}" \
ELO1="${SPRT_ELO1:-5}" \
"$ROOT/scripts/sprt.sh" | tee "$SPRT_LOG" | tee -a "$LOG"
SPRT_RC=${PIPESTATUS[0]}
set -e

PROMOTE=0
if [[ "$SPRT_PROMOTE_POLICY" == "strict" ]]; then
  if grep -Eqi "accepted|H1" "$SPRT_LOG"; then
    PROMOTE=1
  fi
else
  if [[ "$SPRT_RC" -eq 0 ]]; then
    PROMOTE=1
  fi
fi

if [[ "$PROMOTE" -eq 1 ]]; then
  cp "$CAND_NET" "$ROOT/engine/cortex.nnue"
  echo "promotion: accepted, engine net updated" | tee -a "$LOG"
else
  cp "$BASELINE_NET" "$ROOT/engine/cortex.nnue"
  echo "promotion: rejected/unclear, keeping baseline net" | tee -a "$LOG"
fi

echo "== 4h selfplay run done $(date -u) promote=$PROMOTE ==" | tee -a "$LOG"
