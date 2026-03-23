#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data/processed}"
OUT_DIR="${OUT_DIR:-$ROOT/runpod/out}"
CKPT_DIR="${CKPT_DIR:-$OUT_DIR/checkpoints}"
LOG="$OUT_DIR/training.log"
EPOCHS="${EPOCHS:-120}"
SAVE_EVERY="${SAVE_EVERY:-10}"
TRAIN_BIN="${TRAIN_BIN:-$DATA_DIR/train.binpack}"
VAL_BIN="${VAL_BIN:-$DATA_DIR/val.binpack}"
OUT_NET="${OUT_NET:-$OUT_DIR/cortex.nnue}"
PY="${PYTHON_BIN:-python3}"
RESUME_ARG=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARG=(--resume "$RESUME")
fi

mkdir -p "$OUT_DIR" "$CKPT_DIR"
echo "== train_30h: start $(date -u) ==" | tee -a "$LOG"
echo "train=$TRAIN_BIN val=$VAL_BIN out=$OUT_NET" | tee -a "$LOG"

if [[ ! -f "$TRAIN_BIN" ]]; then
  echo "missing train binpack: $TRAIN_BIN" >&2
  exit 1
fi
if [[ ! -f "$VAL_BIN" && "${ALLOW_MISSING_VAL:-0}" != "1" ]]; then
  echo "missing val binpack: $VAL_BIN (set ALLOW_MISSING_VAL=1 to bypass)" >&2
  exit 1
fi

if [[ "${SKIP_PREFLIGHT_VERIFY:-0}" != "1" ]]; then
  SKIP_TRAINER=1 "$ROOT/scripts/verify.sh"
fi

"$PY" "$ROOT/trainer/train_nnue.py" \
  --data "$TRAIN_BIN" \
  --val-data "$VAL_BIN" \
  --epochs "$EPOCHS" \
  --save-every "$SAVE_EVERY" \
  --checkpoint-dir "$CKPT_DIR" \
  "${RESUME_ARG[@]}" \
  --out "$OUT_NET" | tee -a "$LOG"

if [[ "$OUT_NET" != "$ROOT/engine/cortex.nnue" ]]; then
  cp "$OUT_NET" "$ROOT/engine/cortex.nnue"
fi
echo "== train_30h: done $(date -u) ==" | tee -a "$LOG"
