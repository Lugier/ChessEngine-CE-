#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/runpod/out}"
CKPT_DIR="${CKPT_DIR:-$OUT_DIR/checkpoints}"
LAST_CKPT="$(ls -1 "$CKPT_DIR"/epoch-*.pt 2>/dev/null | sort | tail -1 || true)"

if [[ -z "$LAST_CKPT" ]]; then
  echo "no checkpoint found in $CKPT_DIR" >&2
  exit 1
fi

echo "resuming from $LAST_CKPT"
last_epoch="$(basename "$LAST_CKPT" | sed -E 's/epoch-0*([0-9]+)\.pt/\1/')"
target_epochs="${EPOCHS:-120}"
if [[ "$last_epoch" -ge "$target_epochs" ]]; then
  target_epochs=$((last_epoch + 20))
fi
RESUME="$LAST_CKPT" EPOCHS="$target_epochs" "$ROOT/runpod/train_30h.sh"
