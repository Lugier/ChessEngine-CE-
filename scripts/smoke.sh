#!/usr/bin/env bash
# Vollständiger Kurztest: Engine bauen, Perft, UCI, Dataprep, optional Trainer.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== build =="
"$ROOT/scripts/build.sh"

echo "== perft 5 (expect 4865609) =="
n="$("$ROOT/engine/cortex" perft 5)"
if [[ "$n" != "4865609" ]]; then
  echo "perft mismatch: got $n" >&2
  exit 1
fi
echo "$n"

echo "== uci =="
printf 'uci\nisready\nquit\n' | "$ROOT/engine/cortex" | grep -q uciok

echo "== dataprep =="
mkdir -p "$ROOT/data/processed"
python3 "$ROOT/data/prepare_binpack.py" \
  "$ROOT/data/sample_quiet.txt" \
  "$ROOT/data/processed/sample.binpack"

if [[ "${SKIP_TRAINER:-}" == "1" ]]; then
  echo "== trainer (skipped SKIP_TRAINER=1) =="
  exit 0
fi

PY="$ROOT/trainer/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "== trainer venv anlegen (einmalig, braucht Netz für pip) =="
  python3 -m venv "$ROOT/trainer/.venv"
  "$ROOT/trainer/.venv/bin/pip" install -q -r "$ROOT/trainer/requirements.txt"
  PY="$ROOT/trainer/.venv/bin/python"
fi

echo "== train nnue (kurz) =="
"$PY" "$ROOT/trainer/train_nnue.py" \
  --data "$ROOT/data/processed/sample.binpack" \
  --epochs 15 \
  --out "$ROOT/engine/cortex.nnue"

echo "== uci mit nnue =="
printf 'uci\nisready\nposition startpos\ngo depth 2\nquit\n' | "$ROOT/engine/cortex" | grep -q bestmove

echo "OK — alles ausführbar."
