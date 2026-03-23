#!/usr/bin/env bash
# Volle Verifikation: Build, Perft-Stufen, UCI, Python-Dataprep, optional Trainer.
# Exit-Code != 0 bei jedem Fehler (für CI / lokale Kontrolle).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EXE="$ROOT/engine/cortex"

echo "=== Cortex verify (ROOT=$ROOT) ==="

"$ROOT/scripts/build.sh"

perft_expect() {
  local d="$1" want="$2"
  local got
  got="$("$EXE" perft "$d")"
  if [[ "$got" != "$want" ]]; then
    echo "FAIL perft $d: expected $want got $got" >&2
    exit 1
  fi
  echo "OK perft $d = $got"
}

perft_expect 1 20
perft_expect 2 400
perft_expect 3 8902
perft_expect 4 197281
perft_expect 5 4865609

if ! printf 'uci\nisready\nquit\n' | "$EXE" | grep -q uciok; then
  echo "FAIL UCI uciok" >&2
  exit 1
fi
echo "OK UCI handshake"

if ! printf 'uci\nisready\nposition startpos moves e2e4 e7e5\nposition startpos moves e2e4\nquit\n' | "$EXE" >/dev/null; then
  echo "FAIL UCI position replay" >&2
  exit 1
fi
echo "OK UCI position commands"

mkdir -p "$ROOT/data/processed"
python3 "$ROOT/data/prepare_binpack.py" \
  "$ROOT/data/sample_quiet.txt" \
  "$ROOT/data/processed/verify.binpack"
echo "OK prepare_binpack"

if ! python3 -m py_compile "$ROOT/data/prepare_binpack.py" "$ROOT/trainer/train_nnue.py"; then
  echo "FAIL py_compile" >&2
  exit 1
fi
echo "OK Python syntax"

if [[ "${SKIP_TRAINER:-}" == "1" ]]; then
  echo "SKIP trainer (SKIP_TRAINER=1)"
  echo "=== VERIFY OK (ohne Trainer) ==="
  exit 0
fi

PY="$ROOT/trainer/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Creating trainer venv (needs network once)..."
  python3 -m venv "$ROOT/trainer/.venv"
  "$ROOT/trainer/.venv/bin/pip" install -q -r "$ROOT/trainer/requirements.txt"
fi
"$PY" "$ROOT/trainer/train_nnue.py" \
  --data "$ROOT/data/processed/verify.binpack" \
  --epochs 12 \
  --out "$ROOT/engine/cortex.nnue"
if ! printf 'uci\nisready\nposition startpos\ngo depth 2\nquit\n' | "$EXE" | grep -q bestmove; then
  echo "FAIL UCI go with nnue" >&2
  exit 1
fi
echo "OK train + UCI search"

echo "=== VERIFY OK (komplett) ==="
