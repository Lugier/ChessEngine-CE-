#!/usr/bin/env bash
# Gatekeeper: ensures the repo stays shippable. Fails fast on any step.
# 1) Native build (scripts/build.sh)  2) Perft 1..5 from startpos
# 3) UCI smoke  4) prepare_binpack on sample data  5) py_compile
# 6) UCI go with classic eval (UseNNUE false)  7) unless SKIP_TRAINER=1:
#    venv+torch, train_nnue on verify.binpack, export cortex.nnue, UCI go with net.
# See docs/ENGINE.md for what this does / does not prove about search quality.
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
if ! printf 'uci\nisready\nposition fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4\ngo depth 2\nquit\n' |
  "$EXE" | grep -q bestmove; then
  echo "FAIL UCI position fen + go" >&2
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

# Suche muss auch ohne NNUE-Datei laufen (klassische Eval)
if ! printf 'uci\nisready\nsetoption name UseNNUE value false\nposition startpos\ngo depth 3\nquit\n' |
  "$EXE" | grep -q bestmove; then
  echo "FAIL UCI go depth 3 (classic eval, NNUE off)" >&2
  exit 1
fi
echo "OK UCI search (classic eval)"

if [[ -n "${LARGE_NET_FILE:-}" && -f "${LARGE_NET_FILE}" ]]; then
  if ! printf "uci\nisready\nsetoption name EvalFile value ${LARGE_NET_FILE}\nsetoption name UseNNUE value true\nposition startpos\ngo depth 3\nquit\n" |
    "$EXE" | grep -q bestmove; then
    echo "FAIL UCI go with LARGE_NET_FILE=${LARGE_NET_FILE}" >&2
    exit 1
  fi
  echo "OK UCI search (large-net smoke)"
fi

if [[ "${RUN_BENCH:-0}" == "1" ]]; then
  "$ROOT/scripts/bench_strength.sh" 8 >/dev/null
  echo "OK benchmark snapshot"
fi

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
